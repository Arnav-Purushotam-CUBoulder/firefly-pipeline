#!/usr/bin/env python3
"""Stage-1 detection via cuCIM blob detectors (GPU)."""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
import ctypes
import os

os.environ.setdefault('CUPY_DISABLE_JITIFY', '0')
os.environ.setdefault('CUPY_JITIFY_ENABLE', '1')
os.environ.setdefault('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', '1')

import cv2
import numpy as np

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cnd
    from cucim.skimage.feature import blob_log, blob_dog, blob_doh
except Exception as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Stage-1 cuCIM detector requires CuPy and cuCIM. Install e.g. 'pip install cupy-cuda12x cucim'."
    ) from exc


def _ensure_nvrtc_loaded():
    """Ensure libnvrtc from the CUDA 12 runtime wheel is visible before CuPy kernels run."""
    try:
        ctypes.CDLL('libnvrtc.so.12')
        return
    except OSError:
        pass

    try:
        import nvidia.cuda_nvrtc as cuda_nvrtc
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            'CuPy cannot find libnvrtc.so.12. Install the CUDA 12 nvrtc package or export LD_LIBRARY_PATH.'
        ) from exc

    base_dir = getattr(cuda_nvrtc, '__file__', None)
    if base_dir is None:
        base_dir = list(getattr(cuda_nvrtc, '__path__', []))[0]
    base = Path(base_dir).resolve()
    if base.name != 'lib':
        base = base / 'lib'
    candidates = list(base.glob('libnvrtc.so*')) + list(base.glob('libnvrtc-builtins.so*'))
    loaded = False
    for cand in candidates:
        try:
            ctypes.CDLL(str(cand), mode=ctypes.RTLD_GLOBAL)
            loaded = True
        except OSError:
            continue
    if not loaded:
        raise RuntimeError(
            'CuPy could not load libnvrtc runtime components from nvidia-cuda-nvrtc. '
            'Install CUDA toolkit 12.4 or export LD_LIBRARY_PATH to include the nvrtc lib directory.'
        )

_ensure_nvrtc_loaded()


from cupy.cuda import compiler as _cupy_compiler

_original_jitify_prep = getattr(_cupy_compiler, '_jitify_prep', None)
if _original_jitify_prep is not None:

    def _jitify_prep_cxx17(source, options, cu_path):
        options, headers, include_names = _original_jitify_prep(source, options, cu_path)
        # Filter out system and stale toolkit includes; keep Jitify self-contained
        def _ok(opt: str) -> bool:
            if not isinstance(opt, str):
                return True
            if opt.startswith('-I') and ('/usr/local/cuda-' in opt or opt.startswith('-I/usr/include')):
                return False
            return True
        opts = []
        has_std = False
        for opt in options:
            if not _ok(opt):
                continue
            if opt in ('-std=c++11', '--std=c++11'):
                opts.append('--std=c++17')
                has_std = True
            elif isinstance(opt, str) and (opt.startswith('-std=') or opt.startswith('--std=')):
                opts.append(opt)
                has_std = True
            else:
                opts.append(opt)
        if not has_std:
            opts.append('--std=c++17')
        print('[jitify patch] options before return:', tuple(opts))
        return tuple(opts), headers, include_names

    _cupy_compiler._jitify_prep = _jitify_prep_cxx17
    _cupy_compiler._cucim_cxx17_patch = 'v2'
    print('[jitify patch installed]')

_original_compile_using_nvrtc = getattr(_cupy_compiler, 'compile_using_nvrtc', None)
if _original_compile_using_nvrtc is not None and not getattr(_cupy_compiler, '_cucim_nvrtc_patch', False):

    def _compile_using_nvrtc_cxx17(source, options=(), *args, **kwargs):
        # Ensure we don't pass both positional and keyword for 'jitify'
        def _dedupe_jitify(pos_args, kw):
            try:
                if isinstance(kw, dict) and 'jitify' in kw and pos_args and isinstance(pos_args[-1], bool):
                    pos_args = tuple(pos_args[:-1])
            except Exception:
                pass
            return pos_args, kw
        # Helper: filter options that point to system or stale toolkit headers
        def _ok(opt: str) -> bool:
            if not isinstance(opt, str):
                return True
            if opt.startswith('-I') and '/usr/local/cuda-' in opt:
                return False
            if opt.startswith('-I/usr/include'):
                return False
            return True

        # Normalize C++ standard and strip unwanted includes
        opts = []
        has_std = False
        for opt in options:
            if not _ok(opt):
                continue
            # Drop any request to force jitify via macro; we control via function arg
            if opt == '-DCUPY_USE_JITIFY':
                continue
            if opt in ('-std=c++11', '--std=c++11'):
                opts.append('--std=c++17')
                has_std = True
            elif isinstance(opt, str) and (opt.startswith('-std=') or opt.startswith('--std=')):
                opts.append(opt)
                has_std = True
            else:
                opts.append(opt)
        if not has_std:
            opts.append('--std=c++17')

        # Detect caller's jitify flag
        jitify_requested = False
        try:
            if isinstance(kwargs, dict) and 'jitify' in kwargs:
                jitify_requested = bool(kwargs['jitify'])
            elif args and isinstance(args[-1], bool):
                jitify_requested = args[-1]
        except Exception:
            pass

        # Detect ndimage-generated kernels and force Jitify + strip any system includes entirely
        try:
            src_txt = source.decode('utf-8', 'ignore') if isinstance(source, (bytes, bytearray)) else str(source)
        except Exception:
            src_txt = ''
        is_ndimage = ('cupyx' in src_txt and 'ndimage' in src_txt) or ('cupyx_scipy_ndimage' in src_txt)

        if is_ndimage:
            # Force NVRTC (not Jitify) for ndimage kernels, sanitized from system C++ headers
            # Ensure no system include sneaks in; allow CUDA toolkit include for cuda_fp16.h
            opts = [o for o in opts if not (isinstance(o, str) and o.startswith('-I/usr/include'))]
            try:
                candidates = []
                env_cuda = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
                if env_cuda:
                    candidates.append(os.path.join(env_cuda, 'include'))
                candidates += [
                    '/usr/local/cuda/include',
                    '/usr/local/cuda-12.4/include',
                    '/usr/local/cuda-12.3/include',
                    '/usr/local/cuda-12.2/include',
                    '/usr/local/cuda-12.1/include',
                    '/usr/local/cuda-12.0/include',
                    '/usr/local/cuda-11.8/include',
                ]
                for inc_dir in candidates:
                    try:
                        if os.path.isdir(inc_dir) and os.path.exists(os.path.join(inc_dir, 'cuda_fp16.h')):
                            flag = f'-I{inc_dir}'
                            if flag not in opts:
                                opts.append(flag)
                            break
                    except Exception:
                        continue
            except Exception:
                pass
            # Add CuPy's bundled CCCL libcudacxx include dir so cuda_workaround.h can find cuda/std/*
            try:
                import cupy as _cp
                _root = Path(_cp.__file__).resolve().parent
                _libcxx_inc = _root / '_core/include/cupy/_cccl/libcudacxx/include'
                if _libcxx_inc.is_dir():
                    flag = f'-I{str(_libcxx_inc)}'
                    if flag not in opts:
                        opts.append(flag)
            except Exception:
                pass
            for define in ('-D__CUDACC_RTC__=1', '-D__SIZE_TYPE__=unsigned long', '-D__PTRDIFF_TYPE__=long'):
                if define not in opts:
                    opts.append(define)
            # Replace C++ std include with CuPy workaround for plain NVRTC
            src_mod = source
            try:
                import re as _re
                src_text = source.decode('utf-8', 'ignore') if isinstance(source, (bytes, bytearray)) else str(source)
                if '#include <type_traits>' in src_text and '#include <cupy/cuda_workaround.h>' not in src_text:
                    src_text = src_text.replace('#include <type_traits>', '#include <cupy/cuda_workaround.h>  // patched')
                    # Remove specializations that conflict with cuda_workaround aliasing
                    patterns = [
                        r"template\s*<\s*>\s*struct\s+std::is_floating_point\s*<\s*float16\s*>\s*:\s*std::true_type\s*\{\s*\};",
                        r"template\s*<\s*>\s*struct\s+std::is_signed\s*<\s*float16\s*>\s*:\s*std::true_type\s*\{\s*\};",
                        r"template\s*<\s*class\s+T\s*>\s*struct\s+std::is_signed\s*<\s*complex<\s*T\s*>\s*>\s*:\s*std::is_signed<\s*T\s*>\s*\{\s*\};",
                    ]
                    for pat in patterns:
                        src_text = _re.sub(pat, '/* removed for NVRTC */', src_text)
                    src_mod = src_text
            except Exception:
                pass

            # Force jitify=False via binding or sanitized fallback
            try:
                import inspect
                sig = inspect.signature(_original_compile_using_nvrtc)
                ba = sig.bind_partial(src_mod, tuple(opts), *args, **kwargs)
                ba.arguments['jitify'] = False
                print('[nvrtc patch] forcing plain NVRTC for ndimage; options:', tuple(opts))
                try:
                    return _original_compile_using_nvrtc(**ba.arguments)
                except Exception as e:
                    print('[nvrtc path] compilation error:', getattr(e, '__class__', type(e)).__name__, str(e))
                    for attr in ('log', 'stderr', 'nvrtc_log'):
                        log = getattr(e, attr, None)
                        if log:
                            try:
                                txt = str(log)
                                if len(txt) > 20000:
                                    txt = txt[:20000] + '\n... [truncated] ...'
                                print('[nvrtc][log]', txt)
                                break
                            except Exception:
                                pass
                    dump = getattr(e, 'dump', None)
                    if callable(dump):
                        try:
                            dump(sys.stderr)
                        except Exception:
                            pass
                    raise
            except Exception:
                kwargs = dict(kwargs)
                kwargs['jitify'] = False
                args2, kwargs2 = _dedupe_jitify(tuple(args), kwargs)
                print('[nvrtc patch] forcing plain NVRTC for ndimage (fallback); options:', tuple(opts))
                try:
                    return _original_compile_using_nvrtc(src_mod, tuple(opts), *args2, **kwargs2)
                except Exception as e:
                    print('[nvrtc path] compilation error:', getattr(e, '__class__', type(e)).__name__, str(e))
                    for attr in ('log', 'stderr', 'nvrtc_log'):
                        log = getattr(e, attr, None)
                        if log:
                            try:
                                txt = str(log)
                                if len(txt) > 20000:
                                    txt = txt[:20000] + '\n... [truncated] ...'
                                print('[nvrtc][log]', txt)
                                break
                            except Exception:
                                pass
                    dump = getattr(e, 'dump', None)
                    if callable(dump):
                        try:
                            dump(sys.stderr)
                        except Exception:
                            pass
                    raise

        # Non-ndimage path: keep NVRTC self-contained (no system headers), add CUDA toolkit include + typedefs
        try:
            candidates = []
            env_cuda = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
            if env_cuda:
                candidates.append(os.path.join(env_cuda, 'include'))
            candidates += [
                '/usr/local/cuda/include',
                '/usr/local/cuda-12.4/include',
                '/usr/local/cuda-12.3/include',
                '/usr/local/cuda-12.2/include',
                '/usr/local/cuda-12.1/include',
                '/usr/local/cuda-12.0/include',
                '/usr/local/cuda-11.8/include',
            ]
            for inc_dir in candidates:
                try:
                    if os.path.isdir(inc_dir) and os.path.exists(os.path.join(inc_dir, 'cuda_fp16.h')):
                        flag = f'-I{inc_dir}'
                        if flag not in opts:
                            opts.append(flag)
                        break
                except Exception:
                    continue
        except Exception:
            pass
        # Add CuPy's bundled CCCL libcudacxx include dir so cuda_workaround.h can find cuda/std/*
        try:
            import cupy as _cp
            _root = Path(_cp.__file__).resolve().parent
            _libcxx_inc = _root / '_core/include/cupy/_cccl/libcudacxx/include'
            if _libcxx_inc.is_dir():
                flag = f'-I{str(_libcxx_inc)}'
                if flag not in opts:
                    opts.append(flag)
        except Exception:
            pass

        # Minimal builtin typedefs to satisfy libcudacxx expectations under NVRTC
        for define in ('-D__CUDACC_RTC__=1', '-D__SIZE_TYPE__=unsigned long', '-D__PTRDIFF_TYPE__=long'):
            if define not in opts:
                opts.append(define)

        print('[nvrtc patch] using options:', tuple(opts))
        try:
            import inspect
            sig = inspect.signature(_original_compile_using_nvrtc)
            ba = sig.bind_partial(source, tuple(opts), *args, **kwargs)
            try:
                return _original_compile_using_nvrtc(**ba.arguments)
            except Exception as e:
                print('[nvrtc path] compilation error:', getattr(e, '__class__', type(e)).__name__, str(e))
                for attr in ('log', 'stderr', 'nvrtc_log'):
                    log = getattr(e, attr, None)
                    if log:
                        try:
                            txt = str(log)
                            if len(txt) > 20000:
                                txt = txt[:20000] + '\n... [truncated] ...'
                            print('[nvrtc][log]', txt)
                            break
                        except Exception:
                            pass
                dump = getattr(e, 'dump', None)
                if callable(dump):
                    try:
                        dump(sys.stderr)
                    except Exception:
                        pass
                raise
        except Exception:
            args2, kwargs2 = _dedupe_jitify(tuple(args), dict(kwargs))
            try:
                return _original_compile_using_nvrtc(source, tuple(opts), *args2, **kwargs2)
            except Exception as e:
                print('[nvrtc path] compilation error:', getattr(e, '__class__', type(e)).__name__, str(e))
                for attr in ('log', 'stderr', 'nvrtc_log'):
                    log = getattr(e, attr, None)
                    if log:
                        try:
                            txt = str(log)
                            if len(txt) > 20000:
                                txt = txt[:20000] + '\n... [truncated] ...'
                            print('[nvrtc][log]', txt)
                            break
                        except Exception:
                            pass
                dump = getattr(e, 'dump', None)
                if callable(dump):
                    try:
                        dump(sys.stderr)
                    except Exception:
                        pass
                raise

    _cupy_compiler.compile_using_nvrtc = _compile_using_nvrtc_cxx17
    _cupy_compiler._cucim_nvrtc_patch = True

BAR_LEN = 50
_NVRTC_LOG_PRINTED = False


def progress(i: int, total: int, tag: str = "") -> None:
    frac = i / total if total else 0.0
    bar = "=" * int(frac * BAR_LEN) + " " * (BAR_LEN - int(frac * BAR_LEN))
    sys.stdout.write(f"\r{tag} [{bar}] {int(frac * 100):3d}%")
    sys.stdout.flush()
    if i == total:
        sys.stdout.write("\n")


def _make_clahe(use: bool, clip: float, tile: int | tuple[int, int]):
    if not use:
        return None
    if isinstance(tile, (tuple, list)):
        tx, ty = tile
    else:
        tx = ty = tile
    tx = max(1, int(tx))
    ty = max(1, int(ty))
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tx, ty))


def _apply_clahe(gray: np.ndarray, clahe) -> np.ndarray:
    if clahe is None:
        return gray
    return clahe.apply(gray)


def _sigma_to_radius(detector: str, sigma_val: float, log_scale: bool) -> float:
    name = detector.lower()
    if name == "log":
        return float(sigma_val) if not log_scale else float(sigma_val) * math.sqrt(2.0)
    if name == "dog":
        return float(sigma_val) * math.sqrt(2.0)
    return float(sigma_val)


def _clip_box(x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x + w > frame_w:
        w = frame_w - x
    if y + h > frame_h:
        h = frame_h - y
    return x, y, max(0, w), max(0, h)


def _fallback_blob_log(
    gpu_img: cp.ndarray,
    *,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
    log_scale: bool,
) -> cp.ndarray:
    """Pure-CuPy LoG blob detector avoiding cuCIM's peak_local_max.
    Returns cp.ndarray of shape (N, 3): (y, x, sigma).
    """
    H, W = gpu_img.shape
    # Build sigma list
    if num_sigma <= 1:
        sigmas = [float(min_sigma)]
    else:
        if log_scale:
            import math as _m
            r = (float(max_sigma) / float(min_sigma)) ** (1.0 / (num_sigma - 1))
            sigmas = [float(min_sigma) * (r ** i) for i in range(num_sigma)]
        else:
            step = (float(max_sigma) - float(min_sigma)) / (num_sigma - 1)
            sigmas = [float(min_sigma) + i * step for i in range(num_sigma)]

    responses = []
    for s in sigmas:
        # normalized LoG response
        R = -cnd.gaussian_laplace(gpu_img, sigma=s)
        R = R * (s * s)
        responses.append(R)

    # Stack scale-space: (S, H, W)
    S = len(responses)
    vol = cp.stack(responses, axis=0)

    # Per-scale 2D NMS via 3x3 max using slicing (ignore 1px border)
    ys = []
    xs = []
    ss = []
    for k, s in enumerate(sigmas):
        R = vol[k]
        # compute absolute threshold relative to frame max at this scale
        thr = float(threshold) * float(cp.max(R).item() if R.size else 1.0)
        if H < 3 or W < 3:
            continue
        c = R[1:-1, 1:-1]
        neigh = [
            R[0:-2, 0:-2], R[0:-2, 1:-1], R[0:-2, 2:  ],
            R[1:-1, 0:-2], R[1:-1, 1:-1], R[1:-1, 2:  ],
            R[2:  , 0:-2], R[2:  , 1:-1], R[2:  , 2:  ],
        ]
        m = neigh[0]
        for n in neigh[1:]:
            m = cp.maximum(m, n)
        mask = (c == m) & (c > thr)
        yy, xx = cp.where(mask)
        if yy.size:
            ys.append(yy + 1)
            xs.append(xx + 1)
            ss.append(cp.full_like(yy, fill_value=s, dtype=cp.float32))

    if not ys:
        return cp.empty((0, 3), dtype=cp.float32)
    ys = cp.concatenate(ys)
    xs = cp.concatenate(xs)
    ss = cp.concatenate(ss)

    return cp.stack([ys.astype(cp.float32), xs.astype(cp.float32), ss], axis=1)


def detect_stage1_to_csv(
    *,
    orig_path: Path,
    csv_path: Path,
    max_frames: int | None,
    detector: str = "log",
    min_sigma: float = 0.8,
    max_sigma: float = 4.0,
    num_sigma: int = 10,
    sigma_ratio: float = 1.6,
    threshold: float = 0.05,
    overlap: float = 0.5,
    log_scale: bool = False,
    min_area_px: float = 1.0,
    max_area_scale: float = 1.0,
    pad_px: int = 0,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tile: int | tuple[int, int] = 8,
) -> None:
    """Detect blobs using cuCIM (GPU) and write bounding boxes to CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(orig_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {orig_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = frame_w * frame_h

    if max_sigma <= min_sigma:
        raise ValueError("max_sigma must be greater than min_sigma for cuCIM detector")
    if detector not in {"log", "dog", "doh"}:
        raise ValueError(f"Unknown cuCIM detector '{detector}' (expected 'log'|'dog'|'doh')")

    max_area_px = frame_area if float(max_area_scale) <= 0 else int(frame_area * float(max_area_scale))
    pad = max(0, int(pad_px))

    clahe = _make_clahe(use_clahe, clahe_clip, clahe_tile)

    rows: list[tuple[int, int, int, int, int]] = []
    frame_idx = 0

    use_fallback_log = False
    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = _apply_clahe(gray, clahe)

        gpu_img = cp.asarray(gray, dtype=cp.float32) / 255.0

        if detector == "log":
            try:
                blobs = blob_log(
                    gpu_img,
                    min_sigma=float(min_sigma),
                    max_sigma=float(max_sigma),
                    num_sigma=int(max(1, num_sigma)),
                    threshold=float(threshold),
                    overlap=float(overlap),
                    log_scale=bool(log_scale),
                )
            except Exception as e:
                global _NVRTC_LOG_PRINTED
                if not _NVRTC_LOG_PRINTED:
                    _NVRTC_LOG_PRINTED = True
                    print("[cuCIM][Stage1] blob_log raised exception:", type(e).__name__, str(e))
                    # Surface any available compile log once to help debug
                    log = getattr(e, 'log', None)
                    if log:
                        try:
                            # Print a trimmed log to avoid flooding the console
                            txt = str(log)
                            if len(txt) > 20000:
                                txt = txt[:20000] + "\n... [truncated] ..."
                            print("[cuCIM][NVRTC log]", txt)
                        except Exception:
                            pass
                    dump = getattr(e, 'dump', None)
                    if callable(dump):
                        try:
                            dump(sys.stderr)
                        except Exception:
                            pass
                raise
        elif detector == "dog":
            blobs = blob_dog(
                gpu_img,
                min_sigma=float(min_sigma),
                max_sigma=float(max_sigma),
                sigma_ratio=float(max(1.01, sigma_ratio)),
                threshold=float(threshold),
                overlap=float(overlap),
            )
        else:  # 'doh'
            blobs = blob_doh(
                gpu_img,
                min_sigma=float(min_sigma),
                max_sigma=float(max_sigma),
                threshold=float(threshold),
                overlap=float(overlap),
            )

        if blobs.size:
            blobs_np = cp.asnumpy(blobs)
            for (yy, xx, ss) in blobs_np:
                radius = max(1.0, _sigma_to_radius(detector, float(ss), log_scale))
                side = max(3, int(round(2.0 * radius)))

                w = side + pad * 2
                h = w
                x0 = int(round(xx - side / 2.0)) - pad
                y0 = int(round(yy - side / 2.0)) - pad
                x0, y0, w, h = _clip_box(x0, y0, w, h, frame_w, frame_h)
                if w <= 0 or h <= 0:
                    continue

                area = w * h
                if area < float(min_area_px) or area > max_area_px:
                    continue

                rows.append((frame_idx, x0, y0, w, h))

        progress(frame_idx + 1, total_frames, 'detect(cuCIM)')
        frame_idx += 1

    cap.release()

    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))

    with csv_path.open('w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['frame', 'x', 'y', 'w', 'h'])
        wtr.writerows(rows)
