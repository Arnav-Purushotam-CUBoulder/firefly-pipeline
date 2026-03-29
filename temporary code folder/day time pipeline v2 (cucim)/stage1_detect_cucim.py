#!/usr/bin/env python3
"""
Day-time Stage-1 detection via cuCIM blob detectors (GPU), adapted from the
night-time pipeline's cuCIM implementation. Writes CSV in the SAME format as
night-time Stage 1 cuCIM:

Output CSV schema (per video):
  frame, x, y, w, h

We process the whole video (up to MAX_FRAMES) in a single pass; no chunking and
no trajectory/global ids are produced here.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
import ctypes
import os

import cv2
import numpy as np

import params
from utils import open_video, progress, make_writer


# Environment knobs for CuPy NVRTC + Jitify; mirror the night-time pipeline
os.environ.setdefault('CUPY_DISABLE_JITIFY', '0')
os.environ.setdefault('CUPY_JITIFY_ENABLE', '1')
os.environ.setdefault('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', '1')

try:
    import cupy as cp
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


# Patch CuPy's NVRTC/Jitify usage to favor C++17 and stable include resolution,
# following the night-time pipeline implementation.
from cupy.cuda import compiler as _cupy_compiler

_original_jitify_prep = getattr(_cupy_compiler, '_jitify_prep', None)
if _original_jitify_prep is not None:

    def _jitify_prep_cxx17(source, options, cu_path):
        options, headers, include_names = _original_jitify_prep(source, options, cu_path)

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
        def _dedupe_jitify(pos_args, kw):
            try:
                if isinstance(kw, dict) and 'jitify' in kw and pos_args and isinstance(pos_args[-1], bool):
                    pos_args = tuple(pos_args[:-1])
            except Exception:
                pass
            return pos_args, kw

        def _ok(opt: str) -> bool:
            if not isinstance(opt, str):
                return True
            if opt.startswith('-I') and '/usr/local/cuda-' in opt:
                return False
            if opt.startswith('-I/usr/include'):
                return False
            return True

        opts = []
        has_std = False
        for opt in options:
            if not _ok(opt):
                continue
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


def _detect_blobs_for_batch(
    gpu_batch: "cp.ndarray",
    *,
    detector: str,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    sigma_ratio: float,
    threshold: float,
    overlap: float,
    log_scale: bool,
):
    """Run cuCIM detector per-slice (API is 2D) and return list of arrays."""
    out = []
    for bi in range(int(gpu_batch.shape[0])):
        img = gpu_batch[bi]
        if detector == "log":
            blobs = blob_log(
                img,
                min_sigma=float(min_sigma),
                max_sigma=float(max_sigma),
                num_sigma=int(max(1, num_sigma)),
                threshold=float(threshold),
                overlap=float(overlap),
                log_scale=bool(log_scale),
            )
        elif detector == "dog":
            blobs = blob_dog(
                img,
                min_sigma=float(min_sigma),
                max_sigma=float(max_sigma),
                sigma_ratio=float(max(1.01, sigma_ratio)),
                threshold=float(threshold),
                overlap=float(overlap),
            )
        else:  # 'doh'
            blobs = blob_doh(
                img,
                min_sigma=float(min_sigma),
                max_sigma=float(max_sigma),
                threshold=float(threshold),
                overlap=float(overlap),
            )
        out.append(blobs)
    return out


def run_for_video(video_path: Path) -> Path:
    """Run cuCIM-based Stage 1 and write a CSV matching the night-time format.

    Returns path to the per-video CSV under STAGE1_DIR/<stem>/<stem>.csv
    """
    assert video_path.exists(), f"Video not found: {video_path}"
    stem = video_path.stem

    # Output path mirrors previous daytime Stage 1
    out_root = params.STAGE1_DIR / stem
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / f"{stem}.csv"

    # Open video and determine dimensions
    cap, frame_w, frame_h, fps, total_frames = open_video(video_path)
    max_frames = int(params.MAX_FRAMES) if (getattr(params, 'MAX_FRAMES', None) is not None) else None
    # Target total for progress reporting and clamping
    total_target = min(total_frames, max_frames) if (max_frames is not None) else total_frames

    # cuCIM detector parameters with safe defaults; allow optional overrides via params.py if present
    detector = str(getattr(params, 'CUCIM_DETECTOR', 'log')).lower()  # 'log' | 'dog' | 'doh'
    min_sigma = float(getattr(params, 'CUCIM_MIN_SIGMA', 0.8))
    max_sigma = float(getattr(params, 'CUCIM_MAX_SIGMA', 4.0))
    num_sigma = int(getattr(params, 'CUCIM_NUM_SIGMA', 10))
    sigma_ratio = float(getattr(params, 'CUCIM_SIGMA_RATIO', 1.6))
    threshold = float(getattr(params, 'CUCIM_THRESHOLD', 0.05))
    overlap = float(getattr(params, 'CUCIM_OVERLAP', 0.5))
    log_scale = bool(getattr(params, 'CUCIM_LOG_SCALE', False))
    min_area_px = float(getattr(params, 'CUCIM_MIN_AREA_PX', 1.0))
    max_area_scale = float(getattr(params, 'CUCIM_MAX_AREA_SCALE', 1.0))
    pad_px = int(getattr(params, 'CUCIM_PAD_PX', 0))
    use_clahe = bool(getattr(params, 'CUCIM_USE_CLAHE', True))
    clahe_clip = float(getattr(params, 'CUCIM_CLAHE_CLIP', 2.0))
    clahe_tile = getattr(params, 'CUCIM_CLAHE_TILE', 8)
    batch_size = int(getattr(params, 'CUCIM_BATCH_SIZE', 8))
    min_mean_intensity_u8 = int(getattr(params, 'CUCIM_MIN_MEAN_INTENSITY_U8', 0))

    frame_area = int(frame_w * frame_h)
    max_area_px = frame_area if float(max_area_scale) <= 0 else int(frame_area * float(max_area_scale))
    pad = max(0, int(pad_px))

    clahe = _make_clahe(use_clahe, clahe_clip, clahe_tile)

    # Accumulate detections as (frame, x, y, w, h)
    det_rows: list[tuple[int, int, int, int, int]] = []
    frame_idx = 0
    batch_n = max(1, int(batch_size))

    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break

            gray_batch: list[np.ndarray] = []
            idx_batch: list[int] = []
            for _ in range(batch_n):
                if max_frames is not None and frame_idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if clahe is not None:
                    gray = _apply_clahe(gray, clahe)
                gray_batch.append(gray)
                idx_batch.append(frame_idx)
                frame_idx += 1

            if not gray_batch:
                break

            # Upload this CPU batch as a single GPU stack (N,H,W) float32 in [0,1]
            gpu_batch = cp.asarray(np.stack(gray_batch, axis=0), dtype=cp.float32) / 255.0
            # Run cuCIM detector per-slice (API is 2D)
            blobs_list = _detect_blobs_for_batch(
                gpu_batch,
                detector=detector,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                num_sigma=num_sigma,
                sigma_ratio=sigma_ratio,
                threshold=threshold,
                overlap=overlap,
                log_scale=log_scale,
            )

            for bi, fidx in enumerate(idx_batch):
                blobs = blobs_list[bi]
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
                        # Optional intensity gate on the patch (post-CLAHE grayscale, 0..255)
                        if min_mean_intensity_u8 > 0:
                            gimg = gray_batch[bi]
                            roi = gimg[y0:y0+h, x0:x0+w]
                            if roi.size == 0:
                                continue
                            if float(np.mean(roi)) < float(min_mean_intensity_u8):
                                continue
                        det_rows.append((fidx, x0, y0, w, h))

            # Free CuPy memory between batches
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

            progress(min(frame_idx, total_target), total_target, 'Stage1 detect(cuCIM)')
    finally:
        cap.release()

    # Sort for stable ids: by frame, then x, y, w, h
    det_rows.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4]))

    with csv_path.open('w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['frame', 'x', 'y', 'w', 'h'])
        for (t, x, y, w, h) in det_rows:
            wtr.writerow([int(t), int(x), int(y), int(w), int(h)])
    # --- Stats summary ---
    total_candidates = len(det_rows)
    frames_seen = sorted({int(r[0]) for r in det_rows})
    frames_with_dets = len(frames_seen)
    frames_processed = int(frame_idx)
    empty_frames = max(0, frames_processed - frames_with_dets)

    # Per-frame detection counts (include zeros for empty frames)
    counts_by_frame: dict[int, int] = {}
    for (t, *_bbox) in det_rows:
        counts_by_frame[t] = counts_by_frame.get(t, 0) + 1
    counts_all = None
    if frames_processed > 0:
        counts_list = [counts_by_frame.get(t, 0) for t in range(frames_processed)]
        counts_all = np.asarray(counts_list, dtype=np.int32)

    # Box size stats
    if det_rows:
        wh = np.asarray([[r[3], r[4]] for r in det_rows], dtype=np.int32)
        areas = (wh[:, 0] * wh[:, 1]).astype(np.int64)
    else:
        wh = np.zeros((0, 2), dtype=np.int32)
        areas = np.zeros((0,), dtype=np.int64)

    print(f"Stage1  Candidates (total): {total_candidates}")
    print(f"Stage1  Frames processed: {frames_processed} (with detections: {frames_with_dets}, empty: {empty_frames})")
    if counts_all is not None and counts_all.size:
        p25 = float(np.percentile(counts_all, 25))
        p50 = float(np.percentile(counts_all, 50))
        p75 = float(np.percentile(counts_all, 75))
        print(
            f"Stage1  Detections/frame: min={int(counts_all.min())} p25={p25:.1f} p50={p50:.1f} "
            f"mean={counts_all.mean():.2f} p75={p75:.1f} max={int(counts_all.max())}"
        )
        # Top-5 frames by detections (non-zero)
        non_zero = [(t, c) for t, c in enumerate(counts_all.tolist()) if c > 0]
        non_zero.sort(key=lambda x: (-x[1], x[0]))
        top_k = non_zero[:5]
        if top_k:
            top_str = ", ".join([f"t={t}:{c}" for (t, c) in top_k])
            print(f"Stage1  Top frames by detections: {top_str}")
    if areas.size:
        a_p25 = float(np.percentile(areas, 25))
        a_p50 = float(np.percentile(areas, 50))
        a_p75 = float(np.percentile(areas, 75))
        print(
            f"Stage1  Box area(px): min={int(areas.min())} p25={a_p25:.0f} p50={a_p50:.0f} "
            f"mean={areas.mean():.1f} p75={a_p75:.0f} max={int(areas.max())}"
        )
        print(
            f"Stage1  Box w,h mean: ({wh[:,0].mean():.1f},{wh[:,1].mean():.1f}); "
            f"median: ({np.median(wh[:,0]):.0f},{np.median(wh[:,1]):.0f})"
        )
    print(f"Stage1  Wrote CSV → {csv_path}")

    # --- Render overlay video with all Stage 1 candidate boxes ---
    try:
        # Group detections by frame for quick lookup
        boxes_by_t: dict[int, list[tuple[int, int, int, int]]] = {}
        for (t, x, y, w, h) in det_rows:
            boxes_by_t.setdefault(int(t), []).append((int(x), int(y), int(w), int(h)))

        cap2, Wv, Hv, fps_src, total2 = open_video(video_path)
        fps_use = float(params.RENDER_FPS_HINT or fps_src)
        out_video = out_root / f"{stem}_stage1_candidates_overlay.mp4"
        max_frames2 = min(total2, int(params.MAX_FRAMES) if (params.MAX_FRAMES is not None) else total2)
        writer = make_writer(out_video, Wv, Hv, fps_use, codec=params.RENDER_CODEC, is_color=True)

        t = 0
        try:
            while t < max_frames2:
                ok, frame = cap2.read()
                if not ok:
                    break
                if t in boxes_by_t:
                    for (x, y, w, h) in boxes_by_t[t]:
                        x0 = int(x)
                        y0 = int(y)
                        x1 = x0 + int(w)
                        y1 = y0 + int(h)
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 1, cv2.LINE_AA)
                writer.write(frame)
                t += 1
                if t % 50 == 0:
                    progress(t, max_frames2 or t, "Stage1 overlay render")
            progress(t, max_frames2 or t or 1, "Stage1 overlay render done")
        finally:
            cap2.release()
            writer.release()
        print(f"Stage1  Overlay video → {out_video}")
    except Exception as e:
        print(f"Stage1  Overlay render skipped due to error: {e}")
    return csv_path


__all__ = ["run_for_video"]
