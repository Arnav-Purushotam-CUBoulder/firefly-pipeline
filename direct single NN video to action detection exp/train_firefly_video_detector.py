from __future__ import annotations

import argparse
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler, WeightedRandomSampler

# Allow running the script from repo root without installing as a package.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from firefly_video_detector.annotations import load_annotations_csv
from firefly_video_detector.dataset import DatasetSplit, FireflyVideoCenternetDataset
from firefly_video_detector.losses import compute_losses
from firefly_video_detector.model import FireflyVideoCenterNet
from firefly_video_detector.video_io import get_video_info


# =========================
# User-editable config
# =========================
#
# Training input is:
#   X: folder of videos
#   y: folder of CSV files (one per video)
#
# The CSV filename should match the video name (supports either):
#   - <video_stem>.csv        (e.g. my_video.mp4 -> my_video.csv)
#   - <video_filename>.csv    (e.g. my_video.mp4 -> my_video.mp4.csv)
#   - <video_stem>_*.csv      (e.g. GH010181.mp4 -> GH010181_detections_xywh.csv)

# Folder of videos (X). Each video must have a matching CSV in `CSVS_DIR`.
VIDEOS_DIR = "~/Desktop/arnav's files/testing data for e2e NN prototye/training/videos"  # e.g. "/path/to/videos"

# Folder of CSVs (y). Expected columns: x,y,w,h + frame/time column (frame|frame_idx|t) (optional: traj_id/track_id)
CSVS_DIR = "~/Desktop/arnav's files/testing data for e2e NN prototye/training/csvs"  # e.g. "/path/to/csvs"

# Training outputs (checkpoints/config.json)
OUT_DIR = "~/Desktop/arnav's files/testing data for e2e NN prototye/training/training outputs"

# Device: "cuda" | "mps" | "cpu" | None (auto = CUDA → MPS → CPU)
DEVICE = None

# Training epochs (used as the default for --epochs).
EPOCHS = 30


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


class _ResumableWeightedRandomSampler(Sampler[int]):
    """
    WeightedRandomSampler that can resume part-way through a pre-sampled epoch.

    This is used only for mid-epoch resume: we deterministically re-sample the same
    indices for the epoch (via a stored generator state) and start yielding from
    `start_index` (in *sample* units, not batches).
    """

    def __init__(
        self,
        weights: list[float],
        *,
        num_samples: int,
        replacement: bool = True,
        generator: torch.Generator | None = None,
        start_index: int = 0,
    ) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.replacement = bool(replacement)
        self.generator = generator
        self.start_index = max(0, int(start_index))

    def __iter__(self):
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            self.replacement,
            generator=self.generator,
        )
        if self.start_index:
            indices = indices[self.start_index :]
        return iter(indices.tolist())

    def __len__(self) -> int:
        return max(0, int(self.num_samples) - int(self.start_index))


def _atomic_torch_save(obj: object, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _get_rng_state(device: str) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, object] | None, device: str) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])  # type: ignore[arg-type]
    if "numpy" in state:
        np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    if "torch" in state:
        torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    if device == "cuda" and torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])  # type: ignore[arg-type]


def _optimizer_state_to_device(opt: torch.optim.Optimizer, device: str) -> None:
    if device not in {"cuda", "mps"}:
        return
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def _contains_token(haystack: str, needle: str) -> bool:
    """
    Return True if `needle` occurs in `haystack` at token boundaries.

    Token boundary means the character before/after the match is either:
      - start/end of string, or
      - a non-alphanumeric character (e.g. '_', '-', '.', space)

    This makes CSV/video pairing tolerant to small naming variations.
    """

    haystack = str(haystack)
    needle = str(needle)
    if not needle:
        return False

    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return False

        before_ok = idx == 0 or (not haystack[idx - 1].isalnum())
        after_idx = idx + len(needle)
        after_ok = after_idx >= len(haystack) or (not haystack[after_idx].isalnum())
        if before_ok and after_ok:
            return True

        start = idx + 1


def _collect_video_csv_pairs(videos_dir: Path, csvs_dir: Path) -> list[tuple[Path, Path]]:
    videos_dir = Path(videos_dir).expanduser()
    csvs_dir = Path(csvs_dir).expanduser()
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos dir not found: {videos_dir}")
    if not csvs_dir.exists():
        raise FileNotFoundError(f"CSVs dir not found: {csvs_dir}")

    videos = sorted(
        [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda p: p.name,
    )
    if not videos:
        raise FileNotFoundError(f"No videos found in: {videos_dir} (exts: {sorted(VIDEO_EXTS)})")

    csvs = [p for p in csvs_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
    csv_by_name_lower: dict[str, Path] = {}
    for p in csvs:
        k = p.name.lower()
        if k in csv_by_name_lower:
            raise ValueError(f"Duplicate CSV names (case-insensitive): {p} and {csv_by_name_lower[k]}")
        csv_by_name_lower[k] = p
    csvs_lower = [(p.name.lower(), p) for p in csvs]

    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for v in videos:
        cand_names = [f"{v.stem}.csv", f"{v.name}.csv"]
        found_exact = [csv_by_name_lower.get(n.lower()) for n in cand_names]
        found_exact = [p for p in found_exact if p is not None]
        if len({p.resolve() for p in found_exact}) > 1:
            raise ValueError(
                f"Ambiguous CSV match for video {v.name}: {[str(p) for p in found_exact]}"
            )
        if found_exact:
            pairs.append((v, found_exact[0]))
            continue

        # Common patterns:
        #   - "<video_stem>_<something>.csv"
        #   - "<video_stem>-<something>.csv"
        #   - "<something>_<video_stem>_<something>.csv"
        # This stays strict enough to avoid matching similar IDs (e.g. GH010181 vs GH0101812)
        # because it enforces token boundaries around the video name.
        key1 = v.stem.lower()
        key2 = v.name.lower()
        found_prefix = [
            p
            for name_l, p in csvs_lower
            if _contains_token(name_l, key1) or _contains_token(name_l, key2)
        ]
        if not found_prefix:
            missing.append(
                f"- {v.name} (expected {cand_names[0]} or {cand_names[1]} or a CSV containing '{v.stem}' in its name)"
            )
            continue
        if len({p.resolve() for p in found_prefix}) > 1:
            raise ValueError(
                "\n".join(
                    [
                        f"Ambiguous CSV match for video {v.name}.",
                        "Matched multiple CSVs by prefix:",
                        *[f"- {p.name}" for p in sorted(found_prefix, key=lambda x: x.name)],
                        "Fix by renaming CSVs to be unique per video or use --video/--csv for single-video mode.",
                    ]
                )
            )
        pairs.append((v, found_prefix[0]))

    if missing:
        msg = "\n".join(
            [
                "Some videos do not have a matching CSV in CSVS_DIR.",
                f"Videos dir: {videos_dir}",
                f"CSVs dir:  {csvs_dir}",
                "Missing:",
                *missing,
            ]
        )
        raise FileNotFoundError(msg)

    return pairs


def _device_from_args(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos_dir", default=VIDEOS_DIR, help="Folder of videos (X)")
    ap.add_argument("--csvs_dir", default=CSVS_DIR, help="Folder of CSVs (y), one per video")
    ap.add_argument("--video", default=None, help="(Optional) single video path (overrides --videos_dir)")
    ap.add_argument("--csv", default=None, help="(Optional) single CSV path (overrides --csvs_dir)")
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--clip_len", type=int, default=8)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--resize_h", type=int, default=256)
    ap.add_argument("--resize_w", type=int, default=256)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--feat_channels", type=int, default=128)
    ap.add_argument("--wh_weight", type=float, default=0.1)
    ap.add_argument("--off_weight", type=float, default=1.0)
    ap.add_argument("--track_weight", type=float, default=1.0)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--device", default=DEVICE, help="cpu|cuda|mps (auto if omitted)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples_per_epoch", type=int, default=0, help="0 = use len(dataset)")
    ap.add_argument(
        "--resume",
        choices=["auto", "always", "never"],
        default="auto",
        help=(
            "Resume from out_dir/checkpoints/last.pt. "
            "auto=resumes if present and training isn't finished; "
            "always=requires it; never=starts fresh even if it exists."
        ),
    )
    ap.add_argument(
        "--save_every_n_steps",
        type=int,
        default=0,
        help="0 = only save at epoch end; otherwise also save every N optimizer steps (enables mid-epoch resume).",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    def _handle_termination(_signum, _frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _handle_termination)
    except Exception:
        pass

    if int(args.downsample) != 4:
        raise ValueError("This prototype currently uses a fixed output stride of 4 (downsample=4).")

    device = _device_from_args(args.device)

    out_dir = Path(args.out_dir).expanduser()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "last.pt"
    resume_ckpt: dict[str, object] | None = None
    pairs_from_ckpt: list[tuple[Path, Path]] | None = None
    start_epoch = 1
    resume_step_in_epoch = 0
    resume_start_index_samples = 0
    resume_mid_epoch = False

    if str(args.resume) != "never":
        if ckpt_path.exists():
            try:
                loaded = torch.load(ckpt_path, map_location="cpu")
                if not isinstance(loaded, dict):
                    raise TypeError(f"Expected a dict checkpoint, got: {type(loaded)}")
                resume_ckpt = loaded
            except Exception as e:
                raise RuntimeError(
                    "\n".join(
                        [
                            f"Found checkpoint but failed to load: {ckpt_path}",
                            f"Error: {e}",
                            "Fix: delete/rename the checkpoint, use --resume=never, or change --out_dir.",
                        ]
                    )
                ) from e
        elif str(args.resume) == "always":
            raise FileNotFoundError(f"--resume=always but checkpoint not found: {ckpt_path}")

    if resume_ckpt is not None:
        last_completed_epoch = int(resume_ckpt.get("epoch", 0) or 0)
        epoch_in_progress: int | None = None
        try:
            if resume_ckpt.get("epoch_in_progress") is not None:
                epoch_in_progress = int(resume_ckpt["epoch_in_progress"])  # type: ignore[arg-type]
        except Exception:
            epoch_in_progress = None

        step_in_epoch = 0
        try:
            if resume_ckpt.get("step_in_epoch") is not None:
                step_in_epoch = int(resume_ckpt["step_in_epoch"])  # type: ignore[arg-type]
        except Exception:
            step_in_epoch = 0

        ckpt_config = resume_ckpt.get("config")
        ckpt_batch_size: int | None = None
        if isinstance(ckpt_config, dict):
            try:
                if ckpt_config.get("batch_size") is not None:
                    ckpt_batch_size = int(ckpt_config["batch_size"])  # type: ignore[arg-type]
            except Exception:
                ckpt_batch_size = None

        has_mid_epoch = (
            epoch_in_progress is not None
            and step_in_epoch > 0
            and torch.is_tensor(resume_ckpt.get("sampler_gen_state_epoch_start"))
        )
        if has_mid_epoch and ckpt_batch_size is not None and int(args.batch_size) != int(ckpt_batch_size):
            print(
                "\n".join(
                    [
                        "⚠️ Checkpoint is mid-epoch, but --batch_size changed.",
                        f"Checkpoint batch_size={ckpt_batch_size}, current batch_size={int(args.batch_size)}.",
                        "Falling back to epoch-boundary resume (no mid-epoch resume).",
                    ]
                )
            )
            has_mid_epoch = False
            step_in_epoch = 0

        if has_mid_epoch:
            start_epoch = int(epoch_in_progress)
            resume_step_in_epoch = int(step_in_epoch)
            resume_start_index_samples = resume_step_in_epoch * int(args.batch_size)
            resume_mid_epoch = True
        else:
            start_epoch = int(last_completed_epoch) + 1

        if start_epoch > int(args.epochs):
            print(
                "\n".join(
                    [
                        f"✅ Training already completed (checkpoint epoch={last_completed_epoch}, target epochs={int(args.epochs)}).",
                        "Nothing to do. Increase --epochs to keep training, or use a new --out_dir to start fresh.",
                    ]
                )
            )
            return

        if isinstance(ckpt_config, dict):
            ckpt_data = ckpt_config.get("data")
            ckpt_pairs = ckpt_data.get("pairs") if isinstance(ckpt_data, dict) else None
            if isinstance(ckpt_pairs, list) and ckpt_pairs:
                tmp: list[tuple[Path, Path]] = []
                for item in ckpt_pairs:
                    if not isinstance(item, dict):
                        tmp = []
                        break
                    v = item.get("video")
                    c = item.get("csv")
                    if not v or not c:
                        tmp = []
                        break
                    tmp.append((Path(str(v)).expanduser(), Path(str(c)).expanduser()))
                if tmp:
                    pairs_from_ckpt = tmp

        if resume_mid_epoch:
            print(
                f"🔄 Resuming from {ckpt_path} (mid-epoch): epoch {start_epoch}, step {resume_step_in_epoch}"
            )
        elif last_completed_epoch > 0:
            print(f"🔄 Resuming from {ckpt_path}: starting epoch {start_epoch} (last={last_completed_epoch})")

    if pairs_from_ckpt is not None:
        pairs = pairs_from_ckpt
    elif args.video is not None or args.csv is not None:
        if not args.video or not args.csv:
            raise ValueError("--video and --csv must be provided together (single-video mode).")
        pairs = [(Path(args.video).expanduser(), Path(args.csv).expanduser())]
    else:
        if not str(args.videos_dir).strip() or not str(args.csvs_dir).strip():
            raise ValueError(
                "Provide --videos_dir and --csvs_dir (or set VIDEOS_DIR/CSVS_DIR at the top of the script)."
            )
        pairs = _collect_video_csv_pairs(Path(args.videos_dir), Path(args.csvs_dir))

    train_datasets: list[FireflyVideoCenternetDataset] = []
    val_datasets: list[FireflyVideoCenternetDataset] = []
    train_weights: list[float] = []
    video_infos: list[dict[str, object]] = []

    val_fraction = max(0.0, min(0.9, float(args.val_fraction)))
    for video_path, csv_path in pairs:
        info = get_video_info(video_path)
        annotations = load_annotations_csv(csv_path)

        total_frames = info.frame_count
        split_at = int(total_frames * (1.0 - val_fraction))
        split_at = max(1, min(total_frames - 1, split_at))

        train_split = DatasetSplit(0, split_at)
        val_split = DatasetSplit(split_at, total_frames)

        train_ds_i = FireflyVideoCenternetDataset(
            video_path=info.path,
            annotations=annotations,
            total_frames=total_frames,
            orig_size=(info.width, info.height),
            split=train_split,
            resize_hw=(args.resize_h, args.resize_w),
            clip_len=args.clip_len,
            frame_stride=args.frame_stride,
            downsample=args.downsample,
            augment=args.augment,
        )
        val_ds_i = FireflyVideoCenternetDataset(
            video_path=info.path,
            annotations=annotations,
            total_frames=total_frames,
            orig_size=(info.width, info.height),
            split=val_split,
            resize_hw=(args.resize_h, args.resize_w),
            clip_len=args.clip_len,
            frame_stride=args.frame_stride,
            downsample=args.downsample,
            augment=False,
        )
        train_datasets.append(train_ds_i)
        val_datasets.append(val_ds_i)
        train_weights.extend([float(w) for w in train_ds_i.sample_weights])

        video_infos.append(
            {
                "path": str(info.path),
                "csv": str(csv_path),
                "frame_count": info.frame_count,
                "fps": info.fps,
                "width": info.width,
                "height": info.height,
                "train_frames": [train_split.start_frame, train_split.end_frame_exclusive],
                "val_frames": [val_split.start_frame, val_split.end_frame_exclusive],
            }
        )

    train_ds = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    val_ds = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)

    num_samples = int(args.samples_per_epoch) if int(args.samples_per_epoch) > 0 else len(train_ds)
    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(int(args.seed))
    if resume_ckpt is not None:
        key = "sampler_gen_state"
        try:
            eip = resume_ckpt.get("epoch_in_progress")
            if (
                eip is not None
                and int(eip) == int(start_epoch)
                and torch.is_tensor(resume_ckpt.get("sampler_gen_state_epoch_start"))
            ):
                key = "sampler_gen_state_epoch_start"
        except Exception:
            pass
        state = resume_ckpt.get(key)
        if torch.is_tensor(state):
            sampler_gen.set_state(state)
        elif state is not None:
            print(f"⚠️ Checkpoint had '{key}' but it wasn't a torch Tensor; ignoring.")

    def make_train_dl(*, start_index_samples: int = 0) -> DataLoader:
        if start_index_samples > 0:
            sampler: Sampler[int] = _ResumableWeightedRandomSampler(
                train_weights,
                num_samples=num_samples,
                replacement=True,
                generator=sampler_gen,
                start_index=start_index_samples,
            )
        else:
            sampler = WeightedRandomSampler(
                train_weights,
                num_samples=num_samples,
                replacement=True,
                generator=sampler_gen,
            )
        return DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
        )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = FireflyVideoCenterNet(
        base_channels=int(args.base_channels),
        feat_channels=int(args.feat_channels),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if resume_ckpt is not None:
        try:
            model.load_state_dict(resume_ckpt["model"])  # type: ignore[arg-type]
            opt.load_state_dict(resume_ckpt["optimizer"])  # type: ignore[arg-type]
        except Exception as e:
            raise RuntimeError(
                "\n".join(
                    [
                        f"Failed to load checkpoint state from: {ckpt_path}",
                        "If you intended to start a fresh training run, use --resume=never or a new --out_dir.",
                    ]
                )
            ) from e

    model = model.to(device)
    _optimizer_state_to_device(opt, device)
    if resume_ckpt is not None:
        _set_rng_state(resume_ckpt.get("rng_state"), device)

    config = {
        "data": {
            "videos_dir": str(Path(args.videos_dir).expanduser()) if args.video is None else None,
            "csvs_dir": str(Path(args.csvs_dir).expanduser()) if args.csv is None else None,
            "pairs": [{"video": str(v), "csv": str(c)} for v, c in pairs],
        },
        "video_infos": video_infos,
        "resize_hw": [args.resize_h, args.resize_w],
        "clip_len": args.clip_len,
        "frame_stride": args.frame_stride,
        "downsample": args.downsample,
        "base_channels": args.base_channels,
        "feat_channels": args.feat_channels,
        "loss_weights": {
            "wh": args.wh_weight,
            "offset": args.off_weight,
            "tracking": args.track_weight,
        },
        "val_fraction": args.val_fraction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "samples_per_epoch": args.samples_per_epoch,
        "num_samples_per_epoch": int(num_samples),
        "save_every_n_steps": int(args.save_every_n_steps),
        "device": device,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    best_val = float("inf")
    global_step = 0
    if resume_ckpt is not None:
        try:
            if resume_ckpt.get("best_val") is not None:
                best_val = float(resume_ckpt["best_val"])  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            if resume_ckpt.get("global_step") is not None:
                global_step = int(resume_ckpt["global_step"])  # type: ignore[arg-type]
        except Exception:
            pass

    # Back-compat: older checkpoints didn't store `best_val`. If we can't recover it,
    # avoid accidentally overwriting an existing best.pt by estimating its val loss once.
    best_ckpt_path = ckpt_dir / "best.pt"
    if resume_ckpt is not None and (not math.isfinite(best_val)) and best_ckpt_path.exists():
        try:
            loaded_best = torch.load(best_ckpt_path, map_location="cpu")
            best_ckpt = loaded_best if isinstance(loaded_best, dict) else None
            if best_ckpt is not None and best_ckpt.get("val_loss") is not None:
                best_val = float(best_ckpt["val_loss"])  # type: ignore[arg-type]
            elif best_ckpt is not None and best_ckpt.get("model") is not None:
                best_model = FireflyVideoCenterNet(
                    base_channels=int(args.base_channels),
                    feat_channels=int(args.feat_channels),
                )
                best_model.load_state_dict(best_ckpt["model"])  # type: ignore[arg-type]
                best_model.eval()

                val_tot = 0.0
                val_n = 0
                with torch.no_grad():
                    for clip, target in val_dl:
                        target = {k: v for k, v in target.items() if torch.is_tensor(v)}
                        pred = best_model(clip)
                        losses = compute_losses(
                            pred,
                            target,
                            wh_weight=float(args.wh_weight),
                            off_weight=float(args.off_weight),
                            track_weight=float(args.track_weight),
                        )
                        val_tot += float(losses["total"].detach().cpu())
                        val_n += 1
                best_val = val_tot / max(1, val_n)
                print(f"ℹ️ Inferred best_val={best_val:.4f} from existing {best_ckpt_path}")
        except Exception as e:
            print(f"⚠️ Could not infer best_val from {best_ckpt_path}: {e}")

    def save_ckpt(
        *,
        path: Path,
        last_completed_epoch: int,
        epoch_in_progress: int | None = None,
        step_in_epoch: int = 0,
        global_step: int = 0,
        train_loss: float | None = None,
        val_loss: float | None = None,
        sampler_gen_state_epoch_start: torch.ByteTensor | None = None,
        reason: str = "checkpoint",
    ) -> None:
        ckpt: dict[str, object] = {
            "epoch": int(last_completed_epoch),
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": config,
            "video_infos": video_infos,
            "best_val": float(best_val),
            "global_step": int(global_step),
            "rng_state": _get_rng_state(device),
            "sampler_gen_state": sampler_gen.get_state(),
            "saved_at_unix": float(time.time()),
            "reason": str(reason),
        }
        if epoch_in_progress is not None:
            ckpt["epoch_in_progress"] = int(epoch_in_progress)
            ckpt["step_in_epoch"] = int(step_in_epoch)
        if train_loss is not None:
            ckpt["train_loss"] = float(train_loss)
        if val_loss is not None:
            ckpt["val_loss"] = float(val_loss)
        if sampler_gen_state_epoch_start is not None:
            ckpt["sampler_gen_state_epoch_start"] = sampler_gen_state_epoch_start

        _atomic_torch_save(ckpt, path)

    for epoch in range(int(start_epoch), int(args.epochs) + 1):
        start_index_samples = 0
        if resume_mid_epoch and epoch == int(start_epoch):
            start_index_samples = min(int(resume_start_index_samples), int(num_samples))

        train_dl = make_train_dl(start_index_samples=start_index_samples)
        sampler_gen_state_epoch_start = sampler_gen.get_state()

        model.train()
        train_tot = 0.0
        train_n = 0
        step_in_epoch = int(resume_step_in_epoch) if (resume_mid_epoch and epoch == int(start_epoch)) else 0

        try:
            it = train_dl
            if tqdm is not None:
                # Avoid multi-line spam in narrow terminals by letting tqdm size itself.
                it = tqdm(train_dl, desc=f"train {epoch:03d}", dynamic_ncols=True, file=sys.stdout)
            for clip, target in it:
                clip = clip.to(device)
                target = {k: v.to(device) for k, v in target.items() if torch.is_tensor(v)}

                pred = model(clip)
                losses = compute_losses(
                    pred,
                    target,
                    wh_weight=float(args.wh_weight),
                    off_weight=float(args.off_weight),
                    track_weight=float(args.track_weight),
                )

                opt.zero_grad(set_to_none=True)
                losses["total"].backward()
                opt.step()

                train_tot += float(losses["total"].detach().cpu())
                train_n += 1
                step_in_epoch += 1
                global_step += 1

                if int(args.save_every_n_steps) > 0 and (global_step % int(args.save_every_n_steps) == 0):
                    save_ckpt(
                        path=ckpt_dir / "last.pt",
                        last_completed_epoch=epoch - 1,
                        epoch_in_progress=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        train_loss=(train_tot / max(1, train_n)),
                        sampler_gen_state_epoch_start=sampler_gen_state_epoch_start,
                        reason=f"step_{global_step}",
                    )
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted. Saving checkpoint so you can resume…")
            save_ckpt(
                path=ckpt_dir / "last.pt",
                last_completed_epoch=epoch - 1,
                epoch_in_progress=epoch,
                step_in_epoch=step_in_epoch,
                global_step=global_step,
                train_loss=(train_tot / max(1, train_n)),
                sampler_gen_state_epoch_start=sampler_gen_state_epoch_start,
                reason="interrupt",
            )
            print(f"  💾 saved → {ckpt_dir / 'last.pt'}")
            return

        train_avg = train_tot / max(1, train_n)

        model.eval()
        val_tot = 0.0
        val_n = 0
        with torch.no_grad():
            it2 = val_dl
            if tqdm is not None:
                it2 = tqdm(val_dl, desc=f"val   {epoch:03d}", dynamic_ncols=True, file=sys.stdout)
            for clip, target in it2:
                clip = clip.to(device)
                target = {k: v.to(device) for k, v in target.items() if torch.is_tensor(v)}
                pred = model(clip)
                losses = compute_losses(
                    pred,
                    target,
                    wh_weight=float(args.wh_weight),
                    off_weight=float(args.off_weight),
                    track_weight=float(args.track_weight),
                )
                val_tot += float(losses["total"].detach().cpu())
                val_n += 1

        val_avg = val_tot / max(1, val_n)
        best_path = ckpt_dir / "best.pt"
        if math.isfinite(best_val):
            is_best = val_avg < best_val
        else:
            is_best = not best_path.exists()
        if is_best:
            best_val = val_avg

        print(f"Epoch {epoch:03d} | train_loss {train_avg:.4f} | val_loss {val_avg:.4f}")

        save_ckpt(
            path=ckpt_dir / "last.pt",
            last_completed_epoch=epoch,
            global_step=global_step,
            train_loss=train_avg,
            val_loss=val_avg,
            reason="epoch_end",
        )
        if is_best:
            save_ckpt(
                path=ckpt_dir / "best.pt",
                last_completed_epoch=epoch,
                global_step=global_step,
                train_loss=train_avg,
                val_loss=val_avg,
                reason="best",
            )
            print(f"  ✅ saved best → {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    # macOS DataLoader defaults to spawn; be explicit.
    try:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
