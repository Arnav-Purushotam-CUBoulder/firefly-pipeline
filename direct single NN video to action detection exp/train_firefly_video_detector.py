from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

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

        # Common pattern: "<video_stem>_something.csv" (e.g. "GH010181_detections_xywh.csv").
        prefix1 = f"{v.stem.lower()}_"
        prefix2 = f"{v.name.lower()}_"
        found_prefix = [p for name_l, p in csvs_lower if name_l.startswith(prefix1) or name_l.startswith(prefix2)]
        if not found_prefix:
            missing.append(
                f"- {v.name} (expected {cand_names[0]} or {cand_names[1]} or {v.stem}_*.csv)"
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if int(args.downsample) != 4:
        raise ValueError("This prototype currently uses a fixed output stride of 4 (downsample=4).")

    device = _device_from_args(args.device)

    out_dir = Path(args.out_dir).expanduser()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.video is not None or args.csv is not None:
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
    sampler = WeightedRandomSampler(train_weights, num_samples=num_samples, replacement=True)
    train_dl = DataLoader(
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
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

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
        "device": device,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    best_val = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_tot = 0.0
        train_n = 0
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
        print(f"Epoch {epoch:03d} | train_loss {train_avg:.4f} | val_loss {val_avg:.4f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": config,
            "video_infos": video_infos,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_avg < best_val:
            best_val = val_avg
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  ✅ saved best → {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    # macOS DataLoader defaults to spawn; be explicit.
    try:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
