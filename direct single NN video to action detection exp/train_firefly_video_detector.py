from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from firefly_video_detector.annotations import load_annotations_csv
from firefly_video_detector.dataset import DatasetSplit, FireflyVideoCenternetDataset
from firefly_video_detector.losses import compute_losses
from firefly_video_detector.model import FireflyVideoCenterNet
from firefly_video_detector.video_io import get_video_info


def _device_from_args(device: str | None) -> str:
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to MP4 video")
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to CSV (x,y,w,h,frame[,traj_id]) where traj_id enables motion/trajectory supervision",
    )
    ap.add_argument("--out_dir", default="runs/firefly_video_centernet")
    ap.add_argument("--epochs", type=int, default=30)
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
    ap.add_argument("--device", default=None, help="cpu|cuda|mps (auto if omitted)")
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

    video_info = get_video_info(args.video)
    annotations = load_annotations_csv(args.csv)

    total_frames = video_info.frame_count
    val_fraction = max(0.0, min(0.9, float(args.val_fraction)))
    split_at = int(total_frames * (1.0 - val_fraction))
    split_at = max(1, min(total_frames - 1, split_at))

    train_split = DatasetSplit(0, split_at)
    val_split = DatasetSplit(split_at, total_frames)

    train_ds = FireflyVideoCenternetDataset(
        video_path=video_info.path,
        annotations=annotations,
        total_frames=total_frames,
        orig_size=(video_info.width, video_info.height),
        split=train_split,
        resize_hw=(args.resize_h, args.resize_w),
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        downsample=args.downsample,
        augment=args.augment,
    )
    val_ds = FireflyVideoCenternetDataset(
        video_path=video_info.path,
        annotations=annotations,
        total_frames=total_frames,
        orig_size=(video_info.width, video_info.height),
        split=val_split,
        resize_hw=(args.resize_h, args.resize_w),
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        downsample=args.downsample,
        augment=False,
    )

    num_samples = int(args.samples_per_epoch) if int(args.samples_per_epoch) > 0 else len(train_ds)
    sampler = WeightedRandomSampler(train_ds.sample_weights, num_samples=num_samples, replacement=True)
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
        "video": str(video_info.path),
        "csv": str(Path(args.csv).expanduser()),
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
            it = tqdm(train_dl, desc=f"train {epoch:03d}", ncols=100)
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
                it2 = tqdm(val_dl, desc=f"val   {epoch:03d}", ncols=100)
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
            "video_info": {
                "frame_count": video_info.frame_count,
                "fps": video_info.fps,
                "width": video_info.width,
                "height": video_info.height,
            },
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
