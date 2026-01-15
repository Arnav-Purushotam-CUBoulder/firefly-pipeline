from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from firefly_video_detector.decode import decode_centernet
from firefly_video_detector.model import FireflyVideoCenterNet
from firefly_video_detector.video_io import VideoClipReader, get_video_info


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
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--ignore_ckpt_config",
        action="store_true",
        help="Do not override clip/resize settings from the checkpoint config.",
    )
    ap.add_argument("--clip_len", type=int, default=8)
    ap.add_argument("--frame_stride", type=int, default=1)
    ap.add_argument("--resize_h", type=int, default=256)
    ap.add_argument("--resize_w", type=int, default=256)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--feat_channels", type=int, default=128)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--score_thresh", type=float, default=0.3)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--time_col", default="frame", choices=["frame", "t"])
    ap.add_argument(
        "--emit_traj_id",
        action="store_true",
        help="Assign a trajectory id by linking detections using the model's tracking head.",
    )
    ap.add_argument(
        "--max_track_dist",
        type=float,
        default=25.0,
        help="Max distance (in resized pixels) for linking to previous frame.",
    )
    args = ap.parse_args()

    device = _device_from_args(args.device)

    ckpt = torch.load(Path(args.ckpt).expanduser(), map_location="cpu")
    if not args.ignore_ckpt_config and isinstance(ckpt.get("config"), dict):
        cfg = ckpt["config"]
        if isinstance(cfg.get("resize_hw"), (list, tuple)) and len(cfg["resize_hw"]) == 2:
            args.resize_h = int(cfg["resize_hw"][0])
            args.resize_w = int(cfg["resize_hw"][1])
        args.clip_len = int(cfg.get("clip_len", args.clip_len))
        args.frame_stride = int(cfg.get("frame_stride", args.frame_stride))
        args.downsample = int(cfg.get("downsample", args.downsample))
        args.base_channels = int(cfg.get("base_channels", args.base_channels))
        args.feat_channels = int(cfg.get("feat_channels", args.feat_channels))

    if int(args.downsample) != 4:
        raise ValueError("This prototype currently uses a fixed output stride of 4 (downsample=4).")

    if args.emit_traj_id and int(args.step) != int(args.frame_stride):
        raise ValueError(
            "--emit_traj_id expects consecutive processed frames; set --step == --frame_stride."
        )

    model = FireflyVideoCenterNet(
        base_channels=int(args.base_channels),
        feat_channels=int(args.feat_channels),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    video_info = get_video_info(args.video)
    reader = VideoClipReader(video_info.path, frame_count=video_info.frame_count)

    end = int(args.end) if int(args.end) >= 0 else video_info.frame_count
    start = max(0, int(args.start))
    end = min(video_info.frame_count, end)
    step = max(1, int(args.step))

    sx = video_info.width / float(args.resize_w)
    sy = video_info.height / float(args.resize_h)

    out_csv = Path(args.out_csv).expanduser()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["x", "y", "w", "h", args.time_col, "score"]
    if args.emit_traj_id:
        fieldnames.append("traj_id")
    with out_csv.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()

        import cv2

        # Track state at the previous processed frame: {traj_id: (cx, cy)} in resized coords.
        prev_tracks: dict[int, tuple[float, float]] = {}
        next_traj_id = 0

        for frame_index in range(start, end, step):
            frames = reader.read_rgb_clip(
                center_frame=frame_index,
                clip_len=args.clip_len,
                frame_stride=args.frame_stride,
                pad_mode="edge",
            )

            resized = [
                cv2.resize(f, (args.resize_w, args.resize_h), interpolation=cv2.INTER_AREA)
                for f in frames
            ]
            clip_thwc = np.stack(resized, axis=0).astype(np.float32) / 255.0  # [T,H,W,3]
            clip = torch.from_numpy(clip_thwc).permute(3, 0, 1, 2).unsqueeze(0).to(device)  # [1,3,T,H,W]

            with torch.no_grad():
                pred = model(clip)
                dets = decode_centernet(
                    pred,
                    downsample=args.downsample,
                    input_hw=(args.resize_h, args.resize_w),
                    topk=args.topk,
                    score_thresh=args.score_thresh,
                    nms_iou=args.nms_iou,
                )

            det_traj_ids: list[int] = []
            if args.emit_traj_id:
                assigned = [-1] * len(dets)
                used_prev: set[int] = set()
                used_det: set[int] = set()

                # Greedy matching using predicted prev-center (CenterTrack-style).
                candidates: list[tuple[float, int, int]] = []
                for prev_id, (pcx, pcy) in prev_tracks.items():
                    for i, d in enumerate(dets):
                        cx = float(d.x + d.w * 0.5)
                        cy = float(d.y + d.h * 0.5)
                        if d.track_dx is not None and d.track_dy is not None:
                            prev_cx = cx + float(d.track_dx)
                            prev_cy = cy + float(d.track_dy)
                        else:
                            prev_cx, prev_cy = cx, cy
                        dist = math.hypot(prev_cx - pcx, prev_cy - pcy)
                        candidates.append((dist, i, prev_id))

                candidates.sort(key=lambda x: x[0])
                max_d = float(args.max_track_dist)
                for dist, i, prev_id in candidates:
                    if dist > max_d:
                        break
                    if i in used_det or prev_id in used_prev:
                        continue
                    assigned[i] = prev_id
                    used_det.add(i)
                    used_prev.add(prev_id)

                for i in range(len(dets)):
                    if assigned[i] == -1:
                        assigned[i] = next_traj_id
                        next_traj_id += 1

                det_traj_ids = assigned
                # Build tracks for next iteration (only current-frame tracks are needed).
                prev_tracks = {}
                for i, d in enumerate(dets):
                    cx = float(d.x + d.w * 0.5)
                    cy = float(d.y + d.h * 0.5)
                    prev_tracks[int(det_traj_ids[i])] = (cx, cy)

            for i, d in enumerate(dets):
                traj_id = int(det_traj_ids[i]) if args.emit_traj_id else None
                # Convert resized coords -> original video coords
                row = {
                    "x": d.x * sx,
                    "y": d.y * sy,
                    "w": d.w * sx,
                    "h": d.h * sy,
                    args.time_col: frame_index,
                    "score": d.score,
                }
                if traj_id is not None:
                    row["traj_id"] = traj_id
                wr.writerow(row)


if __name__ == "__main__":
    main()
