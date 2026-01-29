from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running the script from repo root without installing as a package.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from firefly_video_detector.decode import decode_centernet
from firefly_video_detector.model import FireflyVideoCenterNet
from firefly_video_detector.video_io import VideoClipReader, VideoInfo, get_video_info


# =========================
# User-editable config
# =========================
#
# Inference input is:
#   X: folder of videos
# Output is:
#   ŷ: folder of CSV files (one per video) with columns x,y,w,h,frame
#   (optional) rendered MP4s with predicted boxes drawn

# Folder of videos to run inference on.
VIDEOS_DIR = "~/Desktop/arnav's files/testing data for e2e NN prototye/training/videos"  # e.g. "/path/to/videos"

# Output folder to write per-video CSVs to.
OUT_CSV_DIR = "~/Desktop/arnav's files/testing data for e2e NN prototye/inference"  # e.g. "/path/to/output_csvs"

# Model checkpoint to load.
CKPT_PATH = "~/Desktop/arnav's files/testing data for e2e NN prototye/training/training outputs/checkpoints/best.pt"  # e.g. "runs/firefly_video_centernet/checkpoints/best.pt"

# Device: "cuda" | "mps" | "cpu" | None (auto = CUDA → MPS → CPU)
DEVICE = None

# Default CSV time column ("frame" matches your label format).
TIME_COL = "frame"  # "frame" or "t"

# Optional extra output columns.
INCLUDE_SCORE = False
EMIT_TRAJ_ID = False
MAX_TRACK_DIST = 25.0

# Naming: if True -> "<video>.mp4.csv", else -> "<video>.csv"
OUTPUT_USE_VIDEO_FILENAME = False

# Optional: process only the first N frames (after --start) for each video.
MAX_FRAMES = 2000  # e.g. 500

# Render annotated videos after writing CSVs.
RENDER_VIDEOS = True
OUT_VIDEO_DIR = ""  # default: same folder as the CSV outputs
OUT_VIDEO_SUFFIX = "_annotated"  # output name: "<stem><suffix>.mp4"
OUT_VIDEO_CODEC = "mp4v"
BBOX_THICKNESS_PX = 1


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def _list_videos(videos_dir: Path) -> list[Path]:
    videos_dir = Path(videos_dir).expanduser()
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos dir not found: {videos_dir}")
    videos = sorted(
        [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
        key=lambda p: p.name,
    )
    if not videos:
        raise FileNotFoundError(f"No videos found in: {videos_dir} (exts: {sorted(VIDEO_EXTS)})")
    return videos


def _range_len(start: int, end: int, step: int) -> int:
    if end <= start:
        return 0
    return (end - start + step - 1) // step


def _default_out_csv(video_path: Path, out_dir: Path, use_video_filename: bool) -> Path:
    name = f"{video_path.name}.csv" if use_video_filename else f"{video_path.stem}.csv"
    return out_dir / name


def _default_out_video(video_path: Path, out_dir: Path, suffix: str) -> Path:
    suffix = str(suffix or "")
    return out_dir / f"{video_path.stem}{suffix}.mp4"


def _open_video_capture(video_path: Path):
    import cv2

    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def _open_video_writer(path: Path, fps: float, width: int, height: int, codec: str):
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fps = float(fps) if float(fps) > 0 else 30.0
    width = int(width)
    height = int(height)

    def _try(codec_name: str):
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height), isColor=True)
        return writer if writer.isOpened() else None

    tried = []
    codec = str(codec or "mp4v")
    writer = _try(codec)
    tried.append(codec)
    if writer is None:
        for fallback in ("mp4v", "avc1", "XVID", "MJPG"):
            if fallback in tried:
                continue
            writer = _try(fallback)
            tried.append(fallback)
            if writer is not None:
                break

    if writer is None:
        raise RuntimeError(f"Could not open VideoWriter for {path} (tried codecs: {tried})")
    return writer


def _render_annotated_video(
    video_info: VideoInfo,
    out_path: Path,
    boxes_by_frame: dict[int, list[tuple[float, float, float, float]]],
    start: int,
    end: int,
    bbox_thickness_px: int = 1,
    codec: str = "mp4v",
    tqdm=None,
) -> None:
    import cv2

    start = max(0, int(start))
    end = min(int(end), int(video_info.frame_count))
    if end <= start:
        return

    out_path = Path(out_path).expanduser()
    cap = _open_video_capture(video_info.path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        fps = (
            float(video_info.fps)
            if float(video_info.fps) > 0
            else float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        )
        writer = _open_video_writer(
            out_path,
            fps=fps,
            width=int(video_info.width),
            height=int(video_info.height),
            codec=str(codec),
        )
        thickness = max(1, int(bbox_thickness_px))

        try:
            frame_iter = range(start, end)
            if tqdm is not None:
                frame_iter = tqdm(
                    frame_iter,
                    desc=f"render {video_info.path.name}",
                    unit="frame",
                    ncols=100,
                )

            for frame_idx in frame_iter:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                boxes = boxes_by_frame.get(int(frame_idx))
                if boxes:
                    H, W = frame.shape[:2]
                    for x, y, w, h in boxes:
                        x0 = int(round(float(x)))
                        y0 = int(round(float(y)))
                        x1 = int(round(float(x) + float(w)))
                        y1 = int(round(float(y) + float(h)))

                        if x1 <= x0 or y1 <= y0:
                            continue

                        x0 = max(0, min(x0, W - 1))
                        y0 = max(0, min(y0, H - 1))
                        x1 = max(0, min(x1, W - 1))
                        y1 = max(0, min(y1, H - 1))
                        if x1 <= x0 or y1 <= y0:
                            continue

                        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), thickness, cv2.LINE_8)

                writer.write(frame)
        finally:
            writer.release()
    finally:
        cap.release()


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
    ap.add_argument("--videos_dir", default=VIDEOS_DIR, help="Folder of videos to run inference on")
    ap.add_argument("--out_dir", default=OUT_CSV_DIR, help="Output folder for per-video CSVs")
    ap.add_argument("--video", default=None, help="(Optional) single video path (overrides --videos_dir)")
    ap.add_argument("--out_csv", default=None, help="(Optional) single output CSV path (overrides --out_dir)")
    ap.add_argument("--ckpt", default=CKPT_PATH, help="Model checkpoint .pt")
    ap.add_argument("--device", default=DEVICE)
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
    ap.add_argument(
        "--max_frames",
        type=int,
        default=MAX_FRAMES,
        help="Process at most this many frames per video (after --start). Default: no limit.",
    )
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--score_thresh", type=float, default=0.3)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--time_col", default=TIME_COL, choices=["frame", "t"])
    ap.add_argument(
        "--include_score",
        action="store_true",
        default=bool(INCLUDE_SCORE),
        help="Add a 'score' column to the output CSV.",
    )
    ap.add_argument(
        "--no_score",
        dest="include_score",
        action="store_false",
        help="Do not add a 'score' column (default).",
    )
    ap.add_argument(
        "--emit_traj_id",
        action="store_true",
        default=bool(EMIT_TRAJ_ID),
        help="Assign a trajectory id by linking detections using the model's tracking head.",
    )
    ap.add_argument(
        "--no_traj_id",
        dest="emit_traj_id",
        action="store_false",
        help="Do not add a 'traj_id' column (default).",
    )
    ap.add_argument(
        "--max_track_dist",
        type=float,
        default=float(MAX_TRACK_DIST),
        help="Max distance (in resized pixels) for linking to previous frame.",
    )
    ap.add_argument(
        "--use_video_filename",
        action="store_true",
        default=bool(OUTPUT_USE_VIDEO_FILENAME),
        help="Name CSVs like '<video>.mp4.csv' instead of '<video>.csv'.",
    )
    ap.add_argument(
        "--use_video_stem",
        dest="use_video_filename",
        action="store_false",
        help="Name CSVs like '<video>.csv' (default).",
    )
    ap.add_argument(
        "--render_videos",
        action="store_true",
        default=bool(RENDER_VIDEOS),
        help="After writing CSV(s), write an annotated .mp4 for each input video.",
    )
    ap.add_argument(
        "--no_render_videos",
        dest="render_videos",
        action="store_false",
        help="Disable annotated video rendering.",
    )
    ap.add_argument(
        "--out_video_dir",
        default=OUT_VIDEO_DIR,
        help="Output folder for rendered videos (default: same as CSV output folder).",
    )
    ap.add_argument(
        "--out_video",
        default=None,
        help="(Optional) single output video path (overrides --out_video_dir).",
    )
    ap.add_argument(
        "--out_video_suffix",
        default=OUT_VIDEO_SUFFIX,
        help="Rendered video filename suffix (default: '_annotated').",
    )
    ap.add_argument(
        "--out_video_codec",
        default=OUT_VIDEO_CODEC,
        help="FourCC codec for rendered video (default: mp4v).",
    )
    ap.add_argument(
        "--bbox_thickness_px",
        type=int,
        default=int(BBOX_THICKNESS_PX),
        help="Bounding box thickness in pixels (default: 1).",
    )
    args = ap.parse_args()

    device = _device_from_args(args.device)

    if not str(args.ckpt).strip():
        raise ValueError("Provide --ckpt (or set CKPT_PATH at the top of the script).")
    ckpt_path = Path(args.ckpt).expanduser()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
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

    if args.video is not None or args.out_csv is not None:
        if not args.video or not args.out_csv:
            raise ValueError("--video and --out_csv must be provided together (single-video mode).")
        tasks = [(Path(args.video).expanduser(), Path(args.out_csv).expanduser())]
    else:
        if not str(args.videos_dir).strip() or not str(args.out_dir).strip():
            raise ValueError(
                "Provide --videos_dir and --out_dir (or set VIDEOS_DIR/OUT_CSV_DIR at the top of the script)."
            )
        out_dir = Path(args.out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        videos = _list_videos(Path(args.videos_dir))
        tasks = [(v, _default_out_csv(v, out_dir, bool(args.use_video_filename))) for v in videos]

    if args.out_video is not None and len(tasks) != 1:
        raise ValueError("--out_video can only be used in single-video mode (use --out_video_dir).")

    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover
        tqdm = None

    step = max(1, int(args.step))
    start_req = max(0, int(args.start))
    end_req = int(args.end)
    max_frames_req = args.max_frames
    if max_frames_req is not None:
        max_frames_req = int(max_frames_req)
        if max_frames_req <= 0:
            max_frames_req = None

    # Pre-scan to compute total work for a single overall progress bar.
    task_infos = []
    total_iters = 0
    for video_path, out_csv in tasks:
        video_info = get_video_info(video_path)
        end = int(end_req) if int(end_req) >= 0 else video_info.frame_count
        start = start_req
        end = min(video_info.frame_count, end)
        if max_frames_req is not None:
            end = min(end, start + max_frames_req)
        total_iters += _range_len(start, end, step)
        task_infos.append((video_info, out_csv, start, end))

    pbar = None
    if tqdm is not None and total_iters > 0:
        pbar = tqdm(total=total_iters, desc="inference", unit="frame", ncols=100)

    processed = 0

    import cv2

    for video_info, out_csv, start, end in task_infos:
        if pbar is not None:
            pbar.set_postfix_str(video_info.path.name)

        reader = VideoClipReader(video_info.path, frame_count=video_info.frame_count)

        sx = video_info.width / float(args.resize_w)
        sy = video_info.height / float(args.resize_h)

        out_csv = Path(out_csv).expanduser()
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["x", "y", "w", "h", args.time_col]
        if args.include_score:
            fieldnames.append("score")
        if args.emit_traj_id:
            fieldnames.append("traj_id")

        boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}

        with out_csv.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fieldnames)
            wr.writeheader()

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
                clip = (
                    torch.from_numpy(clip_thwc).permute(3, 0, 1, 2).unsqueeze(0).to(device)
                )  # [1,3,T,H,W]

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
                    row = {
                        "x": d.x * sx,
                        "y": d.y * sy,
                        "w": d.w * sx,
                        "h": d.h * sy,
                        args.time_col: frame_index,
                    }
                    if args.include_score:
                        row["score"] = d.score
                    if traj_id is not None:
                        row["traj_id"] = traj_id
                    wr.writerow(row)

                    if args.render_videos:
                        boxes_by_frame.setdefault(int(frame_index), []).append(
                            (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
                        )

                if pbar is not None:
                    pbar.update(1)
                elif total_iters > 0:
                    processed += 1
                    if processed % 100 == 0 or processed == total_iters:
                        pct = 100.0 * processed / float(total_iters)
                        print(
                            f"\rProgress: {processed}/{total_iters} ({pct:.1f}%)",
                            end="" if processed < total_iters else "\n",
                            flush=True,
                        )

        reader.close()

        if args.render_videos:
            if args.out_video is not None:
                out_video = Path(args.out_video).expanduser()
            else:
                out_video_dir = (
                    Path(args.out_video_dir).expanduser()
                    if str(args.out_video_dir).strip()
                    else out_csv.parent
                )
                out_video = _default_out_video(video_info.path, out_video_dir, str(args.out_video_suffix))

            _render_annotated_video(
                video_info=video_info,
                out_path=out_video,
                boxes_by_frame=boxes_by_frame,
                start=int(start),
                end=int(end),
                bbox_thickness_px=int(args.bbox_thickness_px),
                codec=str(args.out_video_codec),
                tqdm=(tqdm if (tqdm is not None and pbar is None) else None),
            )
            msg = f"Wrote annotated video -> {out_video}"
            if tqdm is not None and pbar is not None:
                tqdm.write(msg)
            else:
                print(msg, flush=True)

    if pbar is not None:
        pbar.close()


if __name__ == "__main__":
    main()
