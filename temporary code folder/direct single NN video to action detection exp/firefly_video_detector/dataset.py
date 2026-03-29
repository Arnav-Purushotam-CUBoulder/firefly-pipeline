from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset

from .annotations import Box
from .targets import generate_centernet_targets
from .video_io import VideoClipReader


@dataclass(frozen=True)
class DatasetSplit:
    start_frame: int
    end_frame_exclusive: int


class FireflyVideoCenternetDataset(Dataset):
    """
    PyTorch Dataset that returns (clip, targets) for a given center frame.

    clip:  FloatTensor [3, T, H, W] in resized coordinates (0..1).
    targets:
      - heatmap: FloatTensor [1, H', W']
      - wh:      FloatTensor [2, H', W']   (width/height in resized pixels)
      - offset:  FloatTensor [2, H', W']
      - mask:    FloatTensor [1, H', W']   (1 at object centers)
      - tracking:      FloatTensor [2, H', W']   (prev_center - curr_center in output coords)
      - tracking_mask: FloatTensor [1, H', W']   (1 where tracking is supervised)
    """

    def __init__(
        self,
        *,
        video_path: str | Path,
        annotations: dict[int, list[Box]],
        total_frames: int,
        orig_size: tuple[int, int],
        split: DatasetSplit,
        resize_hw: tuple[int, int] = (256, 256),
        clip_len: int = 8,
        frame_stride: int = 1,
        downsample: int = 4,
        augment: bool = False,
    ) -> None:
        self.video_path = Path(video_path).expanduser()
        self.annotations = annotations
        self.total_frames = int(total_frames)
        self.orig_w, self.orig_h = int(orig_size[0]), int(orig_size[1])
        self.split = split
        self.resize_h, self.resize_w = int(resize_hw[0]), int(resize_hw[1])
        self.clip_len = int(clip_len)
        self.frame_stride = int(frame_stride)
        self.downsample = int(downsample)
        self.augment = bool(augment)

        if self.resize_h % self.downsample != 0 or self.resize_w % self.downsample != 0:
            raise ValueError("resize_hw must be divisible by downsample.")

        self.frame_indices = list(
            range(max(0, split.start_frame), min(self.total_frames, split.end_frame_exclusive))
        )

        # Lazy, per-worker reader (cv2.VideoCapture is not pickleable).
        self._reader: VideoClipReader | None = None

        # Precompute per-sample weights for a balanced sampler.
        pos = sum(1 for f in self.frame_indices if self.annotations.get(f))
        neg = len(self.frame_indices) - pos
        pos_w = 0.5 / max(1, pos)
        neg_w = 0.5 / max(1, neg)
        self.sample_weights = [pos_w if self.annotations.get(f) else neg_w for f in self.frame_indices]

    def __len__(self) -> int:
        return len(self.frame_indices)

    def _get_reader(self) -> VideoClipReader:
        if self._reader is None:
            self._reader = VideoClipReader(self.video_path, frame_count=self.total_frames)
        return self._reader

    def _resize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.resize(rgb, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

    def _augment_clip_and_boxes(
        self,
        clip_thwc: np.ndarray,
        boxes_xywh: list[Box],
        prev_boxes_by_traj: dict[int, Box],
    ) -> tuple[np.ndarray, list[Box], dict[int, Box]]:
        # clip_thwc: [T,H,W,3] uint8
        if not self.augment:
            return clip_thwc, boxes_xywh, prev_boxes_by_traj

        import cv2

        h, w = clip_thwc.shape[1], clip_thwc.shape[2]

        # Random horizontal flip
        if np.random.rand() < 0.5:
            clip_thwc = clip_thwc[:, :, ::-1, :]
            boxes_xywh = [
                Box(
                    x=(w - b.x - b.w),
                    y=b.y,
                    w=b.w,
                    h=b.h,
                    traj_id=b.traj_id,
                )
                for b in boxes_xywh
            ]
            prev_boxes_by_traj = {
                tid: Box(
                    x=(w - b.x - b.w),
                    y=b.y,
                    w=b.w,
                    h=b.h,
                    traj_id=b.traj_id,
                )
                for tid, b in prev_boxes_by_traj.items()
            }

        # Random vertical flip
        if np.random.rand() < 0.5:
            clip_thwc = clip_thwc[:, ::-1, :, :]
            boxes_xywh = [
                Box(
                    x=b.x,
                    y=(h - b.y - b.h),
                    w=b.w,
                    h=b.h,
                    traj_id=b.traj_id,
                )
                for b in boxes_xywh
            ]
            prev_boxes_by_traj = {
                tid: Box(
                    x=b.x,
                    y=(h - b.y - b.h),
                    w=b.w,
                    h=b.h,
                    traj_id=b.traj_id,
                )
                for tid, b in prev_boxes_by_traj.items()
            }

        # Mild brightness/contrast jitter (same for all frames)
        if np.random.rand() < 0.5:
            alpha = float(np.random.uniform(0.8, 1.2))  # contrast
            beta = float(np.random.uniform(-10, 10))  # brightness
            clip_thwc = np.clip(alpha * clip_thwc.astype(np.float32) + beta, 0, 255).astype(
                np.uint8
            )

        # Small gaussian blur sometimes (acts like denoise augmentation)
        if np.random.rand() < 0.2:
            k = int(np.random.choice([3, 5]))
            clip_thwc = np.stack([cv2.GaussianBlur(f, (k, k), 0) for f in clip_thwc], axis=0)

        return clip_thwc, boxes_xywh, prev_boxes_by_traj

    def __getitem__(self, idx: int):
        center_frame = int(self.frame_indices[idx])
        reader = self._get_reader()
        frames_rgb = reader.read_rgb_clip(
            center_frame=center_frame,
            clip_len=self.clip_len,
            frame_stride=self.frame_stride,
            pad_mode="edge",
        )

        # Resize frames
        resized_frames = [self._resize_rgb(f) for f in frames_rgb]
        clip_thwc = np.stack(resized_frames, axis=0)  # [T,H,W,3] uint8

        # Scale boxes (orig -> resized)
        sx = self.resize_w / max(1.0, float(self.orig_w))
        sy = self.resize_h / max(1.0, float(self.orig_h))
        orig_boxes = self.annotations.get(center_frame, [])
        boxes_xywh = [
            Box(x=b.x * sx, y=b.y * sy, w=b.w * sx, h=b.h * sy, traj_id=b.traj_id)
            for b in orig_boxes
        ]

        # Previous-frame boxes keyed by traj_id (for tracking supervision).
        prev_boxes_by_traj: dict[int, Box] = {}
        prev_frame = center_frame - self.frame_stride
        if (
            prev_frame >= 0
            and self.split.start_frame <= prev_frame < self.split.end_frame_exclusive
            and prev_frame < self.total_frames
        ):
            needed_ids = {b.traj_id for b in orig_boxes if b.traj_id is not None}
            if needed_ids:
                for b in self.annotations.get(prev_frame, []):
                    if b.traj_id is None or b.traj_id not in needed_ids:
                        continue
                    if b.traj_id in prev_boxes_by_traj:
                        continue
                    prev_boxes_by_traj[b.traj_id] = Box(
                        x=b.x * sx, y=b.y * sy, w=b.w * sx, h=b.h * sy, traj_id=b.traj_id
                    )

        # Augment clip and adjust boxes
        clip_thwc, boxes_xywh, prev_boxes_by_traj = self._augment_clip_and_boxes(
            clip_thwc, boxes_xywh, prev_boxes_by_traj
        )

        # Build targets (center frame)
        targets_np = generate_centernet_targets(
            boxes_xywh,
            input_size=(self.resize_h, self.resize_w),
            downsample=self.downsample,
        )

        # Tracking targets: predict prev_center - curr_center (in OUTPUT coords) at each center.
        out_h = self.resize_h // self.downsample
        out_w = self.resize_w // self.downsample
        tracking = np.zeros((2, out_h, out_w), dtype=np.float32)
        tracking_mask = np.zeros((1, out_h, out_w), dtype=np.float32)
        for b in boxes_xywh:
            if b.traj_id is None:
                continue
            prev = prev_boxes_by_traj.get(b.traj_id)
            if prev is None:
                continue

            curr_cx = b.x + b.w * 0.5
            curr_cy = b.y + b.h * 0.5
            prev_cx = prev.x + prev.w * 0.5
            prev_cy = prev.y + prev.h * 0.5

            curr_cx_out = curr_cx / self.downsample
            curr_cy_out = curr_cy / self.downsample
            prev_cx_out = prev_cx / self.downsample
            prev_cy_out = prev_cy / self.downsample

            cx_int = int(curr_cx_out)
            cy_int = int(curr_cy_out)
            if not (0 <= cx_int < out_w and 0 <= cy_int < out_h):
                continue

            tracking[0, cy_int, cx_int] = prev_cx_out - curr_cx_out
            tracking[1, cy_int, cx_int] = prev_cy_out - curr_cy_out
            tracking_mask[0, cy_int, cx_int] = 1.0

        targets_np["tracking"] = tracking
        targets_np["tracking_mask"] = tracking_mask

        clip = torch.from_numpy(clip_thwc.astype(np.float32) / 255.0).permute(3, 0, 1, 2)
        targets = {k: torch.from_numpy(v) for k, v in targets_np.items()}
        targets["frame_index"] = torch.tensor(center_frame, dtype=torch.long)

        return clip, targets
