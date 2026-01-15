from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


def get_video_info(video_path: str | Path) -> VideoInfo:
    """
    Read basic metadata from a video file using OpenCV.
    """
    video_path = Path(video_path).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    import cv2  # local import so py_compile works without cv2 installed

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

        if frame_count <= 0:
            raise RuntimeError(
                "OpenCV returned 0 frame count; consider re-encoding the video."
            )
        if width <= 0 or height <= 0:
            raise RuntimeError("Could not read video width/height.")
    finally:
        cap.release()

    return VideoInfo(
        path=video_path,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
    )


class VideoClipReader:
    """
    Simple random-access clip reader.

    Notes:
    - The VideoCapture is opened lazily on first read to remain DataLoader-spawn safe.
    - Frames are returned as RGB numpy arrays in ORIGINAL video resolution.
    """

    def __init__(self, video_path: str | Path, frame_count: int | None = None):
        self.video_path = str(Path(video_path).expanduser())
        self.frame_count = frame_count
        self._cap = None

    def _get_cap(self):
        if self._cap is None:
            import cv2

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {self.video_path}")
            self._cap = cap
        return self._cap

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def read_rgb_frame(self, frame_index: int):
        import cv2

        cap = self._get_cap()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise IndexError(f"Could not read frame {frame_index}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def read_rgb_clip(
        self,
        center_frame: int,
        clip_len: int,
        frame_stride: int = 1,
        pad_mode: str = "edge",
    ):
        if clip_len <= 0:
            raise ValueError("clip_len must be > 0")
        if frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")

        half = clip_len // 2
        indices = [
            center_frame + (i - half) * frame_stride for i in range(clip_len)
        ]

        if self.frame_count is not None:
            last = self.frame_count - 1
            if pad_mode == "edge":
                indices = [min(max(0, idx), last) for idx in indices]
            else:
                indices = [idx for idx in indices if 0 <= idx <= last]

        frames = [self.read_rgb_frame(idx) for idx in indices]
        return frames

