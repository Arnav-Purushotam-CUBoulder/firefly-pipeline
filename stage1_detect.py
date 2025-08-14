import csv, math, sys
from pathlib import Path
import cv2
import numpy as np

# ─── tiny progress bar ─────────────────────────────────────────
BAR_LEN = 50
def progress(i, total, tag=''):
    frac = i / total if total else 0
    bar  = '=' * int(frac * BAR_LEN) + ' ' * (BAR_LEN - int(frac * BAR_LEN))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i == total: sys.stdout.write('\n')

# ─── SimpleBlobDetector factory ────────────────────────────────
def _make_blob_detector(min_area, max_area, min_dist=0.25, min_repeat=1):
    p = cv2.SimpleBlobDetector_Params()

    # Work in grayscale; don't force a color match
    p.filterByColor = False

    # Area filter keeps absurd sizes out; keep it permissive
    p.filterByArea = True
    p.minArea = float(min_area)
    p.maxArea = float(max_area)

    # Shape filters off
    p.filterByCircularity = False
    p.filterByConvexity  = False
    p.filterByInertia    = False

    # Fine threshold sweep so dim/soft flashes show up
    p.minThreshold  = 0
    p.maxThreshold  = 255
    p.thresholdStep = 1
    p.minRepeatability = int(min_repeat)

    # Allow keypoints close to each other (flash core + halo)
    p.minDistBetweenBlobs = float(min_dist)

    return cv2.SimpleBlobDetector_create(p)

# ─── CSV writer ───────────────────────────────────────────────
def _write_csv(csv_path: Path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'x', 'y', 'w', 'h'])
        w.writerows(rows)

# ─── public API (called by orchestrator) ──────────────────────
def detect_blobs_to_csv(
    orig_path: Path,            # ← now pass the ORIGINAL video here
    csv_path: Path,
    max_frames=None,
    *,
    # detector + preproc knobs (safe defaults)
    sbd_min_area_px: float = 0.5,     # allow single‑pixel flashes
    sbd_max_area_scale: float = 1.0,  # fraction of frame area allowed
    sbd_min_dist: float = 0.25,       # keypoint separation
    sbd_min_repeat: int = 1,          # keep low; avoids single-threshold miss
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_tile=(8,8),
    # optional band‑pass: set both False to skip
    use_tophat: bool = False,
    tophat_ksize: int = 7,            # odd (5,7,9…)
    use_dog: bool = False,
    dog_sigma1: float = 0.8,
    dog_sigma2: float = 1.6,
):
    """
    Detect bright blobs directly on the ORIGINAL video using SimpleBlobDetector.
    Writes CSV rows: [frame, x, y, w, h] where (x,y,w,h) is a tight-ish square
    around the SBD keypoint size estimate.
    """
    cap = cv2.VideoCapture(str(orig_path))
    if not cap.isOpened():
        sys.exit(f"Could not open ORIGINAL video: {orig_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None:
        total = min(total, max_frames)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_area = int(W * H * sbd_max_area_scale) if sbd_max_area_scale > 0 else (W * H)
    detector = _make_blob_detector(
        min_area=sbd_min_area_px,
        max_area=max_area,
        min_dist=sbd_min_dist,
        min_repeat=sbd_min_repeat
    )

    detections = []
    fr = 0
    while True:
        if max_frames is not None and fr > max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---- Preprocessing (no blur!)
        inp = gray
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
            inp = clahe.apply(inp)

        if use_tophat:
            if tophat_ksize < 3 or tophat_ksize % 2 == 0:
                tophat_ksize = 7
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_ksize, tophat_ksize))
            inp = cv2.morphologyEx(inp, cv2.MORPH_TOPHAT, k)

        if use_dog:
            g1 = cv2.GaussianBlur(inp, (0,0), dog_sigma1)
            g2 = cv2.GaussianBlur(inp, (0,0), dog_sigma2)
            inp = cv2.subtract(g1, g2)

        # ---- Detect
        kps = detector.detect(inp)

        for kp in kps:
            cx, cy = kp.pt
            side   = int(max(3, min(math.ceil(kp.size), min(W, H))))
            x = int(round(cx - side/2))
            y = int(round(cy - side/2))
            x = max(0, min(x, W - side))
            y = max(0, min(y, H - side))
            detections.append([fr, x, y, side, side])

        progress(fr+1, total, 'detect(ORIG)'); fr += 1

    cap.release()
    _write_csv(csv_path, detections)
