import csv, sys
from collections import defaultdict
from pathlib import Path
import cv2

BAR_LEN = 50
def progress(i, total, tag=''):
    frac = i / total if total else 0
    bar  = '=' * int(frac * BAR_LEN) + ' ' * (BAR_LEN - int(frac * BAR_LEN))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i == total: sys.stdout.write('\n')

def render_fixed_10px_from_csv(
    video_path: Path,
    csv_path: Path,
    out_path: Path,
    *,
    thickness=2,
    max_frames=None,
    color=(0,0,255),                # firefly color (red by default for orig)
    draw_background=True,
    background_color=(0,255,0)      # background color (green)
):
    """Render a video where every CSV row is drawn as a 10×10 box.

    CSV schema supported:
      - frame,x,y,w,h
      - optional: class (firefly|background)
      - optional: xy_semantics ('center' means x,y are centroids; otherwise treated as top-left)

    Semantics:
      • If xy_semantics == 'center', we center the 10×10 box at (x,y).
      • Otherwise, we treat (x,y) as top-left and compute center as (x+w/2, y+h/2).
    """
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    has_class  = ('class' in rows[0].keys())
    has_xy_sem = ('xy_semantics' in rows[0].keys())

    # group by frame for efficient playback
    by_frame = defaultdict(list)
    for r in rows:
        try:
            f = int(r['frame'])
            if max_frames is not None and f > max_frames:
                continue
            x = float(r['x']); y = float(r['y'])
            w = int(float(r['w'])); h = int(float(r['h']))
            cls = r.get('class', 'firefly') if has_class else None
            xy_sem = (r.get('xy_semantics','') if has_xy_sem else '')
            by_frame[f].append((x,y,w,h,cls,xy_sem))
        except Exception:
            continue

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    total = min(total_video_frames, max(by_frame.keys(), default=0) or total_video_frames)
    if max_frames is not None:
        total = min(total, max_frames)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))

    FIX_W = 10; FIX_H = 10

    fr = 0
    while True:
        if max_frames is not None and fr > max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        for x, y, w, h, cls, xy_sem in by_frame.get(fr, []):
            # compute center depending on semantics
            if isinstance(xy_sem, str) and xy_sem.lower() == 'center':
                cx, cy = float(x), float(y)
            else:
                cx = float(x) + w / 2.0
                cy = float(y) + h / 2.0

            # build fixed 10×10 box around center
            x0 = int(round(cx - FIX_W / 2.0))
            y0 = int(round(cy - FIX_H / 2.0))

            # clamp to frame
            x0 = max(0, min(x0, W - FIX_W))
            y0 = max(0, min(y0, H - FIX_H))

            # choose color
            if cls is None or cls == 'firefly':
                cv2.rectangle(frame, (x0, y0), (x0 + FIX_W, y0 + FIX_H), color, thickness)
            elif cls == 'background' and draw_background:
                cv2.rectangle(frame, (x0, y0), (x0 + FIX_W, y0 + FIX_H), background_color, thickness)

        out.write(frame)
        progress(fr+1, total, 'render-10px'); fr += 1

    cap.release(); out.release()
