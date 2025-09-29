#!/usr/bin/env python3
import csv, sys, math
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from audit_trail import AuditTrail

BAR_LEN = 50
def progress(i, total, tag=''):
    total = max(1, int(total or 1))
    i = min(i, total)
    frac = min(1.0, max(0.0, i / float(total)))
    bar  = '=' * int(frac * BAR_LEN) + ' ' * (BAR_LEN - int(frac * BAR_LEN))
    sys.stdout.write(f'\r{tag} [{bar}] {int(frac*100):3d}%')
    sys.stdout.flush()
    if i >= total:
        sys.stdout.write('\n')

def _device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _build_backbone(backbone: str) -> nn.Module:
    b = (backbone or 'resnet18').lower()
    fns = {
        'resnet18':  models.resnet18,
        'resnet34':  models.resnet34,
        'resnet50':  models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }
    fn = fns.get(b, models.resnet18)
    net = fn(weights=None)
    in_f = net.fc.in_features
    net.fc = nn.Linear(in_f, 2)
    return net

def _softmax_fire_prob(bg_logit: float, ff_logit: float) -> float:
    m = max(bg_logit, ff_logit)
    eb = math.exp(bg_logit - m)
    ef = math.exp(ff_logit - m)
    denom = eb + ef
    return ef / denom if denom > 0 else 0.5

def _center_crop_clamped_bgr_to_pil(img_bgr, cx: float, cy: float, w: int, h: int) -> Image.Image:
    H, W = img_bgr.shape[:2]
    w = max(1, int(round(w))); h = max(1, int(round(h)))
    x0 = int(round(cx - w/2.0)); y0 = int(round(cy - h/2.0))
    x0 = max(0, min(x0, W - w)); y0 = max(0, min(y0, H - h))
    crop = img_bgr[y0:y0+h, x0:x0+w]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

@torch.no_grad()
def classify_and_filter_csv(
    orig_path: Path, csv_path: Path, *,
    max_frames: Optional[int] = None,
    use_cnn_filter: bool = True,
    model_path: Optional[Path] = None,
    backbone: str = 'resnet18',
    class_to_keep: int = 1,         # FIREFLY index
    patch_w: int = 10,
    patch_h: int = 10,
    patch_batch_size: int = 0,      # 0/<=0 = per-frame batches; >0 = cross-frame patch batching
    firefly_conf_thresh: float = 0.5,
    drop_background_rows: bool = False,
    imagenet_normalize: bool = False,       # set True only if your model expects it
    print_load_status: bool = True,
    fail_if_weights_missing: bool = True,
    debug_save_patches_dir: Optional[Path] = None,
    audit: Optional[AuditTrail] = None,
):
    """Classify detections and overwrite csv_path with added:
       background_logit, firefly_logit, firefly_confidence, and class.
       Semantics:
         - If 'xy_semantics' == 'center', (x,y) is the crop CENTER; else use (x+w/2, y+h/2).
         - Crops are patch_w×patch_h (default 10×10).
    """
    if not use_cnn_filter:
        return

    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return

    # Group rows by zero-based frame
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    for r in rows:
        try:
            f = int(r['frame'])
            if max_frames is not None and f >= max_frames:
                continue
            by_frame[f].append(r)
        except Exception:
            continue

    # Build model and load weights STRICTLY (match Stage 9)
    device = _device()
    model = _build_backbone(backbone).to(device)
    loaded_ok = False
    if model_path:
        try:
            sd = torch.load(str(model_path), map_location=device)
            for key in ('state_dict','model','net','weights'):
                if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break
            missing, unexpected = model.load_state_dict(sd, strict=True)
            loaded_ok = True
            if print_load_status:
                print(f"[stage4] Loaded weights OK from: {model_path}")
        except Exception as e:
            print(f"[stage4] ERROR: failed to load weights from {model_path}: {e}")
    if not loaded_ok:
        msg = "[stage4] Weights not loaded; refusing to run CNN (would yield ~0.5 confidences)."
        if fail_if_weights_missing:
            raise RuntimeError(msg)
        else:
            print(msg)

    model.eval()

    # Transform (match Stage 9 / your standalone script): ToTensor only by default
    tfms = [T.ToTensor()]
    if imagenet_normalize:
        tfms.append(T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
    transform = T.Compose(tfms)

    # Video
    cap = cv2.VideoCapture(str(orig_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {orig_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    has_xy_sem = ('xy_semantics' in rows[0].keys())

    labeled_rows: List[dict] = []

    # Helper to flush an accumulated cross-frame patch batch
    def _flush(pending_tensors: List[torch.Tensor], pending_meta: List[Tuple[dict, float, float]]):
        if not pending_tensors:
            return [], []
        batch = torch.stack(pending_tensors, dim=0).to(device)
        logits = model(batch)
        FIRE_IDX = int(class_to_keep)
        BACK_IDX = 1 - FIRE_IDX
        out_rows: List[dict] = []
        for i, (r, _cx, _cy) in enumerate(pending_meta):
            lg = logits[i].detach().float().cpu().numpy().tolist()
            bg = float(lg[BACK_IDX]); ff = float(lg[FIRE_IDX])
            conf = _softmax_fire_prob(bg, ff)
            cls  = 'firefly' if conf >= float(firefly_conf_thresh) else 'background'
            out = dict(r)
            out['class'] = cls
            out['background_logit'] = bg
            out['firefly_logit'] = ff
            out['firefly_confidence'] = conf
            out_rows.append(out)
        return out_rows, []

    # Accumulator for cross-frame batching (enabled if patch_batch_size > 0)
    pending_tensors: List[torch.Tensor] = []
    pending_meta: List[Tuple[dict, float, float]] = []

    fr = 0
    while True:
        if max_frames is not None and fr >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break

        dets = by_frame.get(fr, [])
        if dets:
            for r in dets:
                try:
                    x = float(r['x']); y = float(r['y'])
                    w = int(float(r['w'])); h = int(float(r['h']))
                except Exception:
                    continue

                if has_xy_sem and str(r.get('xy_semantics','')).lower() == 'center':
                    cx, cy = x, y
                else:
                    cx, cy = x + w/2.0, y + h/2.0

                pil = _center_crop_clamped_bgr_to_pil(frame, cx, cy, patch_w, patch_h)
                if debug_save_patches_dir:
                    debug_save_patches_dir.mkdir(parents=True, exist_ok=True)
                    pil.save(debug_save_patches_dir / f"f{fr:06d}_cx{int(round(cx))}_cy{int(round(cy))}.png")

                ten = transform(pil)
                if patch_batch_size and patch_batch_size > 0:
                    pending_tensors.append(ten)
                    pending_meta.append((r, cx, cy))
                    if len(pending_tensors) >= int(patch_batch_size):
                        outs, _ = _flush(pending_tensors, pending_meta)
                        labeled_rows.extend(outs)
                        pending_tensors.clear(); pending_meta.clear()
                else:
                    # Per-frame behavior: run immediately when this frame is done
                    pending_tensors.append(ten)
                    pending_meta.append((r, cx, cy))

        # If per-frame batching, flush now
        if not (patch_batch_size and patch_batch_size > 0) and pending_tensors:
            outs, _ = _flush(pending_tensors, pending_meta)
            labeled_rows.extend(outs)
            pending_tensors.clear(); pending_meta.clear()

        progress(fr+1, max_frames or total_frames, 'stage4'); fr += 1

    cap.release()

    # Flush any remaining cross-frame patches
    if pending_tensors:
        outs, _ = _flush(pending_tensors, pending_meta)
        labeled_rows.extend(outs)

    # audit: log all scores before any drop
    if audit is not None and labeled_rows:
        audit.log_kept('04_cnn', labeled_rows,
                       extra_cols=['class','background_logit','firefly_logit','firefly_confidence'],
                       filename_suffix='scores')

    if drop_background_rows:
        before_drop = list(labeled_rows)
        labeled_rows = [r for r in labeled_rows if r.get('class') == 'firefly']
        if audit is not None:
            removed = [r for r in before_drop if r.get('class') != 'firefly']
            if removed:
                audit.log_removed('04_cnn', 'classified_background', removed,
                                  extra_cols=['class','background_logit','firefly_logit','firefly_confidence'])

    # Preserve original order, append extras at end
    orig_fields = list(rows[0].keys())
    base = ['frame','x','y','w','h']
    ordered = [c for c in base if c in orig_fields] + [c for c in orig_fields if c not in base]
    extras = ['class','background_logit','firefly_logit','firefly_confidence']
    fieldnames: List[str] = []
    for c in ordered + extras:
        if c not in fieldnames:
            fieldnames.append(c)

    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in labeled_rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
