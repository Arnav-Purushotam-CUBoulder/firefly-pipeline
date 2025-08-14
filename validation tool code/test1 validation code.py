#TODO:
# fix the overlap problem of FP and FP
# fix the overlap problem of FP and FN
# fix the overlap problem of TP and FN (this happens because there were two squares on the same bright large firefly in the 
# in the ground truth annotations, these werent merged in the original pre processing, because they were identical, so
# just based on like 40% overlap merge them based on centroids distance), so basically they're just mistakes in annotation



#!/usr/bin/env python3
"""
bbox_eval.py – Comprehensive verification tool for object-detection CSVs.

Coordinate conventions
----------------------
• **Ground-truth CSV** (`GT_CSV`)
      (x, y) = centre of the w × h box (converted internally to top-left).
• **Prediction CSV** (`PRED_CSV`)
      (x, y) = top-left corner of the w × h box (no conversion needed).

Install deps:
    pip install numpy pandas matplotlib pycocotools tqdm scikit-learn pillow

Author: 2025
"""

# ─────────────────── GLOBAL CONFIG ─────────────────── #
PRED_CSV            = '/Users/arnavps/Desktop/forresti fixing FPs POC/to send data/filtered_output.csv'     # x,y,w,h,frame[,conf,class]
GT_CSV              = "/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/code with resnet forresti/validation tool code/converted csvs forresti C0107 100 frames/above 50 threshold csvs/extended_ground_truth_above_50_threshold.csv"      # x,y,w,h,frame[,class]

# folder containing the source frame images for all crops
IMAGE_ROOT          = '/Users/arnavps/Desktop/to annotate frames/forresti'

# root for all generated reports, plots, and crops
REPORT_DIR          = "/Users/arnavps/Desktop/RA info/New Deep Learning project/TESTING_CODE/background subtraction detection method/actual background subtraction code/forresti, fixing FPs and box overlap/Proof of concept code/test1/validation tool code/reports"

# where per-IoU TP/FP/FN crops will go
IOU_CROPS_ROOT      = '/Users/arnavps/Desktop/forresti fixing FPs POC/to send data/temp to delete'

# NEW: master switch – set to False to skip writing per-IoU TP/FP/FN crops
SAVE_IOU_CROPS      = False

# save per-threshold TP / FP / FN tables?
SAVE_MATCH_CSVS   = True          # False → skip writing
CSV_NAME_PATTERN  = "matches_{kind}_{thr:.2f}.csv"


# sweep from 0.40 to 0.95 in 0.05 steps
import numpy as _np
IOU_THRESHOLDS = [round(t, 2) for t in _np.arange(0.40, 0.96, 0.05)]

CONF_THRESH         = 0.0                             # ignore predictions below this
SIZE_BUCKETS        = dict(small=0.0, medium=32**2, large=96**2)
RELIABILITY_NBINS   = 10
SAVE_ERROR_CROPS    = False
MAX_CROPS_PER_TYPE  = 50
RANDOM_SEED         = 1337

# Defaults for minimal five-column CSVs
DEFAULT_CLASS       = "firefly"
DEFAULT_CONF        = 1.0
REQUIRED_COLUMNS    = ["x", "y", "w", "h", "frame"]
OPTIONAL_COLUMNS    = ["conf", "class"]

CROP_IOU_ONLY      = [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]          # change this at will

# ───────────── IMPORTS ───────────── #
import os, json, shutil, math, itertools, warnings, random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import precision_recall_curve, auc
from PIL import Image, ImageDraw                       # ← added ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ───────────── UTILITIES ───────────── #
def xywh_to_xyxy(row):
    """Convert (x,y,w,h) to (xmin,ymin,xmax,ymax)."""
    x, y, w, h = row[0:4]
    return (x, y, x + w, y + h)

def area(box):
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])

def iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter  = max(0, xb-xa) * max(0, yb-ya)
    union  = area(box_a) + area(box_b) - inter
    return inter / union if union else 0.0

def save_fullframe(img, bbox, out_path, colour="red", thickness=1):
    """
    Draw a single rectangle on a *copy* of the frame and save it.
    Always writes PNG to avoid JPEG colour-bleed.
    """
    if img is None:
        return
    im   = img.copy()
    draw = ImageDraw.Draw(im)
    x1, y1, x2, y2 = map(int, bbox)
    draw.rectangle([x1, y1, x2, y2],
                   outline=colour,
                   width=thickness)          # ← adjust thickness if you like

    # force PNG so edges stay razor-sharp
    png_path = os.path.splitext(out_path)[0] + ".png"
    im.save(png_path, format="PNG", compress_level=1)

# ──────── LOADING & PRE-PROCESSING ──────── #
def load_csv(path: str, *, is_pred: bool) -> pd.DataFrame:
    """
    Accept CSVs with at least x,y,w,h,frame.
    If 'conf' or 'class' are missing they are added with defaults.
    Ground-truth centres are converted to top-left internally.
    """
    df = pd.read_csv(path)
    df["frame"] = df["frame"].astype(str).apply(lambda s: os.path.splitext(os.path.basename(s))[0])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")

    if "conf"  not in df.columns:
        df["conf"] = DEFAULT_CONF
    if "class" not in df.columns:
        df["class"] = DEFAULT_CLASS

    # convert GT centres → top-left
    if not is_pred:
        df["x"] = df["x"] - df["w"] // 2
        df["y"] = df["y"] - df["h"] // 2

    if is_pred and "conf" in df.columns and CONF_THRESH > 0.0:
        df = df[df.conf >= CONF_THRESH]

    df['bbox_xyxy'] = df[['x', 'y', 'w', 'h']].apply(xywh_to_xyxy, axis=1)
    return df

def group_by_frame(df: pd.DataFrame):
    return {f: sub for f, sub in df.groupby('frame')}

# ──────────── GREEDY MATCHING ──────────── #
def match_frame(preds: pd.DataFrame, gts: pd.DataFrame, thr: float):
    """Return (TP list[(idx,IoU)], FP idx list, FN idx list)."""
    if preds.empty and gts.empty:
        return [], [], []
    tp, fp, fn, used = [], [], [], set()
    preds_sorted = preds.sort_values('conf', ascending=False)
    for pi, prow in preds_sorted.iterrows():
        best_iou, best_gid = 0.0, None
        for gi, grow in gts.iterrows():
            if gi in used: continue
            cur = iou(prow.bbox_xyxy, grow.bbox_xyxy)
            if cur > best_iou:
                best_iou, best_gid = cur, gi
        if best_iou >= thr:
            used.add(best_gid); tp.append((pi, best_iou))
        else:
            fp.append(pi)
    fn = [gi for gi in gts.index if gi not in used]
    return tp, fp, fn

# ──────────── EVALUATION CORE ──────────── #

def xywh_avg(a, b):
    """Return the average (x,y,w,h) of two xywh boxes."""
    return (
        int((a.x + b.x) / 2),
        int((a.y + b.y) / 2),
        int((a.w + b.w) / 2),
        int((a.h + b.h) / 2),
    )

def promote_pairs(fp_df, fn_df, thr=0.40):
    """
    Scan every FP ↔ FN pair; if IoU ≥ `thr`:
        • add an averaged box to a TP list
        • mark both originals for removal
    Returns (tp_rows, fp_pruned, fn_pruned)
    """
    tp_rows, keep_fp, keep_fn = [], [], []
    for i, fprow in fp_df.iterrows():
        matched = False
        for j, fnrow in fn_df.iterrows():
            if iou(fprow.bbox_xyxy, fnrow.bbox_xyxy) >= thr:
                tp_rows.append(xywh_avg(fprow, fnrow) + (fprow.frame,))
                matched = True
                fn_df.drop(index=j, inplace=True)
                break
        if not matched:
            keep_fp.append(fprow)
    keep_fn = fn_df   # rows left after any drops
    return tp_rows, pd.DataFrame(keep_fp), keep_fn


from pathlib import Path

def evaluate(pred_df: pd.DataFrame, gt_df: pd.DataFrame):
    """
    Frame-by-frame greedy matching → TP / FP / FN.
    Extra step: after the whole pass, look at the collected FP + FN
    rows for every IoU threshold; if any FP–FN pair overlaps ≥ 40 %
    (IoU), promote **one** averaged box to TP and delete the originals.
    Metrics and CSV collections are updated in place.
    """
    frames_pred, frames_gt = group_by_frame(pred_df), group_by_frame(gt_df)
    classes = sorted(gt_df["class"].unique().tolist())
    metrics = {c: {t: Counter() for t in IOU_THRESHOLDS} for c in classes}

    # ── collectors for optional CSV dump ───────────────────────────────
    if SAVE_MATCH_CSVS:
        valid     = [t for t in IOU_THRESHOLDS if not CROP_IOU_ONLY or t in CROP_IOU_ONLY]
        tp_rows   = {t: [] for t in valid}
        fp_rows   = {t: [] for t in valid}
        fn_rows   = {t: [] for t in valid}

    # ── main greedy-matching loop ──────────────────────────────────────
    for frame in tqdm(sorted(set(frames_gt) | set(frames_pred)),
                      desc="Greedy matching"):
        preds = frames_pred.get(frame, pd.DataFrame(columns=pred_df.columns))
        gts   = frames_gt.get(frame,   pd.DataFrame(columns=gt_df.columns))
        for thr in IOU_THRESHOLDS:
            # always run matching so every IoU contributes to metrics
            tp, fp, fn = match_frame(preds, gts, thr)

            # update per-class counters (TP / FP / FN) for *all* IoUs
            for idx, _ in tp:
                metrics[preds.loc[idx, "class"]][thr]["TP"] += 1
            for idx in fp:
                metrics[preds.loc[idx, "class"]][thr]["FP"] += 1
            for idx in fn:
                metrics[gts.loc[idx,   "class"]][thr]["FN"] += 1

            # if this IoU isn’t in CROP_IOU_ONLY we’re done for this thr
            if CROP_IOU_ONLY and thr not in CROP_IOU_ONLY:
                continue

            # ── collect rows for TP / FP / FN CSVs ────────────────────
            if SAVE_MATCH_CSVS:
                if tp:
                    tp_rows[thr].append(preds.loc[[pi for pi, _ in tp]])
                if fp:
                    fp_rows[thr].append(preds.loc[fp])
                if fn:
                    fn_rows[thr].append(gts.loc[fn])

    # # ── promote FP↔FN pairs with ≥40 % overlap ─────────────────────────
    # if SAVE_MATCH_CSVS:
    #     for thr in tp_rows.keys():
    #         if not fp_rows[thr] or not fn_rows[thr]:
    #             continue
    #         fp_df = pd.concat(fp_rows[thr], ignore_index=True)
    #         fn_df = pd.concat(fn_rows[thr], ignore_index=True)

    #         new_tp, fp_pruned, fn_pruned = promote_pairs(fp_df, fn_df, thr=0.40)

    #         fp_rows[thr] = [fp_pruned] if not fp_pruned.empty else []
    #         fn_rows[thr] = [fn_pruned] if not fn_pruned.empty else []

    #         if new_tp:
    #             ntp_df = pd.DataFrame(new_tp, columns=["x", "y", "w", "h", "frame"])
    #             ntp_df["class"] = DEFAULT_CLASS
    #             ntp_df["bbox_xyxy"] = ntp_df[["x", "y", "w", "h"]].apply(xywh_to_xyxy, axis=1)
    #             tp_rows[thr].append(ntp_df)

    #             # metric corrections (one FP+FN → one TP)
    #             for _ in range(len(new_tp)):
    #                 metrics[DEFAULT_CLASS][thr]["TP"] += 1
    #                 metrics[DEFAULT_CLASS][thr]["FP"] -= 1
    #                 metrics[DEFAULT_CLASS][thr]["FN"] -= 1

    # ── optional CSV dump ──────────────────────────────────────────────
    if SAVE_MATCH_CSVS:
        out_dir = Path(REPORT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _strip(df: pd.DataFrame):
            ret = df[["x", "y", "w", "h", "frame"]].copy()
            ret["frame"] = ret["frame"].astype(str) + ".jpg"
            return ret

        for thr in tp_rows:
            if tp_rows[thr]:
                _strip(pd.concat(tp_rows[thr], ignore_index=True)).to_csv(
                    out_dir / f"matches_TP_{thr:.2f}.csv", index=False)
            if fp_rows[thr]:
                _strip(pd.concat(fp_rows[thr], ignore_index=True)).to_csv(
                    out_dir / f"matches_FP_{thr:.2f}.csv", index=False)
            if fn_rows[thr]:
                _strip(pd.concat(fn_rows[thr], ignore_index=True)).to_csv(
                    out_dir / f"matches_FN_{thr:.2f}.csv", index=False)

    return metrics, tp_rows, fp_rows, fn_rows





def precision(c): return c['TP'] / (c['TP']+c['FP']) if (c['TP']+c['FP']) else 0
def recall(c):    return c['TP'] / (c['TP']+c['FN']) if (c['TP']+c['FN']) else 0
def f1(c): p, r = precision(c), recall(c); return 2*p*r/(p+r) if (p+r) else 0

# ─────── COCO-STYLE mAP (pycocotools) ─────── #
# ─────── COCO-STYLE mAP (pycocotools) ───────
def coco_map(pred_df: pd.DataFrame, gt_df: pd.DataFrame):
    """
    Compute COCO mAP even when the two CSVs list different frames.

    Strategy
    --------
    1.  Make a sorted union of all frame names.
    2.  Give every frame an image_id.
    3.  Add GT annotations only for frames that actually have GT boxes.
        (Frames with zero GT boxes are allowed.)
    4.  Add detections only for frames that have predictions.
    """

    # 1. union of frame names  →  deterministic id
    all_frames = sorted(set(gt_df.frame) | set(pred_df.frame))
    frame2id   = {fid: idx for idx, fid in enumerate(all_frames)}

    # 2. images list
    images = [{"id": frame2id[fid], "file_name": f"{fid}.png"}
              for fid in all_frames]

    # 3. GT annotations
    annotations, categories, cat_lut, ann_id = [], [], {}, 1
    for fid, rows in gt_df.groupby("frame"):
        img_id = frame2id[fid]
        for _, r in rows.iterrows():
            cid = cat_lut.setdefault(r["class"], len(cat_lut) + 1)
            if len(categories) < cid:
                categories.append({"id": cid, "name": r["class"]})
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cid,
                "bbox": [r.x, r.y, r.w, r.h],
                "area": r.w * r.h,
                "iscrowd": 0,
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {"images": images,
                       "annotations": annotations,
                       "categories": categories}
    coco_gt.createIndex()

    # 4. detections
    dets = []
    for fid, rows in pred_df.groupby("frame"):
        img_id = frame2id[fid]          # guaranteed to exist now
        for _, r in rows.iterrows():
            dets.append({
                "image_id":   img_id,
                "category_id": cat_lut.setdefault(r["class"], 0) or 1,
                "bbox":       [r.x, r.y, r.w, r.h],
                "score":      float(r.conf),
            })
    coco_dt = coco_gt.loadRes(dets)

    # 5. evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate(); coco_eval.accumulate()
    try:    coco_eval.summarize(verbose=False)
    except TypeError: coco_eval.summarize()

    # stats[1] = AP50, stats[0] = AP@[.50:.95]
    return float(coco_eval.stats[1]), float(coco_eval.stats[0])


# ───────── PR CURVE & RELIABILITY DIAGRAM ───────── #
def pr_curve(pred_df):
    y_true = pred_df.matched.astype(int).values
    pr, re, _ = precision_recall_curve(y_true, pred_df.conf)
    return re, pr, auc(re, pr)

def reliability(pred_df):
    bins = np.linspace(0, 1, RELIABILITY_NBINS+1)
    bin_ids = np.digitize(pred_df.conf, bins) - 1
    confs, accs = [], []
    for b in range(RELIABILITY_NBINS):
        mask = bin_ids == b
        if mask.any():
            confs.append(pred_df.loc[mask,'conf'].mean())
            accs.append(pred_df.loc[mask,'matched'].mean())
    return confs, accs

# ───────── SIZE-BUCKET RECALL ───────── #
def size_bucket(a):
    if a < SIZE_BUCKETS['medium']: return 'small'
    if a < SIZE_BUCKETS['large']:  return 'medium'
    return 'large'

def stratified_recall(gt_df, matched_idx):
    gt_df['area'] = gt_df.w * gt_df.h
    gt_df['bucket'] = gt_df.area.apply(size_bucket)
    rec = {}
    for buck, g in gt_df.groupby('bucket'):
        total = len(g)
        hit = len(set(g.index) & matched_idx)
        rec[buck] = hit / total if total else 0.0
    return rec

# ───────── OPTIONAL ERROR GALLERY ───────── #
def save_crop(img, bbox, out_path):
    x1, y1, x2, y2 = map(int, bbox)
    img.crop((x1, y1, x2, y2)).save(out_path)

def build_gallery(pred_df, gt_df, thr=0.5):
    if not SAVE_ERROR_CROPS: return
    os.makedirs(f"{REPORT_DIR}/gallery/FP", exist_ok=True)
    os.makedirs(f"{REPORT_DIR}/gallery/FN", exist_ok=True)
    gt_frames = group_by_frame(gt_df)
    for frame, preds in tqdm(group_by_frame(pred_df).items(), desc="Error-gallery crops"):
        img_path = os.path.join(IMAGE_ROOT, f"{frame}.png")
        if not os.path.exists(img_path): continue
        img = Image.open(img_path).convert("RGB")
        gts = gt_frames.get(frame, pd.DataFrame())
        tp, fp, fn = match_frame(preds, gts, thr)
        for idx in fp[:MAX_CROPS_PER_TYPE]:
            save_crop(img, preds.loc[idx].bbox_xyxy, f"{REPORT_DIR}/gallery/FP/{frame}_{idx}.png")
        for idx in fn[:MAX_CROPS_PER_TYPE]:
            save_crop(img, gts.loc[idx].bbox_xyxy, f"{REPORT_DIR}/gallery/FN/{frame}_{idx}.png")

# ───────── PLOTTING ───────── #
def plot_pr(re, pr_vals, auc_val):
    plt.figure(figsize=(6,6))
    plt.plot(re, pr_vals, lw=2, label=f"AUC={auc_val:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/pr_curve.png", dpi=200); plt.close()

def plot_reliability(confs, accs):
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],'--', lw=1)
    plt.plot(confs, accs, 'o-')
    plt.xlabel("Confidence"); plt.ylabel("Empirical Accuracy")
    plt.title("Reliability Diagram"); plt.tight_layout()
    plt.savefig(f"{REPORT_DIR}/reliability.png", dpi=200); plt.close()


# -----------HELPER-------------------- #

def apply_promotions_to_dfs(pred_df, gt_df, fp_rows, fn_rows, tp_rows):
    """
    Physically remove every FP/FN row that was promoted and insert the
    averaged box into *both* dataframes so the later crop loop sees a TP.

    • fp_rows / fn_rows / tp_rows are the final dicts produced in evaluate()
      (after our promotion logic).
    • We assume DEFAULT_CLASS and DEFAULT_CONF.
    """
    for thr in tp_rows:
        if not tp_rows[thr]:
            continue
        new_tp_df = pd.concat(tp_rows[thr], ignore_index=True)

        # —— remove promoted originals from pred_df / gt_df ——
        if fp_rows[thr]:
            for _, r in pd.concat(fp_rows[thr]).iterrows():
                mask = (pred_df.x == r.x) & (pred_df.y == r.y) & \
                       (pred_df.w == r.w) & (pred_df.h == r.h) & \
                       (pred_df.frame == r.frame)
                pred_df.drop(pred_df[mask].index, inplace=True)
        if fn_rows[thr]:
            for _, r in pd.concat(fn_rows[thr]).iterrows():
                mask = (gt_df.x == r.x) & (gt_df.y == r.y) & \
                       (gt_df.w == r.w) & (gt_df.h == r.h) & \
                       (gt_df.frame == r.frame)
                gt_df.drop(gt_df[mask].index, inplace=True)

        # —— insert averaged box into BOTH dfs so they will match ——
        if not new_tp_df.empty:
            tpl = new_tp_df.copy()
            tpl["conf"]  = DEFAULT_CONF
            tpl["class"] = DEFAULT_CLASS
            tpl["bbox_xyxy"] = tpl[["x","y","w","h"]].apply(xywh_to_xyxy, axis=1)
            pred_df = pd.concat([pred_df, tpl], ignore_index=True)
            gt_df   = pd.concat([gt_df,   tpl], ignore_index=True)

    return pred_df, gt_df


def _load_crop_csv_raw(path: str) -> pd.DataFrame:
    """
    Read a TP/FP/FN CSV exactly as written by this script and return a
    dataframe with *numeric* x,y,w,h plus a bbox_xyxy column.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    header = 0 if "frame" in open(path).readline().lower() else None
    df = pd.read_csv(path,
                     names=["x", "y", "w", "h", "frame"],
                     header=header)

    # ── enforce numeric types so IoU math works ─────────────────────
    df[["x", "y", "w", "h"]] = df[["x", "y", "w", "h"]].apply(pd.to_numeric)

    df["frame"] = df["frame"].astype(str).apply(
        lambda s: os.path.splitext(os.path.basename(s))[0])

    df["bbox_xyxy"] = df[["x", "y", "w", "h"]].apply(xywh_to_xyxy, axis=1)
    df["class"]     = DEFAULT_CLASS          # CSVs don’t store class
    return df


def _write_crop_csv(df: pd.DataFrame, path: str):
    """
    Overwrite a TP/FP/FN CSV with the provided dataframe; if DF is empty,
    remove the file instead so later steps don’t see a stale FP list.
    """
    if df.empty:
        try: os.remove(path)
        except FileNotFoundError: pass
        return
    out = df[["x", "y", "w", "h", "frame"]].copy()
    out["frame"] = out["frame"] + ".jpg"
    out.to_csv(path, index=False)

def _delete_fp_rows_from_pred(pred_df, removed_rows):
    """
    removed_rows → iterable of tuples (x, y, w, h, frame)
    Physically drop those predictions from pred_df in-place so every
    downstream metric (PR curve, mAP, gallery, crops) sees the same set.
    """
    for x, y, w, h, frame in removed_rows:
        mask = (
            (pred_df.x == x) & (pred_df.y == y) &
            (pred_df.w == w) & (pred_df.h == h) &
            (pred_df.frame == frame)
        )
        pred_df.drop(index=pred_df[mask].index, inplace=True)

def overlap_ratio(a_xyxy, b_xyxy, ref="fp"):
    """
    Return intersection area divided by the reference box’s area.
    ref == "fp"  →  inter / area(fp)
    ref == "tp"  →  inter / area(tp)
    """
    xA, yA = max(a_xyxy[0], b_xyxy[0]), max(a_xyxy[1], b_xyxy[1])
    xB, yB = min(a_xyxy[2], b_xyxy[2]), min(a_xyxy[3], b_xyxy[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    denom = area(a_xyxy if ref == "fp" else b_xyxy)
    return inter / denom if denom else 0.0



def prune_fp_overlap(report_dir: str,
                     thresholds,
                     pred_df: pd.DataFrame,
                     metrics: dict,
                     overlap_thr: float = 0.30) -> None:
    """
    • For every IoU threshold reload matches_TP / matches_FP.
    • Remove FP boxes that overlap ANY TP (> overlap_thr) in the same frame.
    • Rewrite the FP CSV, update metrics *per-class*, and drop those rows
      from `pred_df` so all later computations stay consistent.
    """
    print("▶ Pruning FP boxes that overlap TP boxes …")
    for thr in thresholds:
        tstr    = f"{thr:.2f}"
        fp_path = os.path.join(report_dir, f"matches_FP_{tstr}.csv")
        tp_path = os.path.join(report_dir, f"matches_TP_{tstr}.csv")
        if not (os.path.exists(fp_path) and os.path.exists(tp_path)):
            continue

        fp_df = _load_crop_csv_raw(fp_path)
        tp_df = _load_crop_csv_raw(tp_path)
        if fp_df.empty or tp_df.empty:
            continue

        tp_by_frame = group_by_frame(tp_df)
        rows_to_drop, removed_specs = [], []

        for idx, fp_row in fp_df.iterrows():
            same_frame_tps = tp_by_frame.get(fp_row.frame)
            if same_frame_tps is None:
                continue
            if any(overlap_ratio(fp_row.bbox_xyxy, tp_box, ref="fp") > overlap_thr
                   for tp_box in same_frame_tps.bbox_xyxy):
                rows_to_drop.append(idx)
                removed_specs.append(
                    (fp_row.x, fp_row.y, fp_row.w, fp_row.h, fp_row.frame)
                )

                # metrics — one FP less for this class @ this IoU thr
                cls = fp_row["class"]
                if cls in metrics and thr in metrics[cls]:
                    metrics[cls][thr]["FP"] -= 1

        if rows_to_drop:
            fp_df = fp_df.drop(index=rows_to_drop)
            _write_crop_csv(fp_df, fp_path)
            _delete_fp_rows_from_pred(pred_df, removed_specs)
            print(f"  ↳ IoU {tstr}: removed {len(rows_to_drop)} FP boxes")





# ─────────────── MAIN ─────────────── #
def main():
    # 0 – setup
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(IOU_CROPS_ROOT, exist_ok=True)

    print("▶ Loading CSVs …")
    pred_df = load_csv(PRED_CSV, is_pred=True)
    gt_df   = load_csv(GT_CSV,   is_pred=False)

    # 1 – initial evaluation
    print("▶ Evaluating …")
    metrics, tp_rows, fp_rows, fn_rows = evaluate(pred_df, gt_df)

    if SAVE_MATCH_CSVS:
        pred_df, gt_df = apply_promotions_to_dfs(pred_df, gt_df,
                                                 fp_rows, fn_rows, tp_rows)

    # 2 – ***NOW*** prune FP overlap (updates metrics + pred_df)
    prune_fp_overlap(REPORT_DIR,
                     (CROP_IOU_ONLY or IOU_THRESHOLDS),
                     pred_df=pred_df,
                     metrics=metrics,
                     overlap_thr=0.30)

    # 3 – metrics table, plots, gallery  (unchanged logic)
    rows = []
    for cls, tmap in metrics.items():
        for thr, c in tmap.items():
            rows.append(dict(class_=cls, IoU=thr,
                             TP=c["TP"], FP=c["FP"], FN=c["FN"],
                             Precision=precision(c),
                             Recall=recall(c),
                             F1=f1(c)))
    pd.DataFrame(rows).to_csv(f"{REPORT_DIR}/table_metrics.csv", index=False)

    map50, map95 = coco_map(pred_df, gt_df)
    print(f"mAP@0.5 = {map50:.3f} | mAP@[.5:.95] = {map95:.3f}")

    pred_df["matched"] = False
    for frame, preds in pred_df.groupby("frame"):
        tp, _, _ = match_frame(preds, gt_df[gt_df.frame == frame], 0.5)
        for idx, _ in tp:
            pred_df.loc[idx, "matched"] = True

    re, pr_vals, auc_val = pr_curve(pred_df)
    plot_pr(re, pr_vals, auc_val)
    confs, accs = reliability(pred_df)
    plot_reliability(confs, accs)
    size_rec = stratified_recall(gt_df,
                                 set(pred_df[pred_df.matched].index))
    json.dump(size_rec,
              open(f"{REPORT_DIR}/size_recall.json", "w"), indent=2)

    build_gallery(pred_df, gt_df)

    # 4 – IoU-crop generation  (runs only if we want crops)
    if SAVE_IOU_CROPS:
        print("▶ Re-loading TP / FP / FN CSVs for crop export …")

        def _load_crop_csv(csv_path):
            if not os.path.exists(csv_path):
                return {}
            df = pd.read_csv(
                csv_path,
                names=["x", "y", "w", "h", "frame"],
                header=0 if "frame" in open(csv_path).readline().lower() else None)
            df["frame"] = df["frame"].astype(str).apply(
                lambda s: os.path.splitext(os.path.basename(s))[0])
            df["bbox_xyxy"] = df[["x", "y", "w", "h"]].apply(xywh_to_xyxy, axis=1)
            return {f: sub for f, sub in df.groupby("frame")}

        def _open_frame_png(frame_name):
            for ext in (".png", ".jpg", ".jpeg"):
                fp = os.path.join(IMAGE_ROOT, frame_name + ext)
                if os.path.exists(fp):
                    return Image.open(fp).convert("RGB")
            raise FileNotFoundError(fp)

        for thr in (CROP_IOU_ONLY or IOU_THRESHOLDS):
            tstr  = f"{thr:.2f}"
            tp_by = _load_crop_csv(os.path.join(REPORT_DIR, f"matches_TP_{tstr}.csv"))
            fp_by = _load_crop_csv(os.path.join(REPORT_DIR, f"matches_FP_{tstr}.csv"))
            fn_by = _load_crop_csv(os.path.join(REPORT_DIR, f"matches_FN_{tstr}.csv"))

            frames_needed = sorted(set(tp_by) | set(fp_by) | set(fn_by))
            if not frames_needed:
                print(f"  ↳ IoU {tstr}: no crops to write")
                continue

            base_dir = os.path.join(IOU_CROPS_ROOT, f"IoU_{tstr}")
            tp_dir, fp_dir, fn_dir = (
                os.path.join(base_dir, s) for s in ("TP", "FP", "FN")
            )
            for d in (tp_dir, fp_dir, fn_dir):
                os.makedirs(d, exist_ok=True)

            print(f"  ↳ IoU {tstr}: writing crops for {len(frames_needed)} frames …")
            for fr in tqdm(frames_needed, desc=f"IoU {tstr}", leave=False):
                try:
                    img = _open_frame_png(fr)
                except FileNotFoundError:
                    warnings.warn(f"Frame image {fr} not found — skipped")
                    continue

                def _export(df, out_dir, tag):
                    if df.empty:
                        return
                    for k, r in df.iterrows():
                        cdir = os.path.join(out_dir, f"{fr}_{tag}_{k}")
                        os.makedirs(cdir, exist_ok=True)
                        save_crop(
                            img,
                            r.bbox_xyxy,
                            os.path.join(cdir, f"{tag}_crop.png")
                        )
                        save_fullframe(
                            img,
                            r.bbox_xyxy,
                            os.path.join(cdir, f"{tag}_full.png")
                        )

                _export(tp_by.get(fr, pd.DataFrame()), tp_dir, "tp")
                _export(fp_by.get(fr, pd.DataFrame()), fp_dir, "fp")
                _export(fn_by.get(fr, pd.DataFrame()), fn_dir, "fn")

        print("✓ All artefacts written to", REPORT_DIR)
    else:
        print("✓ All artefacts written to", REPORT_DIR, "(crops skipped)")





if __name__ == "__main__":
    main()
