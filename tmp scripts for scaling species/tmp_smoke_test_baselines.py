#!/usr/bin/env python3
from __future__ import annotations

"""
Temporary smoke-test for the legacy baselines (lab + Raphael) on a short clip.

Runs:
  - Lab baseline (nolan_mp4_to_predcsv.py) on first N frames
  - Raphael baseline (raphael_oorb_detect_and_gauss.py) on first N frames
  - Stage5 validator (day pipeline) to score predictions vs GT (10px)

By default this uses the small Forresti validation video and auto-derives its GT
from the latest per-species validation annotations.csv.

Safe to delete after use.
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# =========================
# CONFIG (EDIT IF NEEDED)
# =========================

VAL_INDIVIDUAL_ROOT = Path(
    "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/"
    "patch training datasets and pipeline validation data/"
    "Integrated_prototype_validation_datasets/individual species folder"
)

VAL_VIDEOS_ROOT = Path("/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/validation videos")

DEFAULT_SPECIES = "forresti"
DEFAULT_VIDEO_STEM = "4k_to_4_5k_chopped_Forresti_C0107"

RAPHAEL_MODEL_PATH = Path(
    "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data/model zoo/Raphael's model/ffnet_best.pth"
)

DIST_THRESHOLDS_PX: List[float] = [10.0]
CROP_W = 10
CROP_H = 10

# Lab baseline defaults
LAB_THRESHOLD = 0.12
LAB_BLUR_SIGMA = 1.0
LAB_BKGR_WINDOW_SEC = 2.0

# Raphael baseline defaults
RAPHAEL_BW_THR = 0.2
RAPHAEL_CLASSIFY_THR = 0.98
RAPHAEL_BKGR_WINDOW_SEC = 2.0
RAPHAEL_BLUR_SIGMA = 0.0
RAPHAEL_PATCH_SIZE_PX = 33
RAPHAEL_BATCH_SIZE = 1000
RAPHAEL_GAUSS_CROP_SIZE = 10
RAPHAEL_DEVICE = "auto"

# =========================


def _q(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(c)) for c in cmd)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _day_pipeline_dir() -> Path:
    p = _repo_root() / "day time pipeline v3 (yolo + patch classifier ensemble)"
    if not (p / "stage5_validate.py").exists():
        raise FileNotFoundError(p / "stage5_validate.py")
    return p


def _latest_version_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    best: Tuple[int, str, Path] | None = None
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("v"):
            continue
        try:
            n = int(p.name.split("_", 1)[0].replace("v", ""))
        except Exception:
            continue
        key = (n, p.name, p)
        if best is None or key[0] > best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    return best[2] if best else None


def _load_gt_rows_for_video(*, species: str, video_stem: str) -> List[Dict[str, int]]:
    sp_dir = VAL_INDIVIDUAL_ROOT / species
    latest = _latest_version_dir(sp_dir)
    if latest is None:
        raise FileNotFoundError(f"No v*/ folder under: {sp_dir}")
    ann = latest / "annotations.csv"
    if not ann.exists():
        raise FileNotFoundError(ann)

    rows: List[Dict[str, int]] = []
    with ann.open(newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise RuntimeError(f"Empty CSV headers: {ann}")
        for row in r:
            vn = str(row.get("video_name") or "").strip()
            if vn.lower().endswith(".mp4"):
                vn = vn[: -len(".mp4")]
            if vn != video_stem:
                continue
            try:
                x = int(round(float(row.get("x") or 0)))
                y = int(round(float(row.get("y") or 0)))
                t = int(round(float(row.get("t") or 0)))
            except Exception:
                continue
            rows.append({"x": x, "y": y, "t": t})

    if not rows:
        raise RuntimeError(f"No GT rows found for video={video_stem} in {ann}")
    return rows


def _write_gt_csv(path: Path, rows: Sequence[Dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "y", "t"])
        w.writeheader()
        for r in rows:
            w.writerow({"x": int(r["x"]), "y": int(r["y"]), "t": int(r["t"])})


def _parse_validation_metrics(stage_dir: Path) -> Dict[str, Any]:
    """
    Parse Stage5 outputs:
      stage_dir/thr_*.px/{fps.csv,tps.csv,fns.csv}
    """
    if not stage_dir.exists():
        return {"error": f"validation dir not found: {stage_dir}"}

    thr_dirs = sorted([p for p in stage_dir.iterdir() if p.is_dir() and p.name.startswith("thr_")])
    per: List[Dict[str, Any]] = []
    for td in thr_dirs:
        csv_fp = td / "fps.csv"
        csv_tp = td / "tps.csv"
        csv_fn = td / "fns.csv"
        if not (csv_fp.exists() and csv_tp.exists() and csv_fn.exists()):
            continue

        def _count_rows(p: Path) -> int:
            with p.open(newline="") as f:
                r = csv.reader(f)
                next(r, None)
                return sum(1 for _ in r)

        fp = _count_rows(csv_fp)
        tp = _count_rows(csv_tp)
        fn = _count_rows(csv_fn)
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per.append({"thr": td.name, "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1})

    if not per:
        return {"error": f"no thr_* folders with fp/tp/fn csvs found under: {stage_dir}"}
    best = max(per, key=lambda d: (float(d.get("f1", 0.0)), -float(d.get("fp", 0.0))))
    return {"best": best, "per_threshold": per}


def _run_logged(cmd: Sequence[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(_q(cmd) + "\n\n")
        subprocess.run(list(cmd), cwd=str(cwd), check=True, stdout=f, stderr=subprocess.STDOUT, env=env)


def _run_stage5_validator(
    *,
    orig_video_path: Path,
    pred_csv_path: Path,
    gt_csv_path: Path,
    out_dir: Path,
    max_frames: int,
    log_path: Path,
) -> None:
    day_dir = _day_pipeline_dir()
    no_weights = out_dir / "__no_fn_scoring_weights__.pt"
    code = "\n".join(
        [
            "from pathlib import Path",
            "from stage5_validate import stage5_validate_against_gt",
            "stage5_validate_against_gt(",
            f"    orig_video_path=Path({repr(str(orig_video_path))}),",
            f"    pred_csv_path=Path({repr(str(pred_csv_path))}),",
            f"    gt_csv_path=Path({repr(str(gt_csv_path))}),",
            f"    out_dir=Path({repr(str(out_dir))}),",
            f"    dist_thresholds={list(float(x) for x in DIST_THRESHOLDS_PX)!r},",
            f"    crop_w={int(CROP_W)},",
            f"    crop_h={int(CROP_H)},",
            "    gt_t_offset=0,",
            f"    max_frames={int(max_frames)},",
            "    only_firefly_rows=True,",
            "    show_per_frame=False,",
            f"    model_path=Path({repr(str(no_weights))}),",
            "    print_load_status=False,",
            ")",
        ]
    )
    _run_logged([sys.executable, "-c", code], cwd=day_dir, log_path=log_path)


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke-test lab + Raphael baselines on first N frames.")
    p.add_argument("--species", type=str, default=DEFAULT_SPECIES)
    p.add_argument("--video-stem", type=str, default=DEFAULT_VIDEO_STEM)
    p.add_argument("--max-frames", type=int, default=100)
    p.add_argument("--keep", action="store_true", default=False, help="Keep the temp folder (prints path).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    a = _parse_args(argv)

    species = str(a.species).strip()
    video_stem = str(a.video_stem).strip()
    max_frames = int(a.max_frames)
    if max_frames < 1:
        raise SystemExit("--max-frames must be >= 1")

    video_path = (VAL_VIDEOS_ROOT / species / f"{video_stem}.mp4").expanduser().resolve()
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    gt_rows = _load_gt_rows_for_video(species=species, video_stem=video_stem)

    if not RAPHAEL_MODEL_PATH.exists():
        raise SystemExit(f"Raphael model not found: {RAPHAEL_MODEL_PATH}")

    print(f"[smoke] video={video_path}")
    print(f"[smoke] gt_rows={len(gt_rows)} (will be filtered by max_frames={max_frames} in Stage5)")
    print(f"[smoke] raphael_model={RAPHAEL_MODEL_PATH}")

    repo = _repo_root()
    lab_script = repo / "tools" / "legacy_baselines" / "nolan_mp4_to_predcsv.py"
    raphael_script = repo / "tools" / "legacy_baselines" / "raphael_oorb_detect_and_gauss.py"
    if not lab_script.exists():
        raise SystemExit(f"Lab baseline script missing: {lab_script}")
    if not raphael_script.exists():
        raise SystemExit(f"Raphael baseline script missing: {raphael_script}")

    temp_root: Optional[Path] = None
    with tempfile.TemporaryDirectory(prefix="firefly_baseline_smoke_") as td:
        temp_root = Path(td).resolve()
        # Allow keep by copying out at the end if requested.
        work = temp_root

        gt_csv = work / "gt.csv"
        _write_gt_csv(gt_csv, gt_rows)

        logs = work / "logs"

        # Lab baseline
        lab_out = work / "lab"
        lab_pred = lab_out / "predictions.csv"
        print("[smoke] running lab baseline…")
        _run_logged(
            [
                sys.executable,
                str(lab_script),
                "--video",
                str(video_path),
                "--out-csv",
                str(lab_pred),
                "--threshold",
                str(float(LAB_THRESHOLD)),
                "--blur-sigma",
                str(float(LAB_BLUR_SIGMA)),
                "--bkgr-window-sec",
                str(float(LAB_BKGR_WINDOW_SEC)),
                "--max-frames",
                str(int(max_frames)),
                "--box-w",
                str(int(CROP_W)),
                "--box-h",
                str(int(CROP_H)),
            ],
            cwd=repo,
            log_path=logs / "lab_baseline.log",
        )
        _run_stage5_validator(
            orig_video_path=video_path,
            pred_csv_path=lab_pred,
            gt_csv_path=gt_csv,
            out_dir=lab_out / "stage5 validation" / video_path.stem,
            max_frames=int(max_frames),
            log_path=logs / "lab_stage5.log",
        )
        lab_metrics = _parse_validation_metrics(lab_out / "stage5 validation" / video_path.stem)
        print("[smoke] lab metrics:", json.dumps(lab_metrics.get("best") or lab_metrics, indent=2))

        # Raphael baseline
        rap_out = work / "raphael"
        rap_pred = rap_out / "predictions.csv"
        rap_raw = rap_out / "raw.csv"
        rap_gauss = rap_out / "gauss.csv"
        print("[smoke] running raphael baseline…")
        _run_logged(
            [
                sys.executable,
                str(raphael_script),
                "--video",
                str(video_path),
                "--model",
                str(RAPHAEL_MODEL_PATH),
                "--out-csv",
                str(rap_pred),
                "--raw-csv",
                str(rap_raw),
                "--gauss-csv",
                str(rap_gauss),
                "--bw-thr",
                str(float(RAPHAEL_BW_THR)),
                "--classify-thr",
                str(float(RAPHAEL_CLASSIFY_THR)),
                "--bkgr-window-sec",
                str(float(RAPHAEL_BKGR_WINDOW_SEC)),
                "--blur-sigma",
                str(float(RAPHAEL_BLUR_SIGMA)),
                "--patch-size",
                str(int(RAPHAEL_PATCH_SIZE_PX)),
                "--batch-size",
                str(int(RAPHAEL_BATCH_SIZE)),
                "--gauss-crop-size",
                str(int(RAPHAEL_GAUSS_CROP_SIZE)),
                "--max-frames",
                str(int(max_frames)),
                "--device",
                str(RAPHAEL_DEVICE),
            ],
            cwd=repo,
            log_path=logs / "raphael_baseline.log",
        )
        _run_stage5_validator(
            orig_video_path=video_path,
            pred_csv_path=rap_pred,
            gt_csv_path=gt_csv,
            out_dir=rap_out / "stage5 validation" / video_path.stem,
            max_frames=int(max_frames),
            log_path=logs / "raphael_stage5.log",
        )
        rap_metrics = _parse_validation_metrics(rap_out / "stage5 validation" / video_path.stem)
        print("[smoke] raphael metrics:", json.dumps(rap_metrics.get("best") or rap_metrics, indent=2))

        if a.keep:
            keep_dir = Path.cwd() / f"_kept_baseline_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
            keep_dir = keep_dir.resolve()
            print(f"[smoke] keeping outputs → {keep_dir}")
            # Copy the entire temp dir contents out before TemporaryDirectory cleanup.
            subprocess.run(["bash", "-lc", f"cp -a {shlex.quote(str(work))}/. {shlex.quote(str(keep_dir))}/"], check=True)

    print("[smoke] done (temp folder cleaned up).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

