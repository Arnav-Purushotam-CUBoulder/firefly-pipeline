#!/usr/bin/env python3
from __future__ import annotations
import csv, json, shutil, os
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional
import cv2

_BASE_COLS = ['frame','x','y','w','h']

class AuditTrail:
    def __init__(self, root: Path, *, run_tag: str | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_tag = run_tag

    # ---- helpers ----
    def _video_id_from(self, csv_or_video_path: Path | str) -> str:
        p = Path(csv_or_video_path)
        return p.stem

    def _stage_dir(self, video_id: str, stage: str) -> Path:
        d = self.root / video_id / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def record_params(self, stage: str, **params):
        out = self.root / f"params_{stage}.json"
        try:
            with out.open('w') as f:
                json.dump(params, f, indent=2, sort_keys=True)
        except Exception:
            pass

    # ---- snapshots ----
    def copy_snapshot(self, stage: str, csv_path: Path):
        try:
            vid = self._video_id_from(csv_path)
            dst = self._stage_dir(vid, stage) / "snapshot.csv"
            shutil.copy2(csv_path, dst)
        except Exception as e:
            print(f"[audit] copy_snapshot failed: {e}")

    # ---- generic writers ----
    def _write_rows_csv(self, out: Path, rows: Iterable[Dict[str, Any]]):
        rows = list(rows)
        if not rows:
            return
        # Union of fieldnames, keep base columns at front
        fields: List[str] = []
        for b in _BASE_COLS:
            if b in rows[0].keys() and b not in fields:
                fields.append(b)
        for r in rows:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, '') for k in fields})

    def log_kept(self, stage: str, rows: Iterable[Dict[str, Any]], *, filename_suffix: str = "kept", extra_cols: Optional[List[str]] = None):
        rows = list(rows)
        if not rows:
            return
        vid = self._video_id_from(rows[0].get('csv_path', '') or rows[0].get('video','') or rows[0].get('source','') or rows[0].get('src','') or Path('.'))
        if not isinstance(vid, str) or not vid:
            vid = "unknown"
        out = self._stage_dir(vid, stage) / f"{filename_suffix}.csv"
        self._write_rows_csv(out, rows)

    def log_removed(self, stage: str, reason: str, rows: Iterable[Dict[str, Any]], *, extra_cols: Optional[List[str]] = None):
        rows = list(rows)
        if not rows:
            return
        vid = self._video_id_from(rows[0].get('csv_path', '') or rows[0].get('video','') or rows[0].get('source','') or rows[0].get('src','') or Path('.'))
        if not isinstance(vid, str) or not vid:
            vid = "unknown"
        safe_reason = reason.replace(' ', '_')
        out = self._stage_dir(vid, stage) / f"removed__{safe_reason}.csv"
        self._write_rows_csv(out, rows)

    def log_pairs(self, stage: str, pairs: Iterable[Dict[str, Any]], *, filename: str = "pairs.csv"):
        pairs = list(pairs)
        if not pairs:
            return
        vid_guess = pairs[0].get('video') or pairs[0].get('src') or pairs[0].get('source') or ''
        vid = self._video_id_from(vid_guess) if vid_guess else "unknown"
        out = self._stage_dir(vid, stage) / filename
        self._write_rows_csv(out, pairs)

    # ---- visual snippets (slow but handy) ----
    def save_crop(self, video_path: Path, frame_idx: int, x: int, y: int, w: int, h: int, subdir: str, file_stem: str):
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if frame_idx < 0 or (total and frame_idx >= total):
                cap.release(); return
            # read sequentially up to frame idx
            fr = 0
            while fr <= frame_idx:
                ok, frame = cap.read()
                if not ok:
                    break
                if fr == frame_idx:
                    break
                fr += 1
            cap.release()
            if not ok:
                return
            H, W = frame.shape[:2]
            w = max(1, int(w)); h = max(1, int(h))
            x = max(0, min(int(x), W - w))
            y = max(0, min(int(y), H - h))
            crop = frame[y:y+h, x:x+w]
            vid = Path(video_path).stem
            out_dir = self._stage_dir(vid, subdir)
            out = out_dir / f"{file_stem}.png"
            cv2.imwrite(str(out), crop)
        except Exception:
            pass
