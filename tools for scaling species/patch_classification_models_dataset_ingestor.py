#!/usr/bin/env python3
from __future__ import annotations

"""
Patch-classification dataset ingestor for the integrated ingest → train → test pipeline.

Goal
----
Scan a RAW videos root (with one folder per species, prefixed by day_/night_), and ingest ONLY the
videos that have not yet been ingested for each species (supports new species
and new videos added later to an existing species). No training. No testing.

Also writes an append-only change log (JSONL) that records folder/file changes
done by each ingestion job, with patch-image folders treated as "bulk" to keep
log size realistic.
"""

import argparse
import base64
import csv
import difflib
import errno
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = e
else:
    _CV2_IMPORT_ERROR = None

import numpy as np

# -----------------------------------------------------------------------------
# Inlined change-log helpers
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotConfig:
    root: Path
    scopes: Sequence[Path] | None = None
    ignore_files: Sequence[Path] = ()
    max_hash_bytes: int = 5 * 1024 * 1024
    max_inline_bytes: int = 512 * 1024
    max_diff_lines: int = 2000


@dataclass
class FileRec:
    path: str
    size: int
    mtime_ns: int
    sha256: str | None
    inline_kind: str | None
    inline_data: str | None


@dataclass
class DirRec:
    path: str
    bulk: bool
    mtime_ns: int
    n_files: int | None = None
    total_bytes: int | None = None


@dataclass
class Snapshot:
    created_at: str
    root: str
    dirs: Dict[str, DirRec]
    files: Dict[str, FileRec]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _rel_posix(root: Path, p: Path) -> str:
    try:
        rel = p.relative_to(root)
    except Exception:
        return str(p)
    s = rel.as_posix()
    return "" if s == "." else s


def _is_text_path(p: Path) -> bool:
    return p.suffix.lower() in {".json", ".jsonl", ".txt", ".md", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".py", ".ini"}


def _sha256_file(path: Path, *, max_bytes: int) -> str | None:
    try:
        size = path.stat().st_size
    except Exception:
        return None
    if int(size) > int(max_bytes):
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def _read_inline(path: Path, *, max_bytes: int) -> tuple[str | None, str | None]:
    try:
        size = path.stat().st_size
    except Exception:
        return None, None
    if int(size) > int(max_bytes):
        return None, None

    try:
        raw = path.read_bytes()
    except Exception:
        return None, None

    if _is_text_path(path):
        try:
            return "text", raw.decode("utf-8", errors="replace")
        except Exception:
            return "b64", base64.b64encode(raw).decode("ascii")

    if b"\x00" not in raw:
        try:
            return "text", raw.decode("utf-8", errors="strict")
        except Exception:
            pass
    return "b64", base64.b64encode(raw).decode("ascii")


def _is_bulk_dir(dir_path: Path) -> bool:
    if dir_path.parent.name == "inference outputs":
        return True
    name = dir_path.name
    if name not in {"firefly", "background"}:
        return False
    parent = dir_path.parent.name
    if parent in {"initial dataset", "train_patches"}:
        return True
    if parent in {"train", "val", "test"} and dir_path.parent.parent.name == "final dataset":
        return True
    return False


def _iter_scopes(cfg: SnapshotConfig) -> List[Path]:
    root = Path(cfg.root).expanduser().resolve()
    if cfg.scopes is None:
        return [root]
    scopes: List[Path] = []
    for s in cfg.scopes:
        try:
            p = Path(s).expanduser().resolve()
        except Exception:
            continue
        try:
            p.relative_to(root)
        except Exception:
            continue
        scopes.append(p)
    return scopes


def take_snapshot(cfg: SnapshotConfig) -> Snapshot:
    root = Path(cfg.root).expanduser().resolve()
    ignore = {str(Path(p).expanduser().resolve()) for p in (cfg.ignore_files or [])}

    dirs: Dict[str, DirRec] = {}
    files: Dict[str, FileRec] = {}

    stack: List[Path] = list(reversed(_iter_scopes(cfg)))
    seen_dirs: set[str] = set()

    while stack:
        d = stack.pop()
        try:
            d_res = d.resolve()
        except Exception:
            d_res = d
        try:
            if not d_res.exists() or (not d_res.is_dir()):
                continue
        except Exception:
            continue
        d_key = str(d_res)
        if d_key in seen_dirs:
            continue
        seen_dirs.add(d_key)

        rel = _rel_posix(root, d_res)
        try:
            d_mtime_ns = int(d_res.stat().st_mtime_ns)
        except Exception:
            d_mtime_ns = 0

        bulk = _is_bulk_dir(d_res)
        dirs[rel] = DirRec(path=rel, bulk=bulk, mtime_ns=d_mtime_ns, n_files=None, total_bytes=None)
        if bulk:
            continue

        try:
            with os.scandir(d_res) as it:
                for e in it:
                    try:
                        p = Path(e.path)
                        if e.is_dir(follow_symlinks=False):
                            stack.append(p)
                        elif e.is_file(follow_symlinks=False):
                            try:
                                p_res = p.resolve()
                            except Exception:
                                p_res = p
                            if str(p_res) in ignore:
                                continue
                            relp = _rel_posix(root, p_res)
                            try:
                                st = p_res.stat()
                            except Exception:
                                continue
                            files[relp] = FileRec(
                                path=relp,
                                size=int(st.st_size),
                                mtime_ns=int(st.st_mtime_ns),
                                sha256=_sha256_file(p_res, max_bytes=int(cfg.max_hash_bytes)),
                                inline_kind=None,
                                inline_data=None,
                            )
                            files[relp].inline_kind, files[relp].inline_data = _read_inline(
                                p_res,
                                max_bytes=int(cfg.max_inline_bytes),
                            )
                    except Exception:
                        continue
        except Exception:
            continue

    return Snapshot(created_at=_now_iso(), root=str(root), dirs=dirs, files=files)


def _file_changed(a: FileRec, b: FileRec) -> bool:
    if a.size != b.size:
        return True
    if a.sha256 is not None and b.sha256 is not None:
        return a.sha256 != b.sha256
    return a.mtime_ns != b.mtime_ns


def diff_snapshots(*, before: Snapshot, after: Snapshot, cfg: SnapshotConfig) -> Dict[str, Any]:
    before_dirs = set(before.dirs.keys())
    after_dirs = set(after.dirs.keys())
    before_files = set(before.files.keys())
    after_files = set(after.files.keys())

    dirs_added = sorted(after_dirs - before_dirs)
    dirs_removed = sorted(before_dirs - after_dirs)
    files_added = sorted(after_files - before_files)
    files_removed = sorted(before_files - after_files)

    files_modified: List[str] = []
    for p in sorted(before_files & after_files):
        if _file_changed(before.files[p], after.files[p]):
            files_modified.append(p)

    bulk_stats_changed: List[Dict[str, Any]] = []
    for p in sorted(before_dirs & after_dirs):
        a = before.dirs[p]
        b = after.dirs[p]
        if a.bulk and b.bulk and int(a.mtime_ns or 0) != int(b.mtime_ns or 0):
            bulk_stats_changed.append(
                {
                    "path": p,
                    "before": {"mtime_ns": a.mtime_ns, "n_files": a.n_files, "total_bytes": a.total_bytes},
                    "after": {"mtime_ns": b.mtime_ns, "n_files": b.n_files, "total_bytes": b.total_bytes},
                }
            )

    def _file_payload(rec: FileRec, *, include_inline: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "path": rec.path,
            "size": rec.size,
            "mtime_ns": rec.mtime_ns,
            "sha256": rec.sha256,
        }
        if include_inline:
            payload["inline_kind"] = rec.inline_kind
            payload["inline_data"] = rec.inline_data
        return payload

    modified_payload: List[Dict[str, Any]] = []
    for p in files_modified:
        a = before.files[p]
        b = after.files[p]
        diff_txt = None
        if a.inline_kind == "text" and b.inline_kind == "text" and a.inline_data is not None and b.inline_data is not None:
            ud = list(
                difflib.unified_diff(
                    a.inline_data.splitlines(keepends=False),
                    b.inline_data.splitlines(keepends=False),
                    fromfile=f"a/{p}",
                    tofile=f"b/{p}",
                    lineterm="",
                )
            )
            if len(ud) > int(cfg.max_diff_lines):
                ud = ud[: int(cfg.max_diff_lines)] + [f"... (diff truncated; max_lines={int(cfg.max_diff_lines)})"]
            diff_txt = "\n".join(ud)
        modified_payload.append(
            {
                "path": p,
                "before": _file_payload(a, include_inline=True),
                "after": _file_payload(b, include_inline=False),
                "diff": diff_txt,
            }
        )

    return {
        "dirs_added": dirs_added,
        "dirs_added_detail": [
            {
                "path": p,
                "bulk": bool(after.dirs[p].bulk),
                "mtime_ns": int(after.dirs[p].mtime_ns or 0),
                "n_files": after.dirs[p].n_files,
                "total_bytes": after.dirs[p].total_bytes,
            }
            for p in dirs_added
            if p in after.dirs
        ],
        "dirs_removed": dirs_removed,
        "dirs_removed_detail": [
            {
                "path": p,
                "bulk": bool(before.dirs[p].bulk),
                "mtime_ns": int(before.dirs[p].mtime_ns or 0),
                "n_files": before.dirs[p].n_files,
                "total_bytes": before.dirs[p].total_bytes,
            }
            for p in dirs_removed
            if p in before.dirs
        ],
        "files_added": [_file_payload(after.files[p], include_inline=False) for p in files_added],
        "files_removed": [_file_payload(before.files[p], include_inline=True) for p in files_removed],
        "files_modified": modified_payload,
        "bulk_dirs_changed": bulk_stats_changed,
    }


def append_change_log(*, log_path: Path, record: Dict[str, Any]) -> None:
    log_path = Path(log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class ChangeLogRun:
    cfg: SnapshotConfig
    log_path: Path
    meta: Dict[str, Any]
    enabled: bool = True

    _before: Snapshot | None = None
    _closed: bool = False

    def __enter__(self) -> "ChangeLogRun":
        self._closed = False
        if not self.enabled:
            return self
        self._before = take_snapshot(self.cfg)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if not self.enabled or self._closed:
            return
        self._closed = True
        if self._before is None:
            return

        try:
            after = take_snapshot(self.cfg)
            changes = diff_snapshots(before=self._before, after=after, cfg=self.cfg)
            had_error = exc is not None
            if not had_error:
                has_any_change = any(
                    bool(changes.get(k))
                    for k in (
                        "dirs_added",
                        "dirs_removed",
                        "files_added",
                        "files_removed",
                        "files_modified",
                        "bulk_dirs_changed",
                    )
                )
                if not has_any_change:
                    return

            append_change_log(
                log_path=self.log_path,
                record={
                    "schema_version": 1,
                    "timestamp": _now_iso(),
                    "meta": dict(self.meta),
                    "monitored_root": str(Path(self.cfg.root).expanduser().resolve()),
                    "scopes": [str(p) for p in _iter_scopes(self.cfg)],
                    "had_error": bool(had_error),
                    "error": (str(exc) if had_error else None),
                    "snapshot": {
                        "before_created_at": self._before.created_at,
                        "after_created_at": after.created_at,
                    },
                    "changes": changes,
                },
            )
        except Exception:
            return


def default_log_path(log_root: Path) -> Path:
    return Path(log_root) / "codex_change_log.jsonl"


def iter_change_log_records(log_path: Path) -> Iterator[Dict[str, Any]]:
    log_path = Path(log_path).expanduser().resolve()
    if not log_path.exists():
        return
    try:
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if isinstance(rec, dict):
                    yield rec
    except Exception:
        return


def build_ingestion_index(log_path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rec in iter_change_log_records(log_path):
        try:
            had_error = bool(rec.get("had_error"))
            meta = rec.get("meta")
            if not isinstance(meta, dict):
                continue
            if bool(meta.get("dry_run")):
                continue
            actor = str(meta.get("actor") or "").strip()
            if actor not in {"ingestor_only", "patch_classification_models_dataset_ingestor", "orchestrator"}:
                continue

            ts = str(rec.get("timestamp") or "").strip()
            default_species = (
                str(meta.get("species_token") or "").strip()
                or str(meta.get("species_name") or "").strip()
                or str(meta.get("species") or "").strip()
            )
            pairs = meta.get("ingested_pairs")
            if not isinstance(pairs, list):
                continue

            for it in pairs:
                if isinstance(it, str):
                    species_token = default_species
                    video_name = str(it).strip()
                    payload: Dict[str, Any] = {}
                elif isinstance(it, dict):
                    species_token = (
                        str(it.get("species_token") or "").strip()
                        or str(it.get("species_name") or "").strip()
                        or str(it.get("species") or "").strip()
                        or default_species
                    )
                    video_name = str(it.get("video_name") or it.get("video") or "").strip()
                    payload = dict(it)
                else:
                    continue

                if not species_token or not video_name:
                    continue

                info: Dict[str, Any] = {"timestamp": ts, "actor": actor, "had_error": bool(had_error)}
                split = payload.get("split")
                if split is not None:
                    info["split"] = split
                for k in ("video_path", "firefly_csv", "background_csv"):
                    v = payload.get(k)
                    if v is not None:
                        info[k] = v

                out.setdefault(species_token, {})[video_name] = info
        except Exception:
            continue
    return out

# -----------------------------------------------------------------------------
# User-configurable globals (edit these)
# -----------------------------------------------------------------------------

# RAW input root (contains one folder per species).
# Expected folder naming convention (case-insensitive): day_<species> / night_<species>
# Each folder contains .mp4 + annotator .csv files.
RAW_VIDEOS_ROOT: str | Path = "/mnt/Samsung_SSD_2TB/all species raw videos and annotations (from annotator)"

# Integrated pipeline ROOT (this is the same "ROOT_PATH" you use in the integrated orchestrator)
ROOT_PATH: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype data/integrated prototype data"

# Where to write the append-only change log + rollback instructions.
# The user requested this outer folder:
LOG_ROOT_PATH: str | Path = "/mnt/Samsung_SSD_2TB/integrated prototype data"
CHANGE_LOG_FILENAME: str = "codex_change_log.jsonl"

# Subfolders under ROOT_PATH
DATA_SUBDIR: str = "patch training datasets and pipeline validation data"

# Ingestion settings
TRAIN_PAIR_FRACTION: float = 0.8
ONE_DATASET_VERSION_PER_BATCH: bool = True
# Keep batch_exports as scratch only. Successful ingests delete their per-video
# export folders immediately, and the staged CSV folder is removed after the
# species finishes ingesting.
CLEAN_BATCH_EXPORTS_AFTER_SUCCESS: bool = True

# Explicit train/inference routing is defined centrally at RAW_VIDEOS_ROOT.
# TRAIN_PAIR_FRACTION is only used as a fallback when the root catalog is
# disabled or missing.
VIDEO_CATALOG_FILENAME: str = "tmp_scaling_species_training_inference_catalog.json"
USE_ROOT_VIDEO_CATALOG_IF_PRESENT: bool = True
INGEST_TRAINING_VIDEOS_FROM_ROOT_CATALOG: bool = True

# Stage1_ingestor_core config (overrides passed into stage1_ingestor_core)
AUTO_LOAD_SIBLING_CLASS_CSV: bool = True
DATASET_VERSION_COPY_MODE: str = "hardlink"  # "hardlink" | "copy"

# Auto background patch generation
AUTO_GENERATE_BACKGROUND_PATCHES: bool = True
AUTO_BACKGROUND_TO_FIREFLY_RATIO: float = 10.0
AUTO_BACKGROUND_PATCH_SIZE_PX: int = 10
AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME: int = 10
AUTO_BACKGROUND_MAX_FRAME_SAMPLES: int = 5000
AUTO_BACKGROUND_SEED: int = 1337
AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS: bool = True

# Safety/performance: blob detection can be very slow on high-res videos. If it is too slow,
# stage1_ingestor_core will disable blob detection and fall back to random centers only.
AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS: float = 2.0
AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES: int = 1

AUTO_BACKGROUND_SBD_MIN_AREA_PX: float = 0.5
AUTO_BACKGROUND_SBD_MAX_AREA_SCALE: float = 1.0
AUTO_BACKGROUND_SBD_MIN_DIST: float = 0.25
AUTO_BACKGROUND_SBD_MIN_REPEAT: int = 1

AUTO_BACKGROUND_USE_CLAHE: bool = True
AUTO_BACKGROUND_CLAHE_CLIP: float = 2.0
AUTO_BACKGROUND_CLAHE_TILE: Tuple[int, int] = (8, 8)

AUTO_BACKGROUND_USE_TOPHAT: bool = False
AUTO_BACKGROUND_TOPHAT_KSIZE: int = 7

AUTO_BACKGROUND_USE_DOG: bool = False
AUTO_BACKGROUND_DOG_SIGMA1: float = 0.8
AUTO_BACKGROUND_DOG_SIGMA2: float = 1.6

# If True, only ingest species that do not already have a single-species dataset version dir.
# If False (recommended), also ingest *new videos* added later to an existing species folder.
ONLY_INGEST_NOT_YET_INGESTED: bool = False

# Optional: restrict to certain species folder names (as they appear under RAW_VIDEOS_ROOT)
ONLY_RAW_SPECIES_DIRS: Sequence[str] = ()

# Optional: override raw folder name -> species token used by the pipeline.
# Overrides may be keyed by either:
#   - the full raw folder name (e.g. "night_Photinus Knulli"), or
#   - the base species name without the day/night prefix (e.g. "Photinus Knulli")
# IMPORTANT: species token must NOT contain underscores because stage1_ingestor_core parses
# identity from CSV filenames split on underscores.
SPECIES_NAME_OVERRIDES: Dict[str, str] = {
    # Example:
    # "Photinus Knulli": "photinus-knulli",
    # "Photinus Knulli": "s1806ik",
}

# Change log
ENABLE_CODEX_CHANGE_LOG: bool = True

# If True, do everything except the actual stage1 ingestion writes.
DRY_RUN: bool = False

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Inlined stage1 ingestion/background-patch code
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservedPair:
    video_name: str
    video_path: Path
    firefly_csv: Path
    background_csv: Path | None


@dataclass(frozen=True)
class StagedPair:
    pair: ObservedPair
    staged_firefly_csv: Path
    staged_background_csv: Path | None


def safe_name(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _is_annotator_csv(csv_path: Path) -> bool:
    try:
        with csv_path.open(newline="") as fh:
            r = csv.DictReader(fh)
            fieldnames = [str(x or "").strip().lower() for x in (r.fieldnames or [])]
    except Exception:
        return False
    if not fieldnames:
        return False
    cols = set(fieldnames)
    if not {"x", "y", "w", "h"}.issubset(cols):
        return False
    return bool({"t", "frame", "time"} & cols)


def _parse_label_from_csv_name(csv_path: Path, *, video_key: str) -> str:
    """
    Parse label from an annotator CSV filename.

    Expected: <video_stem>_..._[firefly|background].csv
    - label defaults to "firefly" if omitted
    """
    vk = (video_key or "").strip().lower()

    remainder = ""
    raw_key = csv_path.stem.strip().lower()
    if vk and (raw_key == vk or raw_key.startswith(vk + "_")):
        remainder = raw_key[len(vk) :].lstrip("_")
    else:
        stem = csv_path.stem.strip()
        stem = stem.replace("-", "_").replace("(", "_").replace(")", "_")
        stem = re.sub(r"_+", "_", stem).strip("_")
        key = stem.lower()

        vk_norm = vk.replace("-", "_").replace("(", "_").replace(")", "_")
        vk_norm = re.sub(r"_+", "_", vk_norm).strip("_")

        if vk_norm and (key == vk_norm or key.startswith(vk_norm + "_")):
            remainder = key[len(vk_norm) :].lstrip("_")
        else:
            # Fall back to parsing by suffix only.
            remainder = key

    parts = [p for p in remainder.split("_") if p]
    if parts and parts[-1].lower() in {"firefly", "background"}:
        return parts[-1].lower()
    return "firefly"


def discover_observed_pairs(observed_dir: Path) -> List[ObservedPair]:
    observed_dir = Path(observed_dir)
    if not observed_dir.exists():
        raise FileNotFoundError(observed_dir)
    if not observed_dir.is_dir():
        raise NotADirectoryError(observed_dir)

    videos = sorted([p for p in observed_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4"])
    if not videos:
        raise SystemExit(f"No .mp4 videos found under: {observed_dir}")

    videos_by_stem: Dict[str, Path] = {}
    for v in videos:
        key = v.stem.strip().lower()
        if not key:
            continue
        if key in videos_by_stem and str(videos_by_stem[key].resolve()) != str(v.resolve()):
            raise SystemExit(f"Duplicate video stem {v.stem!r} found in: {videos_by_stem[key]} and {v}")
        videos_by_stem[key] = v

    stem_keys = sorted(videos_by_stem.keys(), key=len, reverse=True)

    csv_candidates = sorted([p for p in observed_dir.rglob("*.csv") if p.is_file() and _is_annotator_csv(p)])
    if not csv_candidates:
        raise SystemExit(f"No annotator CSVs found under: {observed_dir} (need x,y,w,h and t/frame columns).")

    grouped: Dict[str, Dict[str, Any]] = {}
    unmatched: List[Path] = []
    for c in csv_candidates:
        c_key = c.stem.strip().lower()
        match: str | None = None
        for vk in stem_keys:
            if c_key == vk or c_key.startswith(vk + "_"):
                match = vk
                break
        if match is None:
            unmatched.append(c)
            continue

        vp = videos_by_stem[match]
        video_name = safe_name(vp.stem)
        label = _parse_label_from_csv_name(c, video_key=match)

        g = grouped.setdefault(
            match,
            {
                "video_path": vp,
                "video_name": video_name,
                "firefly_csv": None,
                "background_csv": None,
            },
        )

        if label == "firefly":
            if g["firefly_csv"] is not None and str(Path(g["firefly_csv"]).resolve()) != str(c.resolve()):
                raise SystemExit(f"Multiple firefly CSVs matched to video {vp.name}: {g['firefly_csv']} and {c}")
            g["firefly_csv"] = c
        elif label == "background":
            if g["background_csv"] is not None and str(Path(g["background_csv"]).resolve()) != str(c.resolve()):
                raise SystemExit(f"Multiple background CSVs matched to video {vp.name}: {g['background_csv']} and {c}")
            g["background_csv"] = c

    if unmatched:
        preview = "\n".join([f"  - {p}" for p in unmatched[:20]])
        more = "" if len(unmatched) <= 20 else f"\n  ... and {len(unmatched) - 20} more"
        raise SystemExit(
            "Found annotator CSVs that did not match any .mp4 by filename prefix:\n"
            f"{preview}{more}\n"
            "Ensure each CSV name starts with the corresponding video stem."
        )

    pairs: List[ObservedPair] = []
    for _, g in sorted(grouped.items(), key=lambda kv: str(kv[1].get("video_name") or "")):
        if g.get("firefly_csv") is None:
            vp = g["video_path"]
            raise SystemExit(f"Missing firefly CSV for video: {vp} (need ..._firefly.csv or unlabeled CSV).")
        pairs.append(
            ObservedPair(
                video_name=str(g["video_name"]),
                video_path=Path(g["video_path"]),
                firefly_csv=Path(g["firefly_csv"]),
                background_csv=Path(g["background_csv"]) if g.get("background_csv") else None,
            )
        )

    if not pairs:
        raise SystemExit(f"No valid (video, firefly_csv) pairs found under: {observed_dir}")

    return pairs


def split_pairs_train_vs_val(pairs: Sequence[ObservedPair], *, train_fraction: float) -> Tuple[List[ObservedPair], List[ObservedPair]]:
    ordered = sorted(pairs, key=lambda p: (p.video_name, str(p.video_path)))
    if not (0.0 < float(train_fraction) < 1.0):
        raise ValueError(f"train_fraction must be between 0 and 1 (got {train_fraction!r})")

    n = len(ordered)
    n_train = int(n * float(train_fraction))
    # Ensure both splits are non-empty when possible.
    if n >= 2:
        n_train = max(1, min(n - 1, n_train))
    return ordered[:n_train], ordered[n_train:]


def stage_pairs_for_ingestor(
    pairs: Sequence[ObservedPair],
    *,
    species_name: str,
    staging_dir: Path,
    time_of_day: str,
    dry_run: bool,
) -> List[StagedPair]:
    """
    stage1_ingestor_core parses identity from CSV filename, so we stage/copy CSVs into a
    canonical naming scheme:
      <video_name>_<species_name>_<day_time|night_time>_<firefly|background>.csv
    """
    staging_dir = Path(staging_dir)
    if not dry_run:
        staging_dir.mkdir(parents=True, exist_ok=True)

    staged: List[StagedPair] = []
    seen: set[str] = set()
    safe_species = safe_name(species_name)
    if not safe_species:
        raise SystemExit(f"Invalid species_name for staging: {species_name!r}")

    for p in pairs:
        base = f"{safe_name(p.video_name)}_{safe_species}_{safe_name(time_of_day)}"
        dst_firefly = staging_dir / f"{base}_firefly.csv"
        if str(dst_firefly) in seen:
            raise SystemExit(f"Staging name collision (firefly): {dst_firefly}")
        seen.add(str(dst_firefly))

        if dry_run:
            print(f"[dry-run] Would stage CSV -> {dst_firefly}  (src={p.firefly_csv})")
        else:
            dst_firefly.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p.firefly_csv, dst_firefly)

        dst_bg: Path | None = None
        if p.background_csv is not None:
            dst_bg = staging_dir / f"{base}_background.csv"
            if str(dst_bg) in seen:
                raise SystemExit(f"Staging name collision (background): {dst_bg}")
            seen.add(str(dst_bg))
            if dry_run:
                print(f"[dry-run] Would stage CSV -> {dst_bg}  (src={p.background_csv})")
            else:
                dst_bg.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p.background_csv, dst_bg)

        staged.append(StagedPair(pair=p, staged_firefly_csv=dst_firefly, staged_background_csv=dst_bg))

    return staged

@dataclass(frozen=True)
class BlobDetectorConfig:
    # Mirror defaults used by tools/v3 daytime pipeline negative patch generator.py
    min_area_px: float = 0.5
    max_area_scale: float = 1.0  # fraction of frame area allowed (<=0 => no upper bound)
    min_dist_px: float = 0.25
    min_repeat: int = 1
    # Performance knobs:
    # SimpleBlobDetector thresholds from min_threshold..max_threshold in steps of threshold_step.
    # Smaller steps are slower on high-res frames.
    min_threshold: int = 0
    max_threshold: int = 255
    threshold_step: int = 10


@dataclass(frozen=True)
class PreprocessConfig:
    # Mirror defaults used by tools/v3 daytime pipeline negative patch generator.py
    use_clahe: bool = True
    clahe_clip: float = 2.0
    clahe_tile: Tuple[int, int] = (8, 8)

    use_tophat: bool = False
    tophat_ksize: int = 7

    use_dog: bool = False
    dog_sigma1: float = 0.8
    dog_sigma2: float = 1.6
    # Downscale frames before blob detection (speeds up large videos a lot).
    # None disables downscaling.
    downscale_max_dim: int | None = 960


# Best-effort caches (avoid re-creating OpenCV objects per frame)
_DETECTOR_CACHE: dict[tuple, object] = {}
_CLAHE_CACHE: dict[tuple, object] = {}


def _make_blob_detector(*, cfg: BlobDetectorConfig, frame_w: int, frame_h: int):
    area = float(max(1, int(frame_w) * int(frame_h)))
    max_area = float(area * cfg.max_area_scale) if float(cfg.max_area_scale) > 0.0 else float(area)

    p = cv2.SimpleBlobDetector_Params()

    p.filterByColor = False

    p.filterByArea = True
    p.minArea = float(cfg.min_area_px)
    p.maxArea = float(max_area)

    p.filterByCircularity = False
    p.filterByConvexity = False
    p.filterByInertia = False

    p.minThreshold = int(cfg.min_threshold)
    p.maxThreshold = int(cfg.max_threshold)
    p.thresholdStep = max(1, int(cfg.threshold_step))
    p.minRepeatability = int(cfg.min_repeat)
    p.minDistBetweenBlobs = float(cfg.min_dist_px)
    return cv2.SimpleBlobDetector_create(p)


def _preprocess(gray_u8: np.ndarray, *, cfg: PreprocessConfig) -> np.ndarray:
    inp = gray_u8
    if cfg.use_clahe:
        key = (float(cfg.clahe_clip), int(cfg.clahe_tile[0]), int(cfg.clahe_tile[1]))
        clahe = _CLAHE_CACHE.get(key)
        if clahe is None:
            clahe = cv2.createCLAHE(clipLimit=float(cfg.clahe_clip), tileGridSize=tuple(cfg.clahe_tile))
            _CLAHE_CACHE[key] = clahe
        inp = clahe.apply(inp)
    if cfg.use_tophat:
        ksize = int(cfg.tophat_ksize)
        if ksize < 3 or ksize % 2 == 0:
            ksize = 7
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        inp = cv2.morphologyEx(inp, cv2.MORPH_TOPHAT, k)
    if cfg.use_dog:
        g1 = cv2.GaussianBlur(inp, (0, 0), float(cfg.dog_sigma1))
        g2 = cv2.GaussianBlur(inp, (0, 0), float(cfg.dog_sigma2))
        inp = cv2.subtract(g1, g2)
    return inp


def find_blob_centers(
    frame_bgr: np.ndarray,
    *,
    blob_cfg: BlobDetectorConfig = BlobDetectorConfig(),
    pre_cfg: PreprocessConfig = PreprocessConfig(),
) -> List[Tuple[int, int]]:
    if frame_bgr is None:
        return []
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return []

    scale = 1.0
    max_dim = pre_cfg.downscale_max_dim
    if max_dim is not None:
        try:
            max_dim_i = int(max_dim)
        except Exception:
            max_dim_i = 0
        if max_dim_i > 0 and max(h, w) > max_dim_i:
            scale = float(max_dim_i) / float(max(h, w))
            nh = max(1, int(round(float(h) * scale)))
            nw = max(1, int(round(float(w) * scale)))
            frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = frame_bgr.shape[:2]

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    proc = _preprocess(gray, cfg=pre_cfg)

    det_key = (
        float(blob_cfg.min_area_px),
        float(blob_cfg.max_area_scale),
        float(blob_cfg.min_dist_px),
        int(blob_cfg.min_repeat),
        int(blob_cfg.min_threshold),
        int(blob_cfg.max_threshold),
        int(blob_cfg.threshold_step),
        int(w),
        int(h),
    )
    detector = _DETECTOR_CACHE.get(det_key)
    if detector is None:
        detector = _make_blob_detector(cfg=blob_cfg, frame_w=w, frame_h=h)
        _DETECTOR_CACHE[det_key] = detector

    keypoints = detector.detect(proc)
    centers: List[Tuple[int, int]] = []
    for kp in keypoints:
        cx, cy = kp.pt
        if scale != 1.0:
            cx, cy = (float(cx) / scale), (float(cy) / scale)
        centers.append((int(round(cx)), int(round(cy))))
    return centers

# Inline stage1-ingestor-core runtime globals.
ANNOTATIONS_CSV_PATH: Path = Path('.')
VIDEO_PATH: Path = Path('.')
TRAIN_FRACTION_OF_BATCH: float = 0.80
TRAIN_VAL_SPLIT_SEED: int = 1337
DATA_ROOT: Path = Path('.')
BATCH_EXPORT_ROOT: Path = Path('.')
TRAIN_DATASETS_ROOT: Path = Path('.')
SINGLE_SPECIES_TARGET_VERSION_DIR: Path | None = None
INTEGRATED_TARGET_VERSION_DIR: Path | None = None
SKIP_FINAL_SPLIT_REBUILD: bool = False
FINAL_TRAIN_PCT: float = 0.80
FINAL_VAL_PCT: float = 0.15
FINAL_TEST_PCT: float = 0.05
FINAL_SPLIT_SEED: int = 1337
VERSION_COPY_MODE: str = 'copy'
SKIP_EXISTING_PATCHES: bool = True
PATCH_IMAGE_EXT: str = '.png'
PATCH_LOCATIONS_CSV_NAME: str = 'patch_locations.csv'
PATCH_LOCATIONS_SPLIT_PREFIX: str = 'patch_locations_'
PATCH_LOCATIONS_BACKGROUND_CSV_NAME: str = 'patch_locations_background.csv'
PATCH_LOCATIONS_BACKGROUND_SPLIT_PREFIX: str = 'patch_locations_background_'
ANNOTATION_XY_SEMANTICS: str = 'center'

INTEGRATED_PIPELINE_DATASETS_DIRNAME = "integrated pipeline datasets"
SINGLE_SPECIES_DATASETS_DIRNAME = "single species datasets"

DAY_DATASET_DIRNAME = "day_time_dataset"
NIGHT_DATASET_DIRNAME = "night_time_dataset"

INITIAL_DATASET_DIRNAME = "initial dataset"
FINAL_DATASET_DIRNAME = "final dataset"

CLASS_FIREFLY = "firefly"
CLASS_BACKGROUND = "background"
SPLIT_NAMES = ("train", "val", "test")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Annotation:
    x: int
    y: int
    w: int
    h: int
    t: int


def _today_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


def _safe_name(s: str) -> str:
    s = s.strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _iter_image_files(d: Path) -> Iterator[Path]:
    if not d.exists():
        return
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def _count_images(d: Path) -> int:
    return sum(1 for _ in _iter_image_files(d))


def _parse_batch_identity(csv_path: Path) -> Tuple[str, str, str, str]:
    """
    Return (video_name, species_name, time_of_day, class_label) from CSV filename.

    Preferred naming (recommended):
      <video_name>_<species_name>_<day_time|night_time>_<firefly|background>.csv

    Back-compat (treated as firefly):
      <video_name>_<species_name>_<day_time|night_time>.csv
    """
    stem = csv_path.stem.strip()
    # tolerate mild variations
    # Keep hyphens so multi-word species can remain a single token, e.g.
    # "photinus-knulli" stays intact during parsing.
    stem = stem.replace("(", "_").replace(")", "_")
    stem = re.sub(r"_+", "_", stem).strip("_")

    parts = [p for p in stem.split("_") if p]
    if len(parts) < 4:
        raise ValueError(
            f"Could not parse identity from CSV name {csv_path.name!r}. "
            "Expected <video_name>_<species_name>_<day_time|night_time>_<firefly|background>."
        )

    label = parts[-1].lower()
    if label in {CLASS_FIREFLY, CLASS_BACKGROUND}:
        parts = parts[:-1]
    else:
        # old convention: no explicit label => assume firefly
        label = CLASS_FIREFLY

    if len(parts) < 3 or parts[-1].lower() != "time" or parts[-2].lower() not in {"day", "night"}:
        raise ValueError(
            f"Could not parse time_of_day from CSV name {csv_path.name!r}. "
            "Expected ..._day_time_... or ..._night_time_...."
        )
    time_of_day = f"{parts[-2].lower()}_time"

    base = parts[:-2]  # everything before day_time|night_time tokens
    if len(base) < 2:
        raise ValueError(
            f"Could not parse video/species from CSV name {csv_path.name!r}. "
            "Expected <video_name>_<species_name>_..."
        )
    species_name = base[-1]
    video_name = "_".join(base[:-1])
    return _safe_name(video_name), _safe_name(species_name), time_of_day, label


def _read_annotator_csv(csv_path: Path) -> List[Annotation]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    def _parse_frame_value(value: object) -> int:
        text = str(value or "").strip()
        if not text:
            raise ValueError("empty frame value")
        try:
            return int(round(float(text)))
        except Exception:
            pass

        m = re.search(r"(\d+)", text)
        if not m:
            raise ValueError(f"could not parse frame value: {text!r}")
        return int(m.group(1))

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return []

        cols = {c.strip(): c for c in reader.fieldnames if c}
        t_col = cols.get("t") or cols.get("frame") or cols.get("time")
        required = ["x", "y", "w", "h"]
        missing = [c for c in required if c not in cols]
        if missing or not t_col:
            raise ValueError(
                f"{csv_path} missing required columns. "
                f"Need x,y,w,h and t (or frame). Got: {reader.fieldnames}"
            )

        out: List[Annotation] = []
        seen: set[Tuple[int, int, int, int, int]] = set()
        for r in reader:
            try:
                x = int(round(float(r[cols["x"]])))
                y = int(round(float(r[cols["y"]])))
                w = int(round(float(r[cols["w"]])))
                h = int(round(float(r[cols["h"]])))
                t = _parse_frame_value(r[t_col])
            except Exception:
                continue
            if w <= 0 or h <= 0 or t < 0:
                continue
            key = (x, y, w, h, t)
            if key in seen:
                continue
            seen.add(key)
            out.append(Annotation(x=x, y=y, w=w, h=h, t=t))
        return out


def _split_annotations(
    anns: Sequence[Annotation], train_frac: float, seed: int
) -> Tuple[List[Annotation], List[Annotation]]:
    if not (0.0 <= train_frac <= 1.0):
        raise ValueError("TRAIN_FRACTION_OF_BATCH must be within [0, 1].")

    anns_shuf = list(anns)
    rng = random.Random(int(seed))
    rng.shuffle(anns_shuf)
    n = len(anns_shuf)
    n_train = int(n * float(train_frac))
    n_train = max(0, min(n, n_train))
    train = anns_shuf[:n_train]
    val = anns_shuf[n_train:]
    return train, val


def _write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _read_csv_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        return [], []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = [{k: (r.get(k) or "") for k in fieldnames} for r in reader]
        return rows, fieldnames


def _merge_csv_rows(
    existing: List[Dict[str, str]],
    new_rows: List[Dict[str, str]],
    key_fields: Sequence[str],
) -> List[Dict[str, str]]:
    out = list(existing)
    seen = {
        tuple(r.get(k, "") for k in key_fields)
        for r in existing
    }
    for r in new_rows:
        key = tuple(r.get(k, "") for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _crop_with_pad(frame_bgr, x: int, y: int, w: int, h: int):
    import numpy as np  # local import to avoid hard dependency if caller doesn't use extraction

    H, W = frame_bgr.shape[:2]
    xy_mode = str(ANNOTATION_XY_SEMANTICS or "center").strip().lower()
    if xy_mode == "center":
        # Convert center point to top-left for fixed-size crop extraction.
        x0 = int(round(float(x) - float(w) / 2.0))
        y0 = int(round(float(y) - float(h) / 2.0))
    elif xy_mode == "top_left":
        x0, y0 = int(x), int(y)
    else:
        raise ValueError(f"Unsupported ANNOTATION_XY_SEMANTICS={ANNOTATION_XY_SEMANTICS!r}")
    x1, y1 = x0 + int(w), y0 + int(h)
    vx0, vy0 = max(0, x0), max(0, y0)
    vx1, vy1 = min(W, x1), min(H, y1)
    patch = np.zeros((max(1, int(h)), max(1, int(w)), 3), dtype=frame_bgr.dtype)
    if vx1 > vx0 and vy1 > vy0:
        px0, py0 = vx0 - x0, vy0 - y0
        patch[py0:py0 + (vy1 - vy0), px0:px0 + (vx1 - vx0)] = frame_bgr[vy0:vy1, vx0:vx1]
    return patch


def _make_patch_filename(a: Annotation, video_name: str, species_name: str) -> str:
    video_name = _safe_name(video_name)
    species_name = _safe_name(species_name)
    return f"{a.x}_{a.y}_{a.w}_{a.h}_{a.t}_{video_name}_{species_name}{PATCH_IMAGE_EXT}"


def _parse_patch_filename(p: Path) -> Optional[Tuple[int, int, int, int, int, str, str]]:
    """
    Parse <x>_<y>_<w>_<h>_<t>_<video_name>_<species_name>.<ext>
    into (x,y,w,h,t,video_name,species_name).
    """
    parts = p.stem.split("_")
    if len(parts) < 7:
        return None
    try:
        x, y, w, h, t = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
    except Exception:
        return None
    species_name = parts[-1]
    video_name = "_".join(parts[5:-1])
    if not video_name:
        return None
    return x, y, w, h, t, video_name, species_name


def _non_firefly_frame_ranges(total_frames: int, firefly_frames: Sequence[int]) -> List[Tuple[int, int]]:
    total_frames = int(total_frames)
    if total_frames <= 0:
        return []
    forbidden = sorted({int(t) for t in firefly_frames if 0 <= int(t) < total_frames})
    out: List[Tuple[int, int]] = []
    prev = -1
    for t in forbidden:
        if t > prev + 1:
            out.append((prev + 1, t - 1))
        prev = t
    if prev < total_frames - 1:
        out.append((prev + 1, total_frames - 1))
    return out


def _sample_frame_from_ranges(ranges: Sequence[Tuple[int, int]], rng: random.Random) -> int:
    if not ranges:
        raise ValueError("ranges must be non-empty")
    lengths = [max(0, int(e) - int(s) + 1) for s, e in ranges]
    total = sum(lengths)
    if total <= 0:
        raise ValueError("ranges total length must be positive")
    pick = rng.randrange(total)
    for (s, e), L in zip(ranges, lengths):
        if pick < L:
            return int(s) + int(pick)
        pick -= L
    # Fallback (should be unreachable)
    s, e = ranges[-1]
    return int(e) if int(e) >= int(s) else int(s)


def _auto_generate_background_train_annotations(
    *,
    video_path: Path,
    firefly_anns_all: Sequence["Annotation"],
    n_needed: int,
) -> List["Annotation"]:
    """
    Generate synthetic BACKGROUND training annotations (center-x, center-y, w, h, t)
    by sampling frames that do not contain any annotated firefly frames.
    """
    if n_needed <= 0:
        return []
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for auto background patch generation. "
            f"Import error: {_CV2_IMPORT_ERROR}"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for background sampling: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        # Best-effort fallback if backend can't report frame count.
        max_t = max((int(a.t) for a in firefly_anns_all), default=-1)
        total_frames = max_t + 1

    if total_frames <= 0:
        cap.release()
        return []

    firefly_frames = [int(a.t) for a in firefly_anns_all]
    ranges = _non_firefly_frame_ranges(total_frames, firefly_frames)
    if not ranges:
        print("[bg] WARNING: no non-firefly frames available; skipping auto background generation.")
        cap.release()
        return []

    # Stable per-video RNG seed (avoid Python hash randomization).
    seed = int(AUTO_BACKGROUND_SEED) ^ (zlib.crc32(str(video_path).encode("utf-8")) & 0xFFFFFFFF)
    rng = random.Random(seed)

    blob_cfg = BlobDetectorConfig(
        min_area_px=float(AUTO_BACKGROUND_SBD_MIN_AREA_PX),
        max_area_scale=float(AUTO_BACKGROUND_SBD_MAX_AREA_SCALE),
        min_dist_px=float(AUTO_BACKGROUND_SBD_MIN_DIST),
        min_repeat=int(AUTO_BACKGROUND_SBD_MIN_REPEAT),
    )
    pre_cfg = PreprocessConfig(
        use_clahe=bool(AUTO_BACKGROUND_USE_CLAHE),
        clahe_clip=float(AUTO_BACKGROUND_CLAHE_CLIP),
        clahe_tile=tuple(AUTO_BACKGROUND_CLAHE_TILE),
        use_tophat=bool(AUTO_BACKGROUND_USE_TOPHAT),
        tophat_ksize=int(AUTO_BACKGROUND_TOPHAT_KSIZE),
        use_dog=bool(AUTO_BACKGROUND_USE_DOG),
        dog_sigma1=float(AUTO_BACKGROUND_DOG_SIGMA1),
        dog_sigma2=float(AUTO_BACKGROUND_DOG_SIGMA2),
    )

    patch_size = max(1, int(AUTO_BACKGROUND_PATCH_SIZE_PX))
    max_per_frame = max(1, int(AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME))
    max_frame_samples = max(1, int(AUTO_BACKGROUND_MAX_FRAME_SAMPLES))

    out: List[Annotation] = []
    seen: Set[Tuple[int, int, int]] = set()  # (t,x,y)
    sampled = 0
    blob_detection_enabled = True
    slow_frames = 0
    slow_s = float(AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS)
    disable_after = max(1, int(AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES))

    pbar = (
        tqdm(total=int(n_needed), desc=f"[bg] {video_path.stem}", unit="patch", dynamic_ncols=True)
        if tqdm is not None and int(n_needed) > 0
        else None
    )
    last_len = 0

    while len(out) < n_needed and sampled < max_frame_samples:
        t = _sample_frame_from_ranges(ranges, rng)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame_bgr = cap.read()
        sampled += 1
        if not ok or frame_bgr is None:
            continue

        centers = []
        if blob_detection_enabled:
            t0 = time.time()
            centers = find_blob_centers(frame_bgr, blob_cfg=blob_cfg, pre_cfg=pre_cfg)
            dt = float(time.time() - t0)
            if slow_s > 0.0 and dt >= slow_s:
                slow_frames += 1
                if slow_frames >= disable_after and bool(AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS):
                    blob_detection_enabled = False
                    print(
                        "[bg] INFO: blob detection is slow; disabling blob detector and using random centers only:",
                        f"dt={dt:.2f}s threshold={slow_s:.2f}s slow_frames={slow_frames}/{disable_after}",
                    )
        if centers:
            if len(centers) > max_per_frame:
                centers = rng.sample(centers, k=max_per_frame)
            else:
                rng.shuffle(centers)
        else:
            if not AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS:
                continue
            h, w = frame_bgr.shape[:2]
            half = patch_size // 2
            if w > patch_size:
                x_lo, x_hi = half, max(half, w - half - 1)
            else:
                x_lo, x_hi = 0, max(0, w - 1)
            if h > patch_size:
                y_lo, y_hi = half, max(half, h - half - 1)
            else:
                y_lo, y_hi = 0, max(0, h - 1)
            centers = [(rng.randint(x_lo, x_hi), rng.randint(y_lo, y_hi)) for _ in range(max_per_frame)]

        for cx, cy in centers:
            key = (int(t), int(cx), int(cy))
            if key in seen:
                continue
            seen.add(key)
            out.append(Annotation(x=int(cx), y=int(cy), w=patch_size, h=patch_size, t=int(t)))
            if len(out) >= n_needed:
                break

        if pbar is not None:
            cur_len = len(out)
            if cur_len > last_len:
                pbar.update(cur_len - last_len)
                last_len = cur_len
            pbar.set_postfix(sampled_frames=int(sampled), ranges=int(len(ranges)))
        elif sampled % 50 == 0:
            print(f"[bg] sampled_frames={sampled} generated={len(out)}/{n_needed}")

    cap.release()
    if pbar is not None:
        pbar.close()
    if len(out) < n_needed:
        print(
            f"[bg] WARNING: generated only {len(out)}/{n_needed} background patches "
            f"(sampled_frames={sampled}, cap={max_frame_samples})."
        )
    return out


def _extract_patches_from_video(
    video_path: Path,
    anns: Sequence[Annotation],
    out_dir: Path,
    video_name: str,
    species_name: str,
) -> int:
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for patch extraction. "
            f"Import error: {_CV2_IMPORT_ERROR}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    rows_by_t: Dict[int, List[Annotation]] = {}
    for a in anns:
        rows_by_t.setdefault(int(a.t), []).append(a)

    saved = 0
    ts = sorted(rows_by_t.keys())
    pbar = (
        tqdm(ts, desc=f"[patches] {video_path.stem}", unit="frame", dynamic_ncols=True)
        if tqdm is not None and ts
        else None
    )
    it = pbar if pbar is not None else ts
    for i, t in enumerate(it):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        for a in rows_by_t[t]:
            base_name = _make_patch_filename(a, video_name=video_name, species_name=species_name)
            out_path = out_dir / base_name
            if out_path.exists():
                if SKIP_EXISTING_PATCHES:
                    continue
                stem = out_path.stem
                suffix = out_path.suffix
                k = 1
                while True:
                    cand = out_dir / f"{stem}__dup{k}{suffix}"
                    if not cand.exists():
                        out_path = cand
                        break
                    k += 1

            patch = _crop_with_pad(frame_bgr, a.x, a.y, a.w, a.h)
            okw = cv2.imwrite(str(out_path), patch)
            if okw:
                saved += 1

        if pbar is not None:
            pbar.set_postfix(saved=int(saved), anns=int(len(rows_by_t.get(int(t), []))))
        elif (i + 1) % 25 == 0:
            print(f"[patches] processed {i+1}/{len(rows_by_t)} unique frames…")

    cap.release()
    if pbar is not None:
        pbar.close()
    return saved


def _ensure_train_root_scaffold(train_root: Path) -> Tuple[Path, Path]:
    integrated = train_root / INTEGRATED_PIPELINE_DATASETS_DIRNAME
    single = train_root / SINGLE_SPECIES_DATASETS_DIRNAME
    integrated.mkdir(parents=True, exist_ok=True)
    single.mkdir(parents=True, exist_ok=True)
    return integrated, single


def _ensure_time_dataset_scaffold(time_root: Path) -> None:
    (time_root / INITIAL_DATASET_DIRNAME / CLASS_FIREFLY).mkdir(parents=True, exist_ok=True)
    (time_root / INITIAL_DATASET_DIRNAME / CLASS_BACKGROUND).mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        (time_root / FINAL_DATASET_DIRNAME / split / CLASS_FIREFLY).mkdir(parents=True, exist_ok=True)
        (time_root / FINAL_DATASET_DIRNAME / split / CLASS_BACKGROUND).mkdir(parents=True, exist_ok=True)


def _version_num_from_name(name: str) -> Optional[int]:
    m = re.match(r"^v(?P<n>\d+)(?:_|$)", name)
    if not m:
        return None
    try:
        return int(m.group("n"))
    except Exception:
        return None


def _latest_version_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    best: Tuple[int, str, Path] | None = None
    for p in root.iterdir():
        if not p.is_dir():
            continue
        n = _version_num_from_name(p.name)
        if n is None:
            continue
        key = (n, p.name, p)
        if best is None or key[0] > best[0] or (key[0] == best[0] and key[1] > best[1]):
            best = key
    return best[2] if best else None


def _next_version_dir(root: Path) -> Tuple[Path, Optional[Path]]:
    prev = _latest_version_dir(root)
    prev_n = _version_num_from_name(prev.name) if prev else 0
    new_n = int(prev_n or 0) + 1
    new_dir = root / f"v{new_n}_{_today_tag()}"
    return new_dir, prev


def _copytree(src: Path, dst: Path, mode: str) -> None:
    if not src.is_dir():
        raise ValueError(f"copytree src is not a directory: {src}")
    if dst.exists():
        raise FileExistsError(dst)
    dst.mkdir(parents=True, exist_ok=False)

    mode = (mode or "copy").strip().lower()

    total_files = 0
    for _, _, files in os.walk(src):
        total_files += len(files)

    pbar = (
        tqdm(total=total_files, desc=f"[version] {src.name} -> {dst.name}", unit="file", dynamic_ncols=True)
        if tqdm is not None and total_files > 0
        else None
    )
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for d in dirs:
            (dst / rel / d).mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            t = dst / rel / f
            if mode == "hardlink":
                try:
                    os.link(s, t)
                except OSError as e:
                    if e.errno in {errno.EXDEV, errno.EPERM, errno.EACCES, errno.EEXIST}:
                        shutil.copy2(s, t)
                    else:
                        shutil.copy2(s, t)
            else:
                shutil.copy2(s, t)
            if pbar is not None:
                pbar.update(1)
    if pbar is not None:
        pbar.close()


def _safe_rmtree(p: Path) -> None:
    if not p.exists():
        return
    if not p.is_dir():
        raise ValueError(f"Refusing to delete non-directory: {p}")
    shutil.rmtree(p)


def _safe_cleanup_batch_exports_path(path: Path | None, *, batch_exports_root: Path) -> None:
    if path is None:
        return

    batch_exports_root = Path(batch_exports_root).expanduser().resolve()
    target = Path(path).expanduser().resolve()

    if not target.exists():
        return
    try:
        target.relative_to(batch_exports_root)
    except Exception as e:
        raise ValueError(f"Refusing to clean path outside batch_exports: {target}") from e

    if target == batch_exports_root:
        return

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()

    current = target.parent
    while True:
        if current == batch_exports_root:
            try:
                next(current.iterdir())
            except StopIteration:
                current.rmdir()
            except Exception:
                pass
            break
        try:
            current.relative_to(batch_exports_root)
        except Exception:
            break
        try:
            next(current.iterdir())
            break
        except StopIteration:
            parent = current.parent
            current.rmdir()
            current = parent
        except Exception:
            break


def _split_dataset_dir(
    src_initial: Path,
    dst_final: Path,
    *,
    train_pct: float,
    val_pct: float,
    test_pct: float,
    seed: int,
    copy_mode: str,
) -> Dict[str, Dict[str, int]]:
    if abs((train_pct + val_pct + test_pct) - 1.0) > 1e-6:
        raise ValueError("FINAL_TRAIN_PCT + FINAL_VAL_PCT + FINAL_TEST_PCT must sum to 1.0")

    _safe_rmtree(dst_final)
    dst_final.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(seed))

    metrics: Dict[str, Dict[str, int]] = {}
    for cls in (CLASS_FIREFLY, CLASS_BACKGROUND):
        src_cls = src_initial / cls
        files = sorted(list(_iter_image_files(src_cls)))
        rng.shuffle(files)

        n_total = len(files)
        n_train = int(n_total * train_pct)
        n_val = int(n_total * val_pct)
        split_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        metrics[cls] = {"total": n_total}
        pbar = (
            tqdm(total=n_total, desc=f"[final split] {cls}", unit="img", dynamic_ncols=True)
            if tqdm is not None and n_total > 0
            else None
        )
        for split, flist in split_files.items():
            dst_dir = dst_final / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            metrics[cls][split] = len(flist)
            for src_path in flist:
                dst_path = dst_dir / src_path.name
                if copy_mode == "hardlink":
                    try:
                        os.link(src_path, dst_path)
                    except OSError:
                        shutil.copy2(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(split=str(split))
        if pbar is not None:
            pbar.close()
    return metrics


def _write_split_patch_locations_csv(
    firefly_dir: Path,
    out_csv: Path,
    *,
    include_species: bool,
) -> int:
    rows: List[Dict[str, object]] = []
    for img in sorted(_iter_image_files(firefly_dir)):
        meta = _parse_patch_filename(img)
        if not meta:
            continue
        x, y, w, h, t, video_name, species_name = meta
        row: Dict[str, object] = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "t": t,
            "video_name": video_name,
        }
        if include_species:
            row["species_name"] = species_name
        rows.append(row)

    fields = ["x", "y", "w", "h", "t", "video_name"]
    if include_species:
        fields.append("species_name")
    _write_csv_rows(out_csv, fields, rows)
    return len(rows)


def _collect_train_stats(train_root: Path) -> Dict[str, Dict[str, int]]:
    integrated_root = train_root / INTEGRATED_PIPELINE_DATASETS_DIRNAME
    single_root = train_root / SINGLE_SPECIES_DATASETS_DIRNAME

    stats: Dict[str, Dict[str, int]] = {
        "single_species_firefly": {},
        "single_species_background": {},
        "integrated_firefly": {},
        "integrated_background": {},
    }

    # single-species: count images in latest version initial/{firefly,background} per species
    if single_root.exists():
        for species_dir in sorted([p for p in single_root.iterdir() if p.is_dir()]):
            latest = _latest_version_dir(species_dir)
            if not latest:
                continue
            firefly_dir = latest / INITIAL_DATASET_DIRNAME / CLASS_FIREFLY
            background_dir = latest / INITIAL_DATASET_DIRNAME / CLASS_BACKGROUND
            stats["single_species_firefly"][species_dir.name] = _count_images(firefly_dir)
            stats["single_species_background"][species_dir.name] = _count_images(background_dir)

    # integrated: count images by parsing filenames in latest version initial/{firefly,background} across day+night
    latest_int = _latest_version_dir(integrated_root)
    if latest_int:
        counts_ff: Dict[str, int] = {}
        counts_bg: Dict[str, int] = {}
        for tod_dir in (DAY_DATASET_DIRNAME, NIGHT_DATASET_DIRNAME):
            for cls, counts in ((CLASS_FIREFLY, counts_ff), (CLASS_BACKGROUND, counts_bg)):
                cls_dir = latest_int / tod_dir / INITIAL_DATASET_DIRNAME / cls
                for img in _iter_image_files(cls_dir):
                    meta = _parse_patch_filename(img)
                    if not meta:
                        continue
                    _, _, _, _, _, _, sp = meta
                    counts[sp] = counts.get(sp, 0) + 1
        stats["integrated_firefly"] = dict(sorted(counts_ff.items(), key=lambda kv: (-kv[1], kv[0])))
        stats["integrated_background"] = dict(sorted(counts_bg.items(), key=lambda kv: (-kv[1], kv[0])))

    return stats


def _print_stats(title: str, stats: Dict[str, Dict[str, int]]) -> None:
    print(f"\n=== {title} ===")
    for section, d in stats.items():
        items = ", ".join([f"{k}={v}" for k, v in d.items()]) if d else "(empty)"
        print(f"{section}: {items}")

def _stage1_ingestor_core_main() -> Path | None:
    csv_path = Path(ANNOTATIONS_CSV_PATH).expanduser()
    video_path = Path(VIDEO_PATH).expanduser()

    video_name, species_name, time_of_day, primary_label = _parse_batch_identity(csv_path)
    dataset_time_dirname = DAY_DATASET_DIRNAME if time_of_day == "day_time" else NIGHT_DATASET_DIRNAME

    csv_paths: Dict[str, Path] = {primary_label: csv_path}
    if AUTO_LOAD_SIBLING_CLASS_CSV:
        other = CLASS_BACKGROUND if primary_label == CLASS_FIREFLY else CLASS_FIREFLY
        cand = csv_path.with_name(f"{video_name}_{species_name}_{time_of_day}_{other}{csv_path.suffix}")
        if cand.exists():
            try:
                v2, s2, t2, l2 = _parse_batch_identity(cand)
            except Exception as e:
                print(f"[warn] Skipping sibling CSV {cand.name!r}: {e}")
            else:
                if (v2, s2, t2, l2) == (video_name, species_name, time_of_day, other):
                    csv_paths[other] = cand
                else:
                    print(f"[warn] Skipping sibling CSV {cand.name!r}: identity mismatch")

    integrated_root, single_root = _ensure_train_root_scaffold(TRAIN_DATASETS_ROOT)

    # ── stats before ──
    train_before = _collect_train_stats(TRAIN_DATASETS_ROOT)
    _print_stats("TRAIN stats (before)", train_before)

    # ── read + split incoming annotations (firefly/background) ──
    anns_by_label: Dict[str, List[Annotation]] = {}
    for lbl, p in csv_paths.items():
        anns = _read_annotator_csv(p)
        anns_by_label[lbl] = anns

    if not any(anns_by_label.values()):
        print(f"No valid rows found in: {', '.join(str(p) for p in csv_paths.values())}; nothing to do.")
        return None

    train_anns_by_label: Dict[str, List[Annotation]] = {}
    final_validation_firefly: List[Annotation] = []

    for lbl, anns in anns_by_label.items():
        if lbl == CLASS_FIREFLY:
            tr, fv = _split_annotations(anns, TRAIN_FRACTION_OF_BATCH, TRAIN_VAL_SPLIT_SEED)
            train_anns_by_label[lbl] = tr
            final_validation_firefly = fv
        else:
            # Background batches are training-only (no pipeline final validation)
            train_anns_by_label[lbl] = list(anns)

    # Auto background generation (TRAIN only)
    if AUTO_GENERATE_BACKGROUND_PATCHES:
        firefly_train = train_anns_by_label.get(CLASS_FIREFLY, [])
        if firefly_train:
            existing_bg = train_anns_by_label.get(CLASS_BACKGROUND, [])
            target_bg = int(round(float(AUTO_BACKGROUND_TO_FIREFLY_RATIO) * float(len(firefly_train))))
            need_bg = max(0, int(target_bg) - int(len(existing_bg)))
            if need_bg > 0:
                print(
                    "[bg] Auto-generating background patches:",
                    f"train_firefly={len(firefly_train)} target_bg={target_bg} existing_bg={len(existing_bg)} need={need_bg}",
                )
                try:
                    gen = _auto_generate_background_train_annotations(
                        video_path=video_path,
                        firefly_anns_all=anns_by_label.get(CLASS_FIREFLY, []),
                        n_needed=need_bg,
                    )
                except Exception as e:
                    print(f"[bg] WARNING: background auto-generation failed: {e}")
                    gen = []
                if gen:
                    train_anns_by_label.setdefault(CLASS_BACKGROUND, []).extend(gen)
                    print(f"[bg] Generated background train rows: {len(gen)}")

    # ── export this run’s split CSVs + train patches ──
    run_tag = _today_tag() + "__" + datetime.now().strftime("%H%M%S")
    batch_tag = _safe_name(f"{video_name}_{species_name}_{time_of_day}")
    export_dir = BATCH_EXPORT_ROOT / f"{batch_tag}__{run_tag}"
    export_patches_root = export_dir / "train_patches"

    export_dir.mkdir(parents=True, exist_ok=True)

    if CLASS_FIREFLY in anns_by_label:
        _write_csv_rows(
            export_dir / "train_annotations_firefly.csv",
            ["x", "y", "w", "h", "t"],
            [{"x": a.x, "y": a.y, "w": a.w, "h": a.h, "t": a.t} for a in train_anns_by_label.get(CLASS_FIREFLY, [])],
        )
        _write_csv_rows(
            export_dir / "final_validation_annotations_firefly.csv",
            ["x", "y", "w", "h", "t"],
            [{"x": a.x, "y": a.y, "w": a.w, "h": a.h, "t": a.t} for a in final_validation_firefly],
        )

    # Write background TRAIN rows if present (including auto-generated backgrounds).
    if train_anns_by_label.get(CLASS_BACKGROUND, []):
        _write_csv_rows(
            export_dir / "train_annotations_background.csv",
            ["x", "y", "w", "h", "t"],
            [{"x": a.x, "y": a.y, "w": a.w, "h": a.h, "t": a.t} for a in train_anns_by_label.get(CLASS_BACKGROUND, [])],
        )

    n_in_firefly = len(anns_by_label.get(CLASS_FIREFLY, []))
    n_in_bg = len(anns_by_label.get(CLASS_BACKGROUND, []))
    print(
        f"\nIncoming rows: firefly={n_in_firefly}, background={n_in_bg} | "
        f"train_firefly={len(train_anns_by_label.get(CLASS_FIREFLY, []))}, "
        f"train_background={len(train_anns_by_label.get(CLASS_BACKGROUND, []))}, "
        f"final_validation_firefly={len(final_validation_firefly)}"
    )

    # Patch extraction
    saved_by_label: Dict[str, int] = {}
    for lbl, tr_anns in train_anns_by_label.items():
        if not tr_anns:
            continue
        out_dir = export_patches_root / lbl
        print(f"Extracting {len(tr_anns)} {lbl.upper()} TRAIN patches → {out_dir}")
        saved_by_label[lbl] = _extract_patches_from_video(
            video_path=video_path,
            anns=tr_anns,
            out_dir=out_dir,
            video_name=video_name,
            species_name=species_name,
        )
        print(f"Saved {saved_by_label[lbl]} {lbl} patches.")

    has_train_data = any(train_anns_by_label.get(lbl) for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND))

    # ────────────────────────────────────────────────────────────────────
    # TRAIN DATASETS (patches + patch_locations.csv + final split)
    # ────────────────────────────────────────────────────────────────────

    if has_train_data:
        # ---- single-species dataset update ----
        species_root = single_root / species_name
        species_root.mkdir(parents=True, exist_ok=True)

        target_species_ver = Path(SINGLE_SPECIES_TARGET_VERSION_DIR).expanduser() if SINGLE_SPECIES_TARGET_VERSION_DIR else None
        if target_species_ver is not None:
            try:
                target_species_ver.relative_to(species_root)
            except Exception as e:
                raise ValueError(
                    f"SINGLE_SPECIES_TARGET_VERSION_DIR must be under {species_root} (got {target_species_ver})"
                ) from e

            new_species_ver = target_species_ver
            prev_species_ver = _latest_version_dir(species_root)
            if not new_species_ver.exists():
                if prev_species_ver:
                    print(f"\n[single] Initializing {new_species_ver} (copying from {prev_species_ver.name})")
                    _copytree(prev_species_ver, new_species_ver, VERSION_COPY_MODE)
                else:
                    print(f"\n[single] Initializing {new_species_ver} (new species)")
                    new_species_ver.mkdir(parents=True, exist_ok=False)
            else:
                print(f"\n[single] Using existing target version dir: {new_species_ver}")
        else:
            new_species_ver, prev_species_ver = _next_version_dir(species_root)
            if prev_species_ver:
                print(f"\n[single] Creating {new_species_ver} (copying from {prev_species_ver.name})")
                _copytree(prev_species_ver, new_species_ver, VERSION_COPY_MODE)
            else:
                print(f"\n[single] Creating {new_species_ver} (new species)")
                new_species_ver.mkdir(parents=True, exist_ok=False)

        _ensure_time_dataset_scaffold(new_species_ver)

        # Add new patches into initial/{firefly,background}
        for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
            src_dir = export_patches_root / lbl
            if not src_dir.exists():
                continue
            dst_dir = new_species_ver / INITIAL_DATASET_DIRNAME / lbl
            dst_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            src_files = sorted(_iter_image_files(src_dir))
            pbar = (
                tqdm(src_files, desc=f"[single] copy {lbl}", unit="img", dynamic_ncols=True)
                if tqdm is not None and src_files
                else None
            )
            it = pbar if pbar is not None else src_files
            for p in it:
                out = dst_dir / p.name
                if out.exists() and SKIP_EXISTING_PATCHES:
                    continue
                if out.exists() and not SKIP_EXISTING_PATCHES:
                    stem = out.stem
                    suffix = out.suffix
                    k = 1
                    while (dst_dir / f"{stem}__dup{k}{suffix}").exists():
                        k += 1
                    out = dst_dir / f"{stem}__dup{k}{suffix}"
                shutil.copy2(p, out)
                copied += 1
                if pbar is not None:
                    pbar.set_postfix(copied=int(copied))
            if pbar is not None:
                pbar.close()
            print(f"[single] Added {copied} {lbl} patches → {dst_dir}")

        # Update patch_locations CSVs for this version (train rows only)
        for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
            tr_anns = train_anns_by_label.get(lbl, [])
            if not tr_anns:
                continue
            out_name = PATCH_LOCATIONS_CSV_NAME if lbl == CLASS_FIREFLY else PATCH_LOCATIONS_BACKGROUND_CSV_NAME
            out_csv = new_species_ver / out_name
            existing_rows, _ = _read_csv_rows(out_csv)
            new_rows = [
                {
                    "x": str(a.x),
                    "y": str(a.y),
                    "w": str(a.w),
                    "h": str(a.h),
                    "t": str(a.t),
                    "video_name": video_name,
                }
                for a in tr_anns
            ]
            merged = _merge_csv_rows(existing_rows, new_rows, key_fields=["x", "y", "w", "h", "t", "video_name"])
            _write_csv_rows(out_csv, ["x", "y", "w", "h", "t", "video_name"], merged)
            print(f"[single] {out_csv.name} rows: {len(merged)}")

        if SKIP_FINAL_SPLIT_REBUILD:
            print("[single] SKIP_FINAL_SPLIT_REBUILD=True; leaving final split unchanged for now.")
        else:
            # Rebuild final dataset split
            src_initial = new_species_ver / INITIAL_DATASET_DIRNAME
            dst_final = new_species_ver / FINAL_DATASET_DIRNAME
            metrics = _split_dataset_dir(
                src_initial,
                dst_final,
                train_pct=FINAL_TRAIN_PCT,
                val_pct=FINAL_VAL_PCT,
                test_pct=FINAL_TEST_PCT,
                seed=FINAL_SPLIT_SEED,
                copy_mode=VERSION_COPY_MODE,
            )
            print(f"[single] Final split metrics: {metrics}")

            # Write split CSVs (firefly + background)
            for split in SPLIT_NAMES:
                for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
                    cls_dir = dst_final / split / lbl
                    prefix = (
                        PATCH_LOCATIONS_SPLIT_PREFIX
                        if lbl == CLASS_FIREFLY
                        else PATCH_LOCATIONS_BACKGROUND_SPLIT_PREFIX
                    )
                    out_csv = new_species_ver / f"{prefix}{split}.csv"
                    nrows = _write_split_patch_locations_csv(cls_dir, out_csv, include_species=False)
                    print(f"[single] Wrote {out_csv.name} rows={nrows}")

        # ---- integrated dataset update ----
        target_int_ver = Path(INTEGRATED_TARGET_VERSION_DIR).expanduser() if INTEGRATED_TARGET_VERSION_DIR else None
        if target_int_ver is not None:
            try:
                target_int_ver.relative_to(integrated_root)
            except Exception as e:
                raise ValueError(
                    f"INTEGRATED_TARGET_VERSION_DIR must be under {integrated_root} (got {target_int_ver})"
                ) from e

            new_int_ver = target_int_ver
            prev_int_ver = _latest_version_dir(integrated_root)
            if not new_int_ver.exists():
                if prev_int_ver:
                    print(f"\n[integrated] Initializing {new_int_ver} (copying from {prev_int_ver.name})")
                    _copytree(prev_int_ver, new_int_ver, VERSION_COPY_MODE)
                else:
                    print(f"\n[integrated] Initializing {new_int_ver} (new integrated dataset)")
                    new_int_ver.mkdir(parents=True, exist_ok=False)
            else:
                print(f"\n[integrated] Using existing target version dir: {new_int_ver}")
        else:
            new_int_ver, prev_int_ver = _next_version_dir(integrated_root)
            if prev_int_ver:
                print(f"\n[integrated] Creating {new_int_ver} (copying from {prev_int_ver.name})")
                _copytree(prev_int_ver, new_int_ver, VERSION_COPY_MODE)
            else:
                print(f"\n[integrated] Creating {new_int_ver} (new integrated dataset)")
                new_int_ver.mkdir(parents=True, exist_ok=False)

        # Ensure both day/night scaffolds exist (even if only one is updated this run)
        day_root = new_int_ver / DAY_DATASET_DIRNAME
        night_root = new_int_ver / NIGHT_DATASET_DIRNAME
        _ensure_time_dataset_scaffold(day_root)
        _ensure_time_dataset_scaffold(night_root)

        target_time_root = new_int_ver / dataset_time_dirname
        # Add new patches into <time>/initial/{firefly,background}
        for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
            src_dir = export_patches_root / lbl
            if not src_dir.exists():
                continue
            dst_dir = target_time_root / INITIAL_DATASET_DIRNAME / lbl
            dst_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            src_files = sorted(_iter_image_files(src_dir))
            pbar = (
                tqdm(src_files, desc=f"[integrated] copy {lbl}", unit="img", dynamic_ncols=True)
                if tqdm is not None and src_files
                else None
            )
            it = pbar if pbar is not None else src_files
            for p in it:
                out = dst_dir / p.name
                if out.exists() and SKIP_EXISTING_PATCHES:
                    continue
                if out.exists() and not SKIP_EXISTING_PATCHES:
                    stem = out.stem
                    suffix = out.suffix
                    k = 1
                    while (dst_dir / f"{stem}__dup{k}{suffix}").exists():
                        k += 1
                    out = dst_dir / f"{stem}__dup{k}{suffix}"
                shutil.copy2(p, out)
                copied += 1
                if pbar is not None:
                    pbar.set_postfix(copied=int(copied))
            if pbar is not None:
                pbar.close()
            print(f"[integrated] Added {copied} {lbl} patches → {dst_dir}")

        # Update integrated patch_locations CSVs (at version root)
        for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
            tr_anns = train_anns_by_label.get(lbl, [])
            if not tr_anns:
                continue
            out_name = PATCH_LOCATIONS_CSV_NAME if lbl == CLASS_FIREFLY else PATCH_LOCATIONS_BACKGROUND_CSV_NAME
            out_csv = new_int_ver / out_name
            existing_rows, _ = _read_csv_rows(out_csv)
            new_rows = [
                {
                    "x": str(a.x),
                    "y": str(a.y),
                    "w": str(a.w),
                    "h": str(a.h),
                    "t": str(a.t),
                    "video_name": video_name,
                    "species_name": species_name,
                }
                for a in tr_anns
            ]
            merged = _merge_csv_rows(
                existing_rows, new_rows, key_fields=["x", "y", "w", "h", "t", "video_name", "species_name"]
            )
            _write_csv_rows(
                out_csv,
                ["x", "y", "w", "h", "t", "video_name", "species_name"],
                merged,
            )
            print(f"[integrated] {out_csv.name} rows: {len(merged)}")

        if SKIP_FINAL_SPLIT_REBUILD:
            print("[integrated] SKIP_FINAL_SPLIT_REBUILD=True; leaving final split unchanged for now.")
        else:
            # Rebuild final dataset split for the affected time-of-day dataset only
            src_initial = target_time_root / INITIAL_DATASET_DIRNAME
            dst_final = target_time_root / FINAL_DATASET_DIRNAME
            metrics = _split_dataset_dir(
                src_initial,
                dst_final,
                train_pct=FINAL_TRAIN_PCT,
                val_pct=FINAL_VAL_PCT,
                test_pct=FINAL_TEST_PCT,
                seed=FINAL_SPLIT_SEED,
                copy_mode=VERSION_COPY_MODE,
            )
            print(f"[integrated] ({dataset_time_dirname}) Final split metrics: {metrics}")

            for split in SPLIT_NAMES:
                for lbl in (CLASS_FIREFLY, CLASS_BACKGROUND):
                    cls_dir = dst_final / split / lbl
                    prefix = (
                        PATCH_LOCATIONS_SPLIT_PREFIX
                        if lbl == CLASS_FIREFLY
                        else PATCH_LOCATIONS_BACKGROUND_SPLIT_PREFIX
                    )
                    out_csv = target_time_root / f"{prefix}{split}.csv"
                    nrows = _write_split_patch_locations_csv(cls_dir, out_csv, include_species=True)
                    print(f"[integrated] Wrote {out_csv} rows={nrows}")
    else:
        print("\nNo TRAIN rows; skipping patch extraction + TRAIN dataset updates.")

    train_after = _collect_train_stats(TRAIN_DATASETS_ROOT)
    _print_stats("TRAIN stats (after)", train_after)

    print(f"\n✅ Done. Batch export: {export_dir}")
    return export_dir


def run_ingestor_core(
    *,
    annotations_csv: Path,
    video_path: Path,
    data_root: Path,
    train_fraction: float,
    train_val_seed: int,
    auto_load_sibling: bool,
    single_species_target_version_dir: Path | None = None,
    integrated_target_version_dir: Path | None = None,
    skip_final_split_rebuild: bool = False,
    scaler_overrides: Dict[str, Any] | None = None,
    dry_run: bool = False,
) -> Path | None:
    if dry_run:
        print(f"[dry-run] Would run stage1_ingestor_core on: {annotations_csv}")
        return None

    global ANNOTATIONS_CSV_PATH
    global VIDEO_PATH
    global DATA_ROOT
    global BATCH_EXPORT_ROOT
    global TRAIN_DATASETS_ROOT
    global TRAIN_FRACTION_OF_BATCH
    global TRAIN_VAL_SPLIT_SEED
    global AUTO_LOAD_SIBLING_CLASS_CSV
    global SINGLE_SPECIES_TARGET_VERSION_DIR
    global INTEGRATED_TARGET_VERSION_DIR
    global SKIP_FINAL_SPLIT_REBUILD

    ANNOTATIONS_CSV_PATH = Path(annotations_csv).expanduser().resolve()
    VIDEO_PATH = Path(video_path).expanduser().resolve()

    DATA_ROOT = Path(data_root).expanduser().resolve()
    BATCH_EXPORT_ROOT = DATA_ROOT / 'batch_exports'
    TRAIN_DATASETS_ROOT = DATA_ROOT / 'Integrated_prototype_datasets'

    TRAIN_FRACTION_OF_BATCH = float(train_fraction)
    TRAIN_VAL_SPLIT_SEED = int(train_val_seed)
    AUTO_LOAD_SIBLING_CLASS_CSV = bool(auto_load_sibling)

    SINGLE_SPECIES_TARGET_VERSION_DIR = (
        Path(single_species_target_version_dir).expanduser().resolve() if single_species_target_version_dir else None
    )
    INTEGRATED_TARGET_VERSION_DIR = (
        Path(integrated_target_version_dir).expanduser().resolve() if integrated_target_version_dir else None
    )
    SKIP_FINAL_SPLIT_REBUILD = bool(skip_final_split_rebuild)

    if scaler_overrides:
        for k, v in scaler_overrides.items():
            globals()[str(k)] = v

    return _stage1_ingestor_core_main()


# Back-compat aliases (older orchestrator revisions used these names)
run_species_scaler = run_ingestor_core
stage_pairs_for_species_scaler = stage_pairs_for_ingestor


def _run_tag() -> str:
    return datetime.now().strftime('%Y%m%d__%H%M%S')


def _next_version_path(root: Path) -> Path:
    prev = _latest_version_dir(root)
    prev_n = _version_num_from_name(prev.name) if prev else 0
    new_n = int(prev_n or 0) + 1
    return root / f"v{new_n}_{_today_tag()}"


def _parse_video_type_and_base_species_name(raw_dir_name: str) -> Tuple[str, str]:
    """
    Extract (video_type, base_species_name) from a raw species folder name.

    Expected convention (case-insensitive):
      - day_<species name>
      - night_<species name>
    """
    raw_dir_name = str(raw_dir_name or "").strip()
    m = re.match(r"^(day|night)[_\-\s]+(.+)$", raw_dir_name, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Raw species dir must start with 'day_' or 'night_' (got {raw_dir_name!r})")
    video_type = str(m.group(1) or "").strip().lower()
    base = str(m.group(2) or "").strip()
    if not base:
        raise ValueError(f"Raw species dir missing species name after prefix (got {raw_dir_name!r})")
    return video_type, base


def _normalize_species_token(raw_dir_name: str, *, base_name: str | None = None) -> str:
    raw_dir_name = str(raw_dir_name or "").strip()
    base_name = str(base_name or "").strip() or raw_dir_name

    # Allow overrides keyed by either:
    #   - the full raw dir name (e.g. "night_Photinus Knulli"), or
    #   - the base species name without the day/night prefix (e.g. "Photinus Knulli")
    override_key: str | None = None
    if raw_dir_name in SPECIES_NAME_OVERRIDES and str(SPECIES_NAME_OVERRIDES[raw_dir_name]).strip():
        override_key = raw_dir_name
    elif base_name in SPECIES_NAME_OVERRIDES and str(SPECIES_NAME_OVERRIDES[base_name]).strip():
        override_key = base_name

    if override_key is not None:
        token = str(SPECIES_NAME_OVERRIDES[override_key]).strip()
        token = token.lower()
        token = re.sub(r"[_\s]+", "-", token)
        token = re.sub(r"[^a-z0-9-]+", "-", token)
        token = re.sub(r"-+", "-", token).strip("-")
        if "_" in token or not token:
            raise ValueError(f"Invalid override species token for {override_key!r}: {SPECIES_NAME_OVERRIDES[override_key]!r}")
        return token

    s = base_name.lower()
    s = re.sub(r"[_\s]+", "-", s)
    s = re.sub(r"[^a-z0-9-]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    if not s:
        raise ValueError(f"Could not normalize species folder name: {base_name!r}")
    if "_" in s:
        # Guardrail: stage1_ingestor_core identity parsing breaks if species token contains underscores.
        s = s.replace("_", "-")
    return s


def _iter_species_dirs(raw_root: Path) -> List[Path]:
    raw_root = Path(raw_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    if not raw_root.is_dir():
        raise NotADirectoryError(raw_root)

    dirs = sorted([p for p in raw_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    only = {str(x).strip() for x in ONLY_RAW_SPECIES_DIRS if str(x).strip()}
    if only:
        dirs = [d for d in dirs if d.name in only]
    return dirs


def _dir_has_any_files(d: Path) -> bool:
    """
    Return True if directory tree contains at least one regular file.
    Used to ignore placeholder/empty species folders.
    """
    d = Path(d)
    try:
        if not d.exists() or (not d.is_dir()):
            return False
    except Exception:
        return False

    try:
        stack = [d]
        while stack:
            cur = stack.pop()
            try:
                with os.scandir(cur) as it:
                    for e in it:
                        try:
                            if e.is_file(follow_symlinks=False):
                                return True
                            if e.is_dir(follow_symlinks=False):
                                stack.append(Path(e.path))
                        except Exception:
                            continue
            except Exception:
                continue
    except Exception:
        # If we can't scan, assume there might be data and let downstream discovery decide.
        return True
    return False


def _video_stem_from_catalog_item(item: Dict[str, Any]) -> str:
    name = str(item.get("video_name") or "").strip()
    if name:
        return Path(name).stem
    stem = str(item.get("video_stem") or "").strip()
    if stem:
        return stem
    path = str(item.get("video_path") or "").strip()
    if path:
        return Path(path).stem
    return ""


def _normalize_catalog_route(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"day", "day_time"}:
        return "day"
    if text in {"night", "night_time"}:
        return "night"
    return text


def _load_root_video_catalog(raw_root: Path) -> Dict[str, Any] | None:
    if not bool(USE_ROOT_VIDEO_CATALOG_IF_PRESENT):
        return None
    p = raw_root / str(VIDEO_CATALOG_FILENAME)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        raise SystemExit(f"Failed to parse root video catalog {p}: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid root video catalog {p}: expected JSON object")
    return data


def _pairs_from_root_video_catalog(
    *,
    pairs: Sequence[ObservedPair],
    video_catalog: Dict[str, Any],
    species_token: str,
    route: str,
) -> Tuple[List[ObservedPair], List[ObservedPair]]:
    pair_by_key: Dict[str, ObservedPair] = {}
    for p in pairs:
        for key in (
            str(p.video_name),
            str(p.video_path.stem),
            str(p.video_path.name),
        ):
            if key:
                pair_by_key[key] = p

    expected_route = _normalize_catalog_route(route)
    matching_entries = [
        item
        for item in list(video_catalog.get("videos") or [])
        if str(item.get("species_name") or "").strip() == str(species_token)
        and _normalize_catalog_route(str(item.get("route") or "")) == expected_route
    ]
    if not matching_entries:
        raise SystemExit(
            f"Root video catalog has no entries for species={species_token!r} route={expected_route!r}. "
            f"Register this species/video set in {VIDEO_CATALOG_FILENAME} first."
        )

    train_stems: List[str] = []
    if bool(INGEST_TRAINING_VIDEOS_FROM_ROOT_CATALOG):
        for item in matching_entries:
            if str(item.get("category") or "").strip().lower() != "training":
                continue
            stem = _video_stem_from_catalog_item(item)
            if stem:
                train_stems.append(stem)

    missing = [s for s in sorted(set(train_stems)) if s not in pair_by_key]
    if missing:
        raise SystemExit(
            "Root video catalog references videos not present as discovered mp4/csv pairs:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )

    train_pairs = [pair_by_key[s] for s in train_stems]
    return train_pairs, []


def _species_already_ingested(*, data_root: Path, species_token: str) -> bool:
    sp_root = data_root / "Integrated_prototype_datasets" / "single species datasets" / str(species_token)
    if not sp_root.exists():
        return False
    latest = _latest_version_dir(sp_root)
    return latest is not None


def _ensure_scaffolds(*, root: Path, data_root: Path) -> None:
    """
    Match the integrated orchestrator's required folder scaffolds under the chosen ROOT.
    """
    root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    (data_root / "batch_exports").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_datasets" / "integrated pipeline datasets").mkdir(parents=True, exist_ok=True)
    (data_root / "Integrated_prototype_datasets" / "single species datasets").mkdir(parents=True, exist_ok=True)


def _scaler_overrides() -> Dict[str, Any]:
    return {
        "VERSION_COPY_MODE": str(DATASET_VERSION_COPY_MODE),
        "AUTO_GENERATE_BACKGROUND_PATCHES": bool(AUTO_GENERATE_BACKGROUND_PATCHES),
        "AUTO_BACKGROUND_TO_FIREFLY_RATIO": float(AUTO_BACKGROUND_TO_FIREFLY_RATIO),
        "AUTO_BACKGROUND_PATCH_SIZE_PX": int(AUTO_BACKGROUND_PATCH_SIZE_PX),
        "AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME": int(AUTO_BACKGROUND_MAX_PATCHES_PER_FRAME),
        "AUTO_BACKGROUND_MAX_FRAME_SAMPLES": int(AUTO_BACKGROUND_MAX_FRAME_SAMPLES),
        "AUTO_BACKGROUND_SEED": int(AUTO_BACKGROUND_SEED),
        "AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS": bool(AUTO_BACKGROUND_FALLBACK_RANDOM_CENTERS),
        "AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS": float(AUTO_BACKGROUND_BLOB_DETECT_SLOW_FRAME_SECONDS),
        "AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES": int(AUTO_BACKGROUND_BLOB_DETECT_DISABLE_AFTER_SLOW_FRAMES),
        "AUTO_BACKGROUND_SBD_MIN_AREA_PX": float(AUTO_BACKGROUND_SBD_MIN_AREA_PX),
        "AUTO_BACKGROUND_SBD_MAX_AREA_SCALE": float(AUTO_BACKGROUND_SBD_MAX_AREA_SCALE),
        "AUTO_BACKGROUND_SBD_MIN_DIST": float(AUTO_BACKGROUND_SBD_MIN_DIST),
        "AUTO_BACKGROUND_SBD_MIN_REPEAT": int(AUTO_BACKGROUND_SBD_MIN_REPEAT),
        "AUTO_BACKGROUND_USE_CLAHE": bool(AUTO_BACKGROUND_USE_CLAHE),
        "AUTO_BACKGROUND_CLAHE_CLIP": float(AUTO_BACKGROUND_CLAHE_CLIP),
        "AUTO_BACKGROUND_CLAHE_TILE": tuple(AUTO_BACKGROUND_CLAHE_TILE),
        "AUTO_BACKGROUND_USE_TOPHAT": bool(AUTO_BACKGROUND_USE_TOPHAT),
        "AUTO_BACKGROUND_TOPHAT_KSIZE": int(AUTO_BACKGROUND_TOPHAT_KSIZE),
        "AUTO_BACKGROUND_USE_DOG": bool(AUTO_BACKGROUND_USE_DOG),
        "AUTO_BACKGROUND_DOG_SIGMA1": float(AUTO_BACKGROUND_DOG_SIGMA1),
        "AUTO_BACKGROUND_DOG_SIGMA2": float(AUTO_BACKGROUND_DOG_SIGMA2),
    }


def _ingest_one_species(
    *,
    root: Path,
    data_root: Path,
    raw_species_dir: Path,
    species_token: str,
    time_of_day: str,
    pairs: Sequence[ObservedPair],
    train_pairs_override: Sequence[ObservedPair] | None,
    val_pairs_override: Sequence[ObservedPair] | None,
    dry_run: bool,
) -> Dict[str, Any]:
    time_of_day = str(time_of_day or "").strip().lower()
    if time_of_day not in {"day_time", "night_time"}:
        raise ValueError(f"time_of_day must be 'day_time' or 'night_time' (got {time_of_day!r})")

    pairs = list(pairs)
    if not pairs:
        raise SystemExit(f"No pairs to ingest for species={species_token} dir={raw_species_dir}")
    if train_pairs_override is not None or val_pairs_override is not None:
        train_pairs = list(train_pairs_override or [])
        val_pairs = list(val_pairs_override or [])
    else:
        train_pairs, val_pairs = split_pairs_train_vs_val(pairs, train_fraction=float(TRAIN_PAIR_FRACTION))

    print(
        "[ingestor-only] Discovered pairs:",
        f"species={species_token} time_of_day={time_of_day} dir={raw_species_dir} "
        f"total={len(pairs)} train={len(train_pairs)} val={len(val_pairs)}",
    )
    if train_pairs_override is not None or val_pairs_override is not None:
        print(
            f"[ingestor-only] Using explicit split from root catalog {VIDEO_CATALOG_FILENAME} "
            f"(train={len(train_pairs)} val={len(val_pairs)})"
        )

    # Dataset roots produced by the inlined stage1 ingestor core.
    integrated_root = data_root / "Integrated_prototype_datasets" / "integrated pipeline datasets"
    single_root = data_root / "Integrated_prototype_datasets" / "single species datasets" / species_token
    # Stage CSVs into canonical names expected by stage1_ingestor_core.
    staging_dir = (
        data_root
        / "batch_exports"
        / "ingestor_only_observed_dir_staging"
        / f"{_run_tag()}__{_safe_name(raw_species_dir.name)}"
    )

    staged_pairs = stage_pairs_for_ingestor(
        pairs,
        species_name=species_token,
        staging_dir=staging_dir,
        time_of_day=time_of_day,
        dry_run=bool(dry_run),
    )
    staged_by_video: Dict[str, StagedPair] = {sp.pair.video_name: sp for sp in staged_pairs}
    batch_exports_root = data_root / "batch_exports"

    overrides = _scaler_overrides()

    batch_integrated_target_ver: Path | None = None
    batch_single_target_ver: Path | None = None

    if ONE_DATASET_VERSION_PER_BATCH:
        batch_integrated_target_ver = _next_version_path(integrated_root)
        batch_single_target_ver = _next_version_path(single_root)

        print(
            "[ingestor-only] Batch ingest versions:",
            f"integrated={batch_integrated_target_ver.name} single_species={batch_single_target_ver.name}",
        )

        # TRAIN ingestion (append into a single target version; finalize split once at the end).
        for i, p in enumerate(train_pairs):
            sp = staged_by_video[p.video_name]
            finalize = (i == (len(train_pairs) - 1))
            export_dir = run_ingestor_core(
                annotations_csv=sp.staged_firefly_csv,
                video_path=p.video_path,
                data_root=data_root,
                train_fraction=1.0,
                train_val_seed=1337,
                auto_load_sibling=bool(AUTO_LOAD_SIBLING_CLASS_CSV),
                single_species_target_version_dir=batch_single_target_ver,
                integrated_target_version_dir=batch_integrated_target_ver,
                skip_final_split_rebuild=(not finalize),
                scaler_overrides=overrides,
                dry_run=bool(dry_run),
            )
            if CLEAN_BATCH_EXPORTS_AFTER_SUCCESS and (not dry_run):
                _safe_cleanup_batch_exports_path(export_dir, batch_exports_root=batch_exports_root)
    else:
        # Legacy behavior: one dataset version per video ingested.
        train_video_names = {p.video_name for p in train_pairs}
        for p in pairs:
            sp = staged_by_video[p.video_name]
            if p.video_name in train_video_names:
                export_dir = run_ingestor_core(
                    annotations_csv=sp.staged_firefly_csv,
                    video_path=p.video_path,
                    data_root=data_root,
                    train_fraction=1.0,
                    train_val_seed=1337,
                    auto_load_sibling=bool(AUTO_LOAD_SIBLING_CLASS_CSV),
                    scaler_overrides=overrides,
                    dry_run=bool(dry_run),
                )
                if CLEAN_BATCH_EXPORTS_AFTER_SUCCESS and (not dry_run):
                    _safe_cleanup_batch_exports_path(export_dir, batch_exports_root=batch_exports_root)

    if CLEAN_BATCH_EXPORTS_AFTER_SUCCESS and (not dry_run):
        _safe_cleanup_batch_exports_path(staging_dir, batch_exports_root=batch_exports_root)

    # Return latest versions for logging/visibility.
    integrated_ver = _latest_version_dir(integrated_root)
    single_ver = _latest_version_dir(single_root)

    ingested_pairs: List[Dict[str, Any]] = []
    for p in train_pairs:
        ingested_pairs.append(
            {
                "species_token": str(species_token),
                "video_name": str(p.video_name),
                "video_path": str(p.video_path),
                "firefly_csv": str(p.firefly_csv),
                "background_csv": str(p.background_csv) if p.background_csv else None,
                "split": "train",
            }
        )
    for p in val_pairs:
        ingested_pairs.append(
            {
                "species_token": str(species_token),
                "video_name": str(p.video_name),
                "video_path": str(p.video_path),
                "firefly_csv": str(p.firefly_csv),
                "background_csv": str(p.background_csv) if p.background_csv else None,
                "split": "validation",
            }
        )

    return {
        "species": species_token,
        "raw_species_dir": str(raw_species_dir),
        "time_of_day": str(time_of_day),
        "n_pairs_total": int(len(pairs)),
        "n_pairs_train": int(len(train_pairs)),
        "n_pairs_validation": int(len(val_pairs)),
        "ingested_pairs": ingested_pairs,
        "paths": {
            "root": str(root),
            "data_root": str(data_root),
            "integrated_dataset_version": integrated_ver.name if integrated_ver else None,
            "single_species_dataset_version": single_ver.name if single_ver else None,
        },
    }


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch-classification dataset ingestor (ingest missing videos per species)."
    )
    p.add_argument("--raw-root", type=str, default="", help="Override RAW_VIDEOS_ROOT.")
    p.add_argument("--root", type=str, default="", help="Override ROOT_PATH.")
    p.add_argument("--log-root", type=str, default="", help="Override LOG_ROOT_PATH (where changelog lives).")
    p.add_argument("--dry-run", action="store_true", default=False, help="Do not run actual ingestion.")
    p.add_argument("--only-species", action="append", default=[], help="Restrict to a raw species folder name (repeatable).")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv)

    raw_root_arg = str(args.raw_root or "").strip() or str(RAW_VIDEOS_ROOT or "").strip()
    root_arg = str(args.root or "").strip() or str(ROOT_PATH or "").strip()
    log_root_arg = str(args.log_root or "").strip() or str(LOG_ROOT_PATH or "").strip()
    dry_run = bool(DRY_RUN) or bool(args.dry_run)

    if not raw_root_arg:
        print("Set RAW_VIDEOS_ROOT at top of file or pass --raw-root.", file=sys.stderr)
        return 2
    if not root_arg:
        print("Set ROOT_PATH at top of file or pass --root.", file=sys.stderr)
        return 2
    if not log_root_arg:
        print("Set LOG_ROOT_PATH at top of file or pass --log-root.", file=sys.stderr)
        return 2

    raw_root = Path(raw_root_arg).expanduser().resolve()
    root = Path(root_arg).expanduser().resolve()
    log_root = Path(log_root_arg).expanduser().resolve()
    data_root = root / str(DATA_SUBDIR)

    # Optional CLI override for ONLY_RAW_SPECIES_DIRS
    cli_only = {str(x).strip() for x in (args.only_species or []) if str(x).strip()}
    global ONLY_RAW_SPECIES_DIRS
    if cli_only:
        ONLY_RAW_SPECIES_DIRS = tuple(sorted(cli_only))

    _ensure_scaffolds(root=root, data_root=data_root)
    video_catalog = _load_root_video_catalog(raw_root)
    video_catalog_path = raw_root / str(VIDEO_CATALOG_FILENAME)

    log_path = (log_root / str(CHANGE_LOG_FILENAME)).expanduser().resolve()
    if not dry_run:
        log_root.mkdir(parents=True, exist_ok=True)

    # Build an ingestion index from the centralized change log.
    ingestion_index = build_ingestion_index(log_path)

    raw_species_dirs = _iter_species_dirs(raw_root)
    if not raw_species_dirs:
        print(f"[ingestor-only] No species folders found under: {raw_root}")
        return 0

    candidates: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for sp_dir in raw_species_dirs:
        if not _dir_has_any_files(sp_dir):
            skipped.append(f"Skipping raw species dir {sp_dir.name!r}: raw species dir is empty")
            continue

        try:
            video_type, base_species_name = _parse_video_type_and_base_species_name(sp_dir.name)
            token = _normalize_species_token(sp_dir.name, base_name=base_species_name)
            time_of_day = f"{video_type}_time"
        except Exception as e:
            skipped.append(f"Skipping raw species dir {sp_dir.name!r}: parse failed: {e}")
            continue

        if ONLY_INGEST_NOT_YET_INGESTED and _species_already_ingested(data_root=data_root, species_token=token):
            skipped.append(f"Skipping {sp_dir.name!r} -> {token}: already ingested (ONLY_INGEST_NOT_YET_INGESTED=True)")
            continue

        try:
            pairs = discover_observed_pairs(sp_dir)
        except SystemExit as e:
            skipped.append(f"Skipping {sp_dir.name!r} -> {token}: discover pairs failed: {e}")
            continue
        except Exception as e:
            skipped.append(f"Skipping {sp_dir.name!r} -> {token}: discover pairs error: {e}")
            continue

        already_ingested = set(ingestion_index.get(token, {}).keys())
        explicit_train_pairs: List[ObservedPair] | None = None
        explicit_val_pairs: List[ObservedPair] | None = None
        if video_catalog is not None:
            try:
                explicit_train_pairs, explicit_val_pairs = _pairs_from_root_video_catalog(
                    pairs=pairs,
                    video_catalog=video_catalog,
                    species_token=token,
                    route=time_of_day,
                )
            except SystemExit as e:
                skipped.append(f"Skipping {sp_dir.name!r} -> {token}: root video catalog failed: {e}")
                continue

            explicit_train_pairs = [p for p in explicit_train_pairs if str(p.video_name) not in already_ingested]
            explicit_val_pairs = [p for p in explicit_val_pairs if str(p.video_name) not in already_ingested]
            pairs_to_ingest = list(explicit_train_pairs) + list(explicit_val_pairs)
        else:
            pairs_to_ingest = [p for p in pairs if str(p.video_name) not in already_ingested]

        if not pairs_to_ingest:
            skipped.append(f"Skipping {sp_dir.name!r} -> {token}: all {len(pairs)} video(s) already ingested")
            continue

        candidates.append(
            {
                "raw_species_dir": sp_dir,
                "species_token": token,
                "time_of_day": time_of_day,
                "pairs_total": int(len(pairs)),
                "pairs_to_ingest": list(pairs_to_ingest),
                "already_ingested_video_names": sorted(already_ingested),
                "video_catalog_path": str(video_catalog_path) if video_catalog is not None else None,
                "explicit_train_pairs": list(explicit_train_pairs or []),
                "explicit_val_pairs": list(explicit_val_pairs or []),
            }
        )

    if not candidates:
        print("[ingestor-only] Up to date: no new videos found to ingest (or all species were skipped).")
        for msg in skipped[:50]:
            print(f"[ingestor-only] Note: {msg}")
        return 0

    print(f"[ingestor-only] Found {len(candidates)} species with new videos to ingest. dry_run={dry_run}")
    for c in candidates:
        sp_dir = Path(c["raw_species_dir"])
        token = str(c["species_token"])
        time_of_day = str(c.get("time_of_day") or "")
        print(
            f"  - {sp_dir.name} -> {token} ({time_of_day}): ingest_videos={len(c['pairs_to_ingest'])}/{int(c['pairs_total'])} total",
        )
        if c.get("video_catalog_path"):
            print(
                f"      root catalog split: train={len(c.get('explicit_train_pairs') or [])} "
                f"val={len(c.get('explicit_val_pairs') or [])}"
            )

    results: List[Dict[str, Any]] = []
    for c in candidates:
        sp_dir = Path(c["raw_species_dir"])
        token = str(c["species_token"])
        time_of_day = str(c.get("time_of_day") or "")
        pairs_to_ingest = list(c["pairs_to_ingest"])
        already_ingested_names = list(c.get("already_ingested_video_names") or [])
        explicit_train_pairs = list(c.get("explicit_train_pairs") or [])
        explicit_val_pairs = list(c.get("explicit_val_pairs") or [])

        print(f"\n[ingestor-only] === Ingesting {sp_dir.name} -> {token} ({time_of_day}) ===")
        print(
            "[ingestor-only] Work:",
            f"total_pairs={int(c['pairs_total'])} already_ingested={len(already_ingested_names)} to_ingest={len(pairs_to_ingest)}",
        )

        cfg = SnapshotConfig(
            root=root,
            scopes=[data_root],
        )
        meta = {
            "actor": "patch_classification_models_dataset_ingestor",
            "raw_species_dir": str(sp_dir),
            "species_token": str(token),
            "time_of_day": str(time_of_day),
            "dry_run": bool(dry_run),
            "raw_pairs_total": int(c["pairs_total"]),
            "raw_pairs_to_ingest": int(len(pairs_to_ingest)),
            "raw_pairs_already_ingested": int(len(already_ingested_names)),
            "video_catalog_path": c.get("video_catalog_path"),
            # NOTE: we intentionally do not inline all existing ingested history here; only what we ingest now.
        }

        with ChangeLogRun(cfg=cfg, log_path=log_path, meta=meta, enabled=bool(ENABLE_CODEX_CHANGE_LOG)):
            try:
                out = _ingest_one_species(
                    root=root,
                    data_root=data_root,
                    raw_species_dir=sp_dir,
                    species_token=token,
                    time_of_day=str(time_of_day),
                    pairs=pairs_to_ingest,
                    train_pairs_override=explicit_train_pairs if c.get("video_catalog_path") else None,
                    val_pairs_override=explicit_val_pairs if c.get("video_catalog_path") else None,
                    dry_run=dry_run,
                )
                meta["ingested_pairs"] = list(out.get("ingested_pairs") or [])
            except SystemExit as e:
                # stage1_ingestor uses SystemExit for "no data" type errors; treat as skip.
                skipped.append(f"Ingest skipped for {sp_dir.name!r} -> {token}: {e}")
                continue
            except Exception as e:
                print(f"[ingestor-only] ERROR ingesting {sp_dir.name} -> {token}: {e}", file=sys.stderr)
                raise

        results.append(out)
        print(
            "[ingestor-only] Done:",
            f"species={token} integrated_ver={out['paths'].get('integrated_dataset_version')} single_ver={out['paths'].get('single_species_dataset_version')}",
        )

    if results:
        print(f"\n[ingestor-only] Completed {len(results)} ingestion job(s).")
        print(f"[ingestor-only] Change log → {log_path}")
    else:
        print("\n[ingestor-only] No ingestion jobs were executed.")
        print(f"[ingestor-only] Change log → {log_path}")

    for msg in skipped[:50]:
        print(f"[ingestor-only] Note: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
