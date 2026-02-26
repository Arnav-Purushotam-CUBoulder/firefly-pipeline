#!/usr/bin/env python3
from __future__ import annotations

import base64
import dataclasses
import difflib
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


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
    path: str  # relative (POSIX)
    size: int
    mtime_ns: int
    sha256: str | None
    inline_kind: str | None  # "text" | "b64"
    inline_data: str | None


@dataclass
class DirRec:
    path: str  # relative (POSIX)
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
    ext = p.suffix.lower()
    return ext in {".json", ".jsonl", ".txt", ".md", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".py", ".ini"}


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
    """
    Return (kind, data) where kind is:
      - "text" for UTF-8-ish text
      - "b64" for binary (base64)
    """
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

    # Heuristic: treat as text if it has no NUL bytes and is decodable.
    if b"\x00" not in raw:
        try:
            txt = raw.decode("utf-8", errors="strict")
            return "text", txt
        except Exception:
            pass
    return "b64", base64.b64encode(raw).decode("ascii")


def _is_bulk_dir(dir_path: Path) -> bool:
    """
    Bulk dirs contain huge numbers of patch images. We do not enumerate their files.

    Current heuristics:
    - .../(initial dataset|train_patches)/{firefly,background}
    - .../final dataset/{train,val,test}/{firefly,background}
    - .../inference outputs/<run_id>/...  (treat each run dir as bulk)
    """
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


def _bulk_dir_summary(dir_path: Path) -> tuple[int, int]:
    """
    Best-effort: count immediate files + bytes under dir_path (no recursion).
    """
    n = 0
    total = 0
    try:
        with os.scandir(dir_path) as it:
            for e in it:
                try:
                    if e.is_file(follow_symlinks=False):
                        n += 1
                        try:
                            total += int(e.stat(follow_symlinks=False).st_size)
                        except Exception:
                            pass
                except Exception:
                    continue
    except Exception:
        return 0, 0
    return int(n), int(total)


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

    scopes = _iter_scopes(cfg)
    stack: List[Path] = list(reversed(scopes))
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

        # Record directory
        try:
            d_stat = d_res.stat()
            d_mtime_ns = int(d_stat.st_mtime_ns)
        except Exception:
            d_mtime_ns = 0

        bulk = _is_bulk_dir(d_res)
        if bulk:
            # Do not enumerate huge patch dirs; directory mtime is enough to indicate changes.
            dirs[rel] = DirRec(path=rel, bulk=True, mtime_ns=int(d_mtime_ns), n_files=None, total_bytes=None)
            continue
        dirs[rel] = DirRec(path=rel, bulk=False, mtime_ns=int(d_mtime_ns))

        # Descend
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
                                size = int(st.st_size)
                                mtime_ns = int(st.st_mtime_ns)
                            except Exception:
                                continue

                            sha = _sha256_file(p_res, max_bytes=int(cfg.max_hash_bytes))
                            inline_kind, inline_data = _read_inline(p_res, max_bytes=int(cfg.max_inline_bytes))
                            files[relp] = FileRec(
                                path=relp,
                                size=size,
                                mtime_ns=mtime_ns,
                                sha256=sha,
                                inline_kind=inline_kind,
                                inline_data=inline_data,
                            )
                        else:
                            # Skip symlinks/other special entries for now.
                            continue
                    except Exception:
                        continue
        except Exception:
            continue

    return Snapshot(created_at=_now_iso(), root=str(root), dirs=dirs, files=files)


def _file_changed(a: FileRec, b: FileRec) -> bool:
    if a.size != b.size:
        return True
    # Prefer hashes when available.
    if a.sha256 is not None and b.sha256 is not None:
        return a.sha256 != b.sha256
    # Fall back to mtime (best-effort).
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
        if not (a.bulk and b.bulk):
            continue
        if int(a.mtime_ns or 0) != int(b.mtime_ns or 0):
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

    # Rollback-focused logging:
    # - For added files, we usually only need metadata (rollback = delete).
    # - For removed/modified files, we keep inline data when small (rollback = restore).
    added_payload = [_file_payload(after.files[p], include_inline=False) for p in files_added]
    removed_payload = [_file_payload(before.files[p], include_inline=True) for p in files_removed]

    modified_payload: List[Dict[str, Any]] = []
    for p in files_modified:
        a = before.files[p]
        b = after.files[p]
        diff_txt = None
        if a.inline_kind == "text" and b.inline_kind == "text" and a.inline_data is not None and b.inline_data is not None:
            a_lines = a.inline_data.splitlines(keepends=False)
            b_lines = b.inline_data.splitlines(keepends=False)
            ud = list(
                difflib.unified_diff(
                    a_lines,
                    b_lines,
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
        "files_added": added_payload,
        "files_removed": removed_payload,
        "files_modified": modified_payload,
        "bulk_dirs_changed": bulk_stats_changed,
    }


def append_change_log(
    *,
    log_path: Path,
    record: Dict[str, Any],
) -> None:
    log_path = Path(log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class ChangeLogRun:
    """
    Context manager that snapshots before/after and appends a JSONL record.
    """

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
        before = self._before
        if before is None:
            return

        try:
            after = take_snapshot(self.cfg)
            changes = diff_snapshots(before=before, after=after, cfg=self.cfg)
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

            rec = {
                "schema_version": 1,
                "timestamp": _now_iso(),
                "meta": dict(self.meta),
                "monitored_root": str(Path(self.cfg.root).expanduser().resolve()),
                "scopes": [str(p) for p in _iter_scopes(self.cfg)],
                "had_error": bool(had_error),
                "error": (str(exc) if had_error else None),
                "snapshot": {"before_created_at": before.created_at, "after_created_at": after.created_at},
                "changes": changes,
            }
            append_change_log(log_path=self.log_path, record=rec)
        except Exception:
            # Best-effort only: never break the main automation if logging fails.
            return


def default_log_path(log_root: Path) -> Path:
    return Path(log_root) / "codex_change_log.jsonl"


def iter_change_log_records(log_path: Path) -> Iterator[Dict[str, Any]]:
    """
    Best-effort iterator over JSONL records written by ChangeLogRun.
    Skips malformed lines.
    """
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
    """
    Build a (species_token -> video_name -> info) index from the append-only change log.

    This is intended as a centralized "what has been ingested?" source of truth.
    Producers:
      - ingestor_only.py (actor="ingestor_only")
      - orchestrator.py  (actor="orchestrator")

    Expected metadata shape:
      meta["ingested_pairs"] = [
        {"video_name": "...", "split": "train"|"validation", ...},
        ...
      ]

    Returns a dict keyed by species_token (or species_name), then video_name.
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rec in iter_change_log_records(log_path):
        try:
            if bool(rec.get("had_error")):
                continue
            meta = rec.get("meta")
            if not isinstance(meta, dict):
                continue
            if bool(meta.get("dry_run")):
                continue
            actor = str(meta.get("actor") or "").strip()
            if actor not in {"ingestor_only", "orchestrator"}:
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
                    payload = {}
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

                info: Dict[str, Any] = {"timestamp": ts, "actor": actor}
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
