#!/usr/bin/env python3
from __future__ import annotations

"""
Ingest YOLO image/label pairs into the day-v3 global training dataset.

Primary use case:
- A new species batch arrives as a zip of YOLO labels and/or image+label pairs.
- Matching images may optionally be supplied through one or more
  `--local-image-root` directories when the source provides labels only.
- This script validates the pairs, copies them into the global YOLO train split,
  writes a manifest, and can optionally delete the temporary zip/extraction once
  the copied dataset pairs are safely in place.
"""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

YOLO_ROOT = Path("/mnt/Samsung_SSD_2TB/integrated prototype data/v3 daytime YOLO model data")
# Do not edit this path. The ingest/training code now depends on this exact
# species-folder dataset layout directly under YOLO_ROOT / "dataset".
DEFAULT_DATASET_ROOT = YOLO_ROOT / "dataset"
DEFAULT_MANIFEST_ROOT = YOLO_ROOT / "ingest manifests"
DEFAULT_EXTRACT_ROOT = YOLO_ROOT / "_tmp_ingest_extracts"
# Set the species being ingested here. If this species already exists under the
# dataset root, new pairs are appended into that existing folder. If it does not
# exist yet, the script creates the species folder automatically.
DEFAULT_SPECIES_TAG = "day_Photinus greeni"
DATASET_MANIFEST_FILENAME = "dataset_manifest.json"
DATASET_LAYOUT_VERSION = "species_grouped_v1"

SPECIES_FOLDER_ALIASES: Dict[str, str] = {
    "day-photinus-greeni": "photinus-greeni",
    "night-photinus-greeni": "photinus-greeni",
    "photinus-greeni": "photinus-greeni",
    "bicellonycha-wickershamorum": "bicellonycha-wickershamorum",
    "photinus-acuminatus": "photinus-acuminatus",
    "photuris-bethaniensis": "photuris-bethaniensis",
    "pyrallis-gopro": "pyrallis-gopro",
    "pyrallis-gopro-v3": "pyrallis-gopro",
    "pyractomena-pyralis": "pyrallis-gopro",
}


@dataclass(frozen=True)
class PairPlan:
    stem: str
    label_path: Path
    image_path: Path
    dest_stem: str
    dest_label_path: Path
    dest_image_path: Path
    collision_strategy: str


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_file_content(a: Path, b: Path) -> bool:
    if not a.exists() or not b.exists():
        return False
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
    except Exception:
        return False
    return _sha256(a) == _sha256(b)


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return (p for p in root.rglob("*") if p.is_file())


def _safe_slug(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in {" ", "-", "_"}:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "source"


def _canonical_species_folder_name(species_tag: str) -> str:
    slug = _safe_slug(species_tag)
    if slug.startswith("day-"):
        slug = slug[len("day-") :]
    elif slug.startswith("night-"):
        slug = slug[len("night-") :]
    slug = SPECIES_FOLDER_ALIASES.get(slug, slug)
    if not slug:
        raise ValueError(f"Could not derive a species folder name from {species_tag!r}")
    return slug


def _species_dataset_paths(dataset_root: Path, species_tag: str) -> tuple[str, Path, Path, Path]:
    folder_name = _canonical_species_folder_name(species_tag)
    species_dir = dataset_root / folder_name
    return folder_name, species_dir, species_dir / "images", species_dir / "labels"


def _ensure_species_dataset_dirs(dataset_root: Path, species_tag: str, *, dry_run: bool) -> tuple[str, Path, Path, Path, bool]:
    folder_name, species_dir, images_dir, labels_dir = _species_dataset_paths(dataset_root, species_tag)
    created = not species_dir.exists()
    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
    return folder_name, species_dir, images_dir, labels_dir, created


def _build_dataset_manifest(dataset_root: Path) -> Dict[str, object]:
    species_records: List[Dict[str, object]] = []
    total_images = 0
    total_labels = 0

    for species_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        images_dir = species_dir / "images"
        labels_dir = species_dir / "labels"
        if not images_dir.exists() and not labels_dir.exists():
            continue

        image_files = sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if images_dir.exists() else []
        label_files = sorted([p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]) if labels_dir.exists() else []
        image_stems = {p.stem for p in image_files}
        label_stems = {p.stem for p in label_files}
        total_images += len(image_files)
        total_labels += len(label_files)

        species_records.append(
            {
                "species_folder": species_dir.name,
                "species_dir": str(species_dir),
                "images_dir": str(images_dir),
                "labels_dir": str(labels_dir),
                "n_images": len(image_files),
                "n_labels": len(label_files),
                "missing_labels": sorted(image_stems - label_stems),
                "missing_images": sorted(label_stems - image_stems),
            }
        )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_layout_version": DATASET_LAYOUT_VERSION,
        "dataset_root": str(dataset_root),
        "n_species_folders": len(species_records),
        "n_images_total": total_images,
        "n_labels_total": total_labels,
        "species": species_records,
    }


def _write_dataset_manifest(dataset_root: Path, *, dry_run: bool) -> Path:
    manifest_path = dataset_root / DATASET_MANIFEST_FILENAME
    payload = _build_dataset_manifest(dataset_root)
    if not dry_run:
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _validate_yolo_label_file(path: Path) -> List[str]:
    issues: List[str] = []
    try:
        lines = _read_nonempty_lines(path)
    except Exception as e:
        return [f"could not read label file {path}: {e}"]

    for i, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 5:
            issues.append(f"{path}: line {i} has {len(parts)} fields, expected at least 5")
            continue
        try:
            int(parts[0])
            for val in parts[1:5]:
                float(val)
        except Exception:
            issues.append(f"{path}: line {i} is not valid YOLO numeric text")
    return issues


def _build_image_index(roots: Sequence[Path]) -> Dict[str, List[Path]]:
    out: Dict[str, List[Path]] = {}
    for root in roots:
        if not root.exists():
            continue
        for p in _iter_files(root):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            out.setdefault(p.stem, []).append(p)
    return out


def _select_best_image(stem: str, candidates: Sequence[Path], extracted_root: Optional[Path]) -> Path:
    if len(candidates) == 1:
        return candidates[0]
    if extracted_root is not None:
        extracted_root = extracted_root.resolve()
        in_extract = [p for p in candidates if extracted_root in p.resolve().parents]
        if len(in_extract) == 1:
            return in_extract[0]
    preferred = sorted(candidates, key=lambda p: (p.suffix.lower() != ".png", len(str(p)), str(p)))
    return preferred[0]


def _discover_label_files(source_root: Path) -> List[Path]:
    labels: List[Path] = []
    for p in _iter_files(source_root):
        if p.suffix.lower() != ".txt":
            continue
        # Skip obvious non-label text files if they appear in the archive.
        if p.name.lower() in {"classes.txt", "readme.txt", "readme.roboflow.txt"}:
            continue
        labels.append(p)
    return sorted(labels)


def _discover_coco_annotation_files(source_root: Path) -> List[Path]:
    out: List[Path] = []
    for p in _iter_files(source_root):
        if p.name == "_annotations.coco.json":
            out.append(p)
    return sorted(out)


def _find_existing_image_for_coco_file(*, source_root: Path, coco_path: Path, file_name: str) -> Path:
    rel = Path(str(file_name))
    candidates = [
        coco_path.parent / rel,
        source_root / rel,
        source_root / rel.name,
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Could not resolve COCO image {file_name!r} under {source_root}")


def _coco_bbox_to_yolo_line(*, bbox: Sequence[object], width: int, height: int) -> Optional[str]:
    if len(bbox) < 4:
        return None
    try:
        x = float(bbox[0])
        y = float(bbox[1])
        w = float(bbox[2])
        h = float(bbox[3])
    except Exception:
        return None
    if width <= 0 or height <= 0 or w <= 0 or h <= 0:
        return None

    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0:
        return None

    cx = (x1 + x2) / 2.0 / float(width)
    cy = (y1 + y2) / 2.0 / float(height)
    nw = bw / float(width)
    nh = bh / float(height)
    return f"0 {cx:.10f} {cy:.10f} {nw:.10f} {nh:.10f}"


def _materialize_yolo_labels_from_coco(*, source_root: Path, generated_root: Path) -> Tuple[List[Path], Dict[str, object]]:
    coco_paths = _discover_coco_annotation_files(source_root)
    if not coco_paths:
        return [], {
            "label_source_format": None,
            "coco_json_files": 0,
            "generated_label_files": 0,
        }

    label_paths: List[Path] = []
    total_images = 0
    total_annotations = 0

    for coco_path in coco_paths:
        payload = json.loads(coco_path.read_text(encoding="utf-8"))
        images = list(payload.get("images") or [])
        annotations = list(payload.get("annotations") or [])
        total_images += len(images)
        total_annotations += len(annotations)

        anns_by_image: Dict[int, List[dict]] = defaultdict(list)
        for ann in annotations:
            try:
                image_id = int(ann["image_id"])
            except Exception:
                continue
            anns_by_image[image_id].append(dict(ann))

        for im in images:
            try:
                image_id = int(im["id"])
                file_name = str(im["file_name"])
                width = int(im["width"])
                height = int(im["height"])
            except Exception as e:
                raise RuntimeError(f"Invalid COCO image entry in {coco_path}: {im!r} ({e})") from e

            image_path = _find_existing_image_for_coco_file(
                source_root=source_root,
                coco_path=coco_path,
                file_name=file_name,
            )
            rel_label = Path(file_name).with_suffix(".txt")
            label_path = generated_root / rel_label
            label_path.parent.mkdir(parents=True, exist_ok=True)

            lines: List[str] = []
            for ann in anns_by_image.get(image_id, []):
                line = _coco_bbox_to_yolo_line(
                    bbox=list(ann.get("bbox") or []),
                    width=width,
                    height=height,
                )
                if line is not None:
                    lines.append(line)

            label_path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")
            label_paths.append(label_path)

    return sorted(label_paths), {
        "label_source_format": "coco_json",
        "coco_json_files": len(coco_paths),
        "generated_label_files": len(label_paths),
        "coco_images": total_images,
        "coco_annotations": total_annotations,
    }


def _prepare_label_files(*, source_root: Path, generated_root: Path) -> Tuple[List[Path], Dict[str, object]]:
    label_files = _discover_label_files(source_root)
    if label_files:
        return label_files, {
            "label_source_format": "yolo_txt",
            "coco_json_files": 0,
            "generated_label_files": 0,
        }
    return _materialize_yolo_labels_from_coco(source_root=source_root, generated_root=generated_root)


def _resolve_source_root(source_path: Path, extract_root: Path) -> tuple[Path, Optional[Path], Optional[Path]]:
    """
    Return (resolved_source_root, extracted_dir_if_any, zip_path_if_any).
    """
    source_path = source_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if source_path.is_dir():
        return source_path, None, None

    if source_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a directory or .zip source, got: {source_path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extract_dir = extract_root / f"{source_path.stem}__{stamp}"
    extract_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(source_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir, extract_dir, source_path


def _plan_pairs(
    *,
    label_files: Sequence[Path],
    source_root: Path,
    extracted_root: Optional[Path],
    local_image_roots: Sequence[Path],
    train_images_dir: Path,
    train_labels_dir: Path,
    source_slug: str,
) -> tuple[List[PairPlan], Dict[str, object]]:
    if not label_files:
        raise RuntimeError(f"No .txt label files found under {source_root}")

    image_index = _build_image_index([source_root, *local_image_roots])
    problems: List[str] = []
    label_validation_issues: List[str] = []
    plans: List[PairPlan] = []
    skipped_existing = 0
    renamed_due_to_collision = 0

    for label_path in label_files:
        label_validation_issues.extend(_validate_yolo_label_file(label_path))
        stem = label_path.stem
        candidates = image_index.get(stem) or []
        if not candidates:
            problems.append(f"Missing image for label stem '{stem}' from {label_path}")
            continue
        image_path = _select_best_image(stem, candidates, extracted_root)

        dest_stem = stem
        dest_label_path = train_labels_dir / f"{dest_stem}.txt"
        dest_image_path = train_images_dir / f"{dest_stem}{image_path.suffix.lower()}"
        collision_strategy = "direct_copy"

        if dest_label_path.exists() or dest_image_path.exists():
            if dest_label_path.exists() and dest_image_path.exists():
                if _same_file_content(label_path, dest_label_path) and _same_file_content(image_path, dest_image_path):
                    skipped_existing += 1
                    continue
            suffix_base = f"__src_{source_slug}"
            candidate_stem = f"{stem}{suffix_base}"
            n = 1
            while True:
                test_stem = candidate_stem if n == 1 else f"{candidate_stem}__dup{n}"
                test_label = train_labels_dir / f"{test_stem}.txt"
                test_image = train_images_dir / f"{test_stem}{image_path.suffix.lower()}"
                if not test_label.exists() and not test_image.exists():
                    dest_stem = test_stem
                    dest_label_path = test_label
                    dest_image_path = test_image
                    collision_strategy = "renamed_on_collision"
                    renamed_due_to_collision += 1
                    break
                if test_label.exists() and test_image.exists():
                    if _same_file_content(label_path, test_label) and _same_file_content(image_path, test_image):
                        skipped_existing += 1
                        dest_stem = ""
                        break
                n += 1
            if not dest_stem:
                continue

        plans.append(
            PairPlan(
                stem=stem,
                label_path=label_path,
                image_path=image_path,
                dest_stem=dest_stem,
                dest_label_path=dest_label_path,
                dest_image_path=dest_image_path,
                collision_strategy=collision_strategy,
            )
        )

    if label_validation_issues:
        problems.extend(label_validation_issues)
    if problems:
        raise RuntimeError("Validation failed:\n" + "\n".join(problems))

    summary = {
        "label_files_found": len(label_files),
        "pairs_to_copy": len(plans),
        "skipped_existing_pairs": skipped_existing,
        "renamed_due_to_collision": renamed_due_to_collision,
    }
    return plans, summary


def _copy_pairs(plans: Sequence[PairPlan], *, dry_run: bool) -> None:
    for plan in plans:
        if dry_run:
            print(f"[dry-run] {plan.image_path} -> {plan.dest_image_path}")
            print(f"[dry-run] {plan.label_path} -> {plan.dest_label_path}")
            continue
        plan.dest_image_path.parent.mkdir(parents=True, exist_ok=True)
        plan.dest_label_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.image_path, plan.dest_image_path)
        shutil.copy2(plan.label_path, plan.dest_label_path)


def _write_manifest(
    *,
    manifest_root: Path,
    species_tag: str,
    source_input: Path,
    source_root: Path,
    dataset_root: Path,
    species_folder_name: str,
    species_dir: Path,
    species_images_dir: Path,
    species_labels_dir: Path,
    local_image_roots: Sequence[Path],
    plans: Sequence[PairPlan],
    summary: Dict[str, object],
    extracted_root: Optional[Path],
    zip_path: Optional[Path],
    dry_run: bool,
) -> Path:
    manifest_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = manifest_root / f"{ts}__{_safe_slug(species_tag)}__yolo_ingest_manifest.json"
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "species_tag": species_tag,
        "dry_run": bool(dry_run),
        "source_input": str(source_input),
        "resolved_source_root": str(source_root),
        "extracted_root": str(extracted_root) if extracted_root is not None else None,
        "zip_path": str(zip_path) if zip_path is not None else None,
        "dataset_layout_version": DATASET_LAYOUT_VERSION,
        "dataset_root": str(dataset_root),
        "species_folder_name": species_folder_name,
        "species_dir": str(species_dir),
        "species_images_dir": str(species_images_dir),
        "species_labels_dir": str(species_labels_dir),
        "local_image_roots": [str(p) for p in local_image_roots],
        "summary": summary,
        "copied_pairs": [
            {
                "source_stem": p.stem,
                "source_image_path": str(p.image_path),
                "source_label_path": str(p.label_path),
                "dest_stem": p.dest_stem,
                "dest_image_path": str(p.dest_image_path),
                "dest_label_path": str(p.dest_label_path),
                "collision_strategy": p.collision_strategy,
            }
            for p in plans
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _delete_path(path: Optional[Path], *, dry_run: bool) -> None:
    if path is None or not path.exists():
        return
    if dry_run:
        print(f"[dry-run] delete {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest YOLO image/label pairs into the species-grouped day-v3 daytime dataset."
    )
    p.add_argument(
        "source",
        type=Path,
        help="Path to a local extracted folder or zip containing YOLO labels and optionally images.",
    )
    p.add_argument(
        "--species-tag",
        type=str,
        default=DEFAULT_SPECIES_TAG,
        help=f"Species label for manifesting and collision suffixes (default: {DEFAULT_SPECIES_TAG!r}).",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Species-grouped YOLO dataset root (default: {DEFAULT_DATASET_ROOT}).",
    )
    p.add_argument(
        "--local-image-root",
        action="append",
        type=Path,
        default=[],
        help=(
            "Optional additional root to search for matching images when the source "
            "provides labels only. Repeatable."
        ),
    )
    p.add_argument(
        "--extract-root",
        type=Path,
        default=DEFAULT_EXTRACT_ROOT,
        help=f"Where to temporarily extract zips (default: {DEFAULT_EXTRACT_ROOT}).",
    )
    p.add_argument(
        "--manifest-root",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help=f"Where to write ingest manifests (default: {DEFAULT_MANIFEST_ROOT}).",
    )
    p.add_argument(
        "--delete-zip",
        action="store_true",
        help="Delete the source zip after a successful non-dry-run ingest.",
    )
    p.add_argument(
        "--delete-extracted",
        action="store_true",
        help="Delete the extracted source directory after a successful non-dry-run ingest.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the copy plan without writing dataset files.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    extract_root_base = args.extract_root.expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    species_folder_name, species_dir, species_images_dir, species_labels_dir, species_created = _ensure_species_dataset_dirs(
        dataset_root,
        str(args.species_tag),
        dry_run=bool(args.dry_run),
    )

    local_image_roots = [Path(p).expanduser().resolve() for p in args.local_image_root]
    source_input = args.source.expanduser().resolve()
    source_root, extracted_root, zip_path = _resolve_source_root(source_input, extract_root_base)

    generated_labels_root: Optional[Path] = None
    label_files: List[Path] = []
    label_prep_summary: Dict[str, object] = {}
    if extracted_root is not None:
        generated_labels_root = extracted_root / "__generated_yolo_labels"
    else:
        generated_labels_root = extract_root_base / "__generated_yolo_labels" / f"{source_input.stem}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    label_files, label_prep_summary = _prepare_label_files(
        source_root=source_root,
        generated_root=generated_labels_root,
    )

    source_slug = _safe_slug(args.species_tag)
    plans, summary = _plan_pairs(
        label_files=label_files,
        source_root=source_root,
        extracted_root=extracted_root,
        local_image_roots=local_image_roots,
        train_images_dir=species_images_dir,
        train_labels_dir=species_labels_dir,
        source_slug=source_slug,
    )
    summary = dict(summary)
    summary.update(label_prep_summary)

    print(f"[config] source_input={source_input}")
    print(f"[config] resolved_source_root={source_root}")
    print(f"[config] dataset_root={dataset_root}")
    print(f"[config] species_folder={species_folder_name} created={species_created}")
    print(f"[config] species_images_dir={species_images_dir}")
    print(f"[config] species_labels_dir={species_labels_dir}")
    print(f"[config] local_image_roots={[str(p) for p in local_image_roots]}")
    print(
        "[summary] "
        f"label_files_found={summary['label_files_found']} "
        f"pairs_to_copy={summary['pairs_to_copy']} "
        f"skipped_existing_pairs={summary['skipped_existing_pairs']} "
        f"renamed_due_to_collision={summary['renamed_due_to_collision']}"
    )
    print(
        "[labels] "
        f"source_format={summary.get('label_source_format')} "
        f"coco_json_files={summary.get('coco_json_files', 0)} "
        f"generated_label_files={summary.get('generated_label_files', 0)}"
    )

    _copy_pairs(plans, dry_run=bool(args.dry_run))
    manifest_path = _write_manifest(
        manifest_root=args.manifest_root.expanduser().resolve(),
        species_tag=str(args.species_tag),
        source_input=source_input,
        source_root=source_root,
        dataset_root=dataset_root,
        species_folder_name=species_folder_name,
        species_dir=species_dir,
        species_images_dir=species_images_dir,
        species_labels_dir=species_labels_dir,
        local_image_roots=local_image_roots,
        plans=plans,
        summary=summary,
        extracted_root=extracted_root,
        zip_path=zip_path,
        dry_run=bool(args.dry_run),
    )
    print(f"[done] manifest={manifest_path}")
    dataset_manifest_path = _write_dataset_manifest(dataset_root, dry_run=bool(args.dry_run))
    print(f"[done] dataset_manifest={dataset_manifest_path}")

    if generated_labels_root is not None and generated_labels_root.exists():
        _delete_path(generated_labels_root, dry_run=bool(args.dry_run))
        if bool(args.dry_run):
            print(f"[cleanup] would delete generated labels dir: {generated_labels_root}")
        else:
            print(f"[cleanup] deleted generated labels dir: {generated_labels_root}")

    if not args.dry_run:
        if args.delete_extracted:
            _delete_path(extracted_root, dry_run=False)
            print(f"[cleanup] deleted extracted dir: {extracted_root}")
        if args.delete_zip:
            _delete_path(zip_path, dry_run=False)
            print(f"[cleanup] deleted zip: {zip_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
