#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


ID_TO_META_RAW: dict[str, str] = {
    "s1806ik": "Photinus Knulli s1806ik 2021/08/06 peleg-group-1/Photinus knulli 2021/20210806/Max A/A1/4k",
    "s2607ia": "Photinus acuminatus s2607ia 2022/06/07 peleg-group-1/Fireflies Citizen Science/2022/xerces/20220607/gp1/4k",
    "s2607ic": "Photinus carolinus s2607ic 2022/06/07 peleg-group-1/Fireflies Citizen Science/2022/tnc/20220607 OH Pc -- done/TNC Ohio GOPro 220607 Camera 1/4k",
    "s2617bw": "Bicellonycha wickershamorum s2617bw 2022/06/17 peleg-group-1/Fireflies Citizen Science/2022/tnc/20220617 MSR MS N -- done/MS Ranch 220617 Camera 1 NE/4k",
    "s2618bw": "Bicellonycha wickershamorum s2618bw 2022/06/18 peleg-group-1/Fireflies Citizen Science/2022/tnc/20220618 MSR MS S -- done/MS Ranch 220618 Camera 1 SW/4k",
    "s2705ub": "Photuris bethaniensis s2705ub 2022/07/05 peleg-group-1/Fireflies Citizen  Science/2022/dnrec/20220705/DFWJuly52022Swale402/4k",
}


_ID_RE = re.compile(r"s\d{4}[a-z]{2}", re.IGNORECASE)


def sanitize_prefix(raw: str) -> str:
    s = raw.strip()
    s = s.replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


ID_TO_PREFIX = {k.lower(): sanitize_prefix(v) for k, v in ID_TO_META_RAW.items()}


def extract_id(name: str) -> str | None:
    m = _ID_RE.search(name)
    return m.group(0).lower() if m else None


@dataclass(frozen=True)
class Pair:
    mp4: Path
    csv: Path

    @property
    def stem(self) -> str:
        return self.mp4.stem


@dataclass
class SpeciesStats:
    name: str
    pairs: int = 0
    moved_pairs: int = 0
    orphan_mp4: int = 0
    acted_orphan_mp4: int = 0
    orphan_csv: int = 0
    ambiguous: int = 0
    missing_id: int = 0
    missing_meta: int = 0
    collisions: int = 0


def collect_pairs(species_dir: Path, skip_dirnames: set[str]) -> tuple[list[Pair], list[Path], list[Path], list[str]]:
    pairs: list[Pair] = []
    orphan_mp4: list[Path] = []
    orphan_csv: list[Path] = []
    ambiguous_msgs: list[str] = []

    for dirpath_str, dirnames, filenames in os.walk(species_dir):
        dirpath = Path(dirpath_str)
        dirnames[:] = [d for d in dirnames if d not in skip_dirnames]

        mp4_by_stem: dict[str, list[Path]] = {}
        csv_by_stem: dict[str, list[Path]] = {}

        for fn in filenames:
            p = dirpath / fn
            ext = p.suffix.lower()
            key = p.stem.lower()
            if ext == ".mp4":
                mp4_by_stem.setdefault(key, []).append(p)
            elif ext == ".csv":
                csv_by_stem.setdefault(key, []).append(p)

        for key, mp4_list in mp4_by_stem.items():
            csv_list = csv_by_stem.get(key)
            if not csv_list:
                orphan_mp4.extend(mp4_list)
                continue
            if len(mp4_list) != 1 or len(csv_list) != 1:
                ambiguous_msgs.append(
                    f"{species_dir.name}: ambiguous duplicates in {dirpath}: stem={key} mp4={len(mp4_list)} csv={len(csv_list)}"
                )
                continue
            pairs.append(Pair(mp4=mp4_list[0], csv=csv_list[0]))

        for key, csv_list in csv_by_stem.items():
            if key not in mp4_by_stem:
                orphan_csv.extend(csv_list)

    return pairs, orphan_mp4, orphan_csv, ambiguous_msgs


def ensure_unique_pair_dest(
    dest_dir: Path, base_stem: str, mp4_suffix: str, csv_suffix: str
) -> tuple[str, int]:
    stem = base_stem
    collisions = 0
    for i in range(0, 10000):
        if i > 0:
            stem = f"{base_stem}__{i}"
            collisions += 1
        mp4_dest = dest_dir / f"{stem}{mp4_suffix}"
        csv_dest = dest_dir / f"{stem}{csv_suffix}"
        if not mp4_dest.exists() and not csv_dest.exists():
            return stem, collisions
    raise RuntimeError(f"Unable to find collision-free destination for stem={base_stem} in {dest_dir}")


def move_file(src: Path, dest: Path, execute: bool) -> None:
    if not execute:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))


def prune_empty_dirs(species_dir: Path, keep_dirnames: set[str], execute: bool) -> int:
    if not execute:
        return 0
    removed = 0
    for dirpath_str, dirnames, filenames in os.walk(species_dir, topdown=False):
        dirpath = Path(dirpath_str)
        if dirpath == species_dir:
            continue
        if dirpath.name in keep_dirnames:
            continue
        if any(dirpath.iterdir()):
            continue
        dirpath.rmdir()
        removed += 1
    return removed


def process_species(
    species_dir: Path,
    execute: bool,
    orphan_action: str,
    prune_empty: bool,
    orphan_dirname: str,
    show_examples: int,
) -> tuple[SpeciesStats, list[str]]:
    stats = SpeciesStats(name=species_dir.name)
    logs: list[str] = []

    skip_dirnames = {orphan_dirname}
    pairs, orphan_mp4, orphan_csv, ambiguous = collect_pairs(species_dir, skip_dirnames=skip_dirnames)
    stats.pairs = len(pairs)
    stats.orphan_mp4 = len(orphan_mp4)
    stats.orphan_csv = len(orphan_csv)
    stats.ambiguous = len(ambiguous)
    logs.extend(ambiguous)

    # Handle orphan mp4s first so we don't move them later.
    if orphan_mp4:
        if orphan_action == "delete":
            for p in orphan_mp4:
                logs.append(f"[orphan mp4 delete] {p}")
                if execute:
                    p.unlink()
            stats.acted_orphan_mp4 = len(orphan_mp4)
        elif orphan_action == "quarantine":
            qdir = species_dir / orphan_dirname
            for p in orphan_mp4:
                dest_name = p.name
                dest = qdir / dest_name
                if dest.exists():
                    # avoid clobbering
                    base = p.stem
                    suf = p.suffix
                    for i in range(1, 10000):
                        dest = qdir / f"{base}__{i}{suf}"
                        if not dest.exists():
                            break
                logs.append(f"[orphan mp4 quarantine] {p} -> {dest}")
                move_file(p, dest, execute=execute)
            stats.acted_orphan_mp4 = len(orphan_mp4)
        else:
            raise ValueError(f"Unknown orphan_action={orphan_action}")

    # Now move/rename pairs into the species root.
    examples_shown = 0
    for pair in sorted(pairs, key=lambda x: str(x.mp4)):
        src_mp4 = pair.mp4
        src_csv = pair.csv
        file_id = extract_id(src_mp4.name) or extract_id(src_csv.name)
        if not file_id:
            stats.missing_id += 1
            prefix = None
        else:
            prefix = ID_TO_PREFIX.get(file_id)
            if prefix is None:
                stats.missing_meta += 1

        base_stem = pair.stem
        if prefix:
            base_stem = f"{prefix}_{base_stem}"

        unique_stem, collision_count = ensure_unique_pair_dest(
            species_dir, base_stem, mp4_suffix=src_mp4.suffix, csv_suffix=src_csv.suffix
        )
        stats.collisions += collision_count

        dest_mp4 = species_dir / f"{unique_stem}{src_mp4.suffix}"
        dest_csv = species_dir / f"{unique_stem}{src_csv.suffix}"

        if show_examples and examples_shown < show_examples:
            logs.append(f"[pair] {src_mp4} -> {dest_mp4}")
            logs.append(f"[pair] {src_csv} -> {dest_csv}")
            examples_shown += 1

        move_file(src_mp4, dest_mp4, execute=execute)
        move_file(src_csv, dest_csv, execute=execute)
        stats.moved_pairs += 1

    if prune_empty:
        removed_dirs = prune_empty_dirs(species_dir, keep_dirnames=skip_dirnames, execute=execute)
        if removed_dirs:
            logs.append(f"[prune] removed {removed_dirs} empty dirs under {species_dir}")

    return stats, logs


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reorganize integrated raw videos: move mp4/csv pairs into species folder root, delete/quarantine orphan mp4s, and rename using provided metadata."
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/Samsung_SSD_2TB/integrated prototype raw videos"),
        help="Root directory containing species subfolders.",
    )
    p.add_argument("--execute", action="store_true", help="Apply changes. Default is dry-run.")
    p.add_argument(
        "--orphan-action",
        choices=["delete", "quarantine"],
        default="quarantine",
        help="What to do with .mp4 files that have no matching .csv in the same folder.",
    )
    p.add_argument(
        "--orphan-dirname",
        default="__orphan_mp4_no_csv",
        help="Quarantine folder name (only used when --orphan-action=quarantine).",
    )
    p.add_argument("--prune-empty", action="store_true", help="Remove empty directories after moving.")
    p.add_argument(
        "--species",
        action="append",
        default=[],
        help="Limit processing to specific species folder names (repeatable).",
    )
    p.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Log up to N example pair moves per species (0 to disable).",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root: Path = args.root
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 2

    wanted_species = set(args.species)
    species_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (not wanted_species or p.name in wanted_species)]
    if not species_dirs:
        print("No species folders found to process.", file=sys.stderr)
        return 2

    totals = SpeciesStats(name="TOTAL")
    all_logs: list[str] = []
    for sp_dir in species_dirs:
        stats, logs = process_species(
            sp_dir,
            execute=args.execute,
            orphan_action=args.orphan_action,
            prune_empty=args.prune_empty,
            orphan_dirname=args.orphan_dirname,
            show_examples=args.show_examples,
        )
        all_logs.extend(logs)

        totals.pairs += stats.pairs
        totals.moved_pairs += stats.moved_pairs
        totals.orphan_mp4 += stats.orphan_mp4
        totals.acted_orphan_mp4 += stats.acted_orphan_mp4
        totals.orphan_csv += stats.orphan_csv
        totals.ambiguous += stats.ambiguous
        totals.missing_id += stats.missing_id
        totals.missing_meta += stats.missing_meta
        totals.collisions += stats.collisions

        print(
            f"{stats.name}: pairs={stats.pairs} moved_pairs={stats.moved_pairs} orphan_mp4={stats.orphan_mp4} orphan_csv={stats.orphan_csv} ambiguous={stats.ambiguous} missing_id={stats.missing_id} missing_meta={stats.missing_meta} collisions={stats.collisions}"
        )

    if all_logs:
        print("\n--- Logs (sampled) ---")
        for line in all_logs:
            print(line)

    print(
        f"\nTOTAL: pairs={totals.pairs} moved_pairs={totals.moved_pairs} orphan_mp4={totals.orphan_mp4} acted_orphan_mp4={totals.acted_orphan_mp4} orphan_csv={totals.orphan_csv} ambiguous={totals.ambiguous} missing_id={totals.missing_id} missing_meta={totals.missing_meta} collisions={totals.collisions}"
    )

    if totals.ambiguous:
        print("\nERROR: ambiguous duplicates were found; refusing to continue safely.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

