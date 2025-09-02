#!/usr/bin/env python3
from pathlib import Path
import csv

def rebuild_fireflies_logits_from_main(main_csv_path: Path):
    """
    Rebuild <stem>_fireflies_logits.csv from the CURRENT MAIN CSV rows.
    Keeps only class=='firefly'. Writes columns: t,x,y,background_logit,firefly_logit.
    main_csv_path: Path to your main pipeline CSV (the one Stage 7/8.6/8.7 keep deduped).
    """
    main_csv_path = Path(main_csv_path)
    logits_path = main_csv_path.with_name(main_csv_path.stem + "_fireflies_logits.csv")

    with main_csv_path.open("r", newline="") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
        cols = rd.fieldnames or []

    has_class = ("class" in cols)
    out_cols = ["t", "x", "y", "background_logit", "firefly_logit"]

    # Build unique (t,x,y) from MAIN (already deduped upstream)
    unique = {}
    for r in rows:
        try:
            if has_class and (r.get("class", "firefly").strip().lower() != "firefly"):
                continue
            t = int(r.get("t", r.get("frame")))
            x = int(round(float(r["x"])))
            y = int(round(float(r["y"])))
            bg = str(r.get("background_logit", ""))
            ff = str(r.get("firefly_logit", ""))
            unique[(t, x, y)] = (bg, ff)
        except Exception:
            continue

    with logits_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        for (t, x, y), (bg, ff) in unique.items():
            w.writerow({
                "t": t,
                "x": x,
                "y": y,
                "background_logit": bg,
                "firefly_logit": ff
            })

    print(f"[sync] Rewrote fireflies logits from MAIN: {logits_path}")
