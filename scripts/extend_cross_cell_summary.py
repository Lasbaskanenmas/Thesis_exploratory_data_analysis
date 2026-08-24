#!/usr/bin/env python
"""
Append the 2026-08 arm cells to cross_cell_summary.csv, sourced from their pooled_oof_metrics.json.

ADDITIVE. Existing rows are copied through byte-for-byte; only new rows are appended, and a cell
already present is skipped rather than rewritten. The header is never changed.

Schema note: the `route_*` columns come from the per-route confusion matrices (module E6), which have
only been built for `convnext_upernet_rgb`. They are left EMPTY for the new rows rather than
invented -- exactly as `iou_unknown` and `iou_unknown2` are already empty, those being the ignore
index and the report-only class. Everything else comes straight from the pooled JSON.

    python extend_cross_cell_summary.py            # dry run, prints what it would add
    python extend_cross_cell_summary.py --write
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

TARGET = C.TABLES / "cross_cell_summary.csv"

def discover_cells():
    """Every scored cell in the matrix tree: (model_dir, cell), sorted.

    Discovered rather than hard-coded so the 24 frozen cells and the 2026-08 arms are handled by
    the same path. A cell counts as scored only if its pooled_oof_metrics.json exists.
    """
    out = []
    for p in sorted(C.SPATIAL_MATRIX.glob("*/oof_*")):
        if (p / "pooled_oof_metrics.json").is_file():
            out.append((p.parent.name, p.name[len("oof_"):]))
    return sorted(out, key=lambda t: t[1])


NEW_CELLS = discover_cells()

CLASS_ORDER = ["unknown", "asfalt", "fliser", "grus", "ubefestet", "green_roof",
               "drivhus", "betonflade", "brosten", "unknown2", "solceller"]
ROUTE_COLS = ["route_median_macro_iou", "route_min", "route_max", "route_spread",
              "route_std", "n_routes_scored"]


def row_for(model_dir, cell):
    p = C.SPATIAL_MATRIX / model_dir / f"oof_{cell}" / "pooled_oof_metrics.json"
    if not p.is_file():
        sys.exit(f"missing pooled score for {cell}: {p}")
    j = json.load(open(p))
    h = j["headline_pooled_oof"]
    agg = j["fold_aggregate_diagnostic"]["macro_iou"]

    row = {
        "cell": cell,
        "macro_iou": h["macro_iou"],
        "macro_f1": h["macro_f1"],
        "overall_accuracy": h["overall_accuracy"],
        "evaluated_pixels": h["evaluated_pixels"],
        "fold_macro_iou_mean": agg["mean"],
        "fold_macro_iou_std": agg["std"],
    }
    for cls in CLASS_ORDER:
        rec = h["per_class"].get(cls)
        row[f"iou_{cls}"] = "" if (rec is None or rec.get("iou") is None) else rec["iou"]
    for col in ROUTE_COLS:
        row[col] = ""            # E6 not run for this cell; not invented
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    if not TARGET.is_file():
        sys.exit(f"missing {TARGET}")
    with open(TARGET, newline="", encoding="utf8") as fh:
        rdr = csv.DictReader(fh)
        header = rdr.fieldnames
        existing = list(rdr)
    have = {r["cell"] for r in existing}
    print(f"existing rows : {len(existing)}  ({', '.join(sorted(have))})")

    add = []
    for model_dir, cell in NEW_CELLS:
        if cell in have:
            print(f"  skip (already present, not rewritten): {cell}")
            continue
        add.append(row_for(model_dir, cell))

    print(f"\nrows to append: {len(add)}")
    print(f"  {'cell':<36}{'macro_iou':>11}{'macro_f1':>11}{'overall_acc':>13}{'fold mean':>11}{'fold std':>10}")
    for r in add:
        print(f"  {r['cell']:<36}{r['macro_iou']:>11.4f}{r['macro_f1']:>11.4f}"
              f"{r['overall_accuracy']:>13.4f}{r['fold_macro_iou_mean']:>11.4f}"
              f"{r['fold_macro_iou_std']:>10.4f}")

    missing_cols = [k for k in add[0] if k not in header] if add else []
    if missing_cols:
        sys.exit(f"schema mismatch, refusing to write: {missing_cols}")
    print(f"\nschema check : all {len(add[0]) if add else 0} produced fields exist in the header  OK")

    if not args.write:
        print("\ndry run -- pass --write to append")
        return
    if not add:
        print("\nnothing to append")
        return

    out = C.assert_writes_are_local(TARGET)
    with open(out, "a", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        for r in add:
            w.writerow(r)
    with open(out, newline="", encoding="utf8") as fh:
        final = list(csv.DictReader(fh))
    print(f"\nappended {len(add)} rows -> {out}")
    print(f"file now has {len(final)} data rows: {', '.join(r['cell'] for r in final)}")


if __name__ == "__main__":
    main()
