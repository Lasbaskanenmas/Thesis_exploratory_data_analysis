#!/usr/bin/env python
"""
Assemble the per-fold Macro-IoU of every scored cell into one small CSV. LOCATES, never computes:
every number is lifted from the cell's `pooled_oof_metrics.json` (`per_fold_diagnostic`), the same
file `extend_cross_cell_summary.py` reads to fill `fold_macro_iou_mean` / `fold_macro_iou_std`.

Promoted from the 2026-09-13 session scratch so it can be re-run as cells are added. Checks, across
the whole table rather than three hand-picked cells:
  * `per_fold_diagnostic` agrees with `fold_aggregate_diagnostic.macro_iou.per_fold` exactly
  * mean and ddof=1 SD of the three folds reproduce the summary's columns to 4 decimals
    (ddof=1 is what `per_category_metrics.aggregate_folds` uses)

Output: results/tables/per_fold_macro_iou.csv  (cell, fold, macro_iou, macro_f1, overall_accuracy,
source_file), plus a copy in C:\\thesis\\Results\\tables_new\\ when that folder exists.

    python assemble_per_fold_macro_iou.py            # refuses to overwrite
    python assemble_per_fold_macro_iou.py --force
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

SUMMARY = C.TABLES / "cross_cell_summary.csv"
OUT = C.TABLES / "per_fold_macro_iou.csv"
MAC = Path(r"C:\thesis\Results\tables_new")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    summary = {r["cell"]: r for r in csv.DictReader(open(SUMMARY, newline=""))}
    rows, short, missing, disagree, bad_mean, bad_sd = [], [], [], [], [], []
    for cell in summary:
        hits = list(C.SPATIAL_MATRIX.glob(f"*/oof_{cell}/pooled_oof_metrics.json"))
        if len(hits) != 1:
            missing.append(cell)
            continue
        j = json.load(open(hits[0]))
        diags = {d["split"]: d for d in j.get("per_fold_diagnostic", [])}
        vals = []
        for f in range(C.NFOLDS):
            d = diags.get(f"fold_{f}")
            if d is None or d.get("macro_iou") is None:
                continue
            vals.append(d["macro_iou"])
            rows.append({"cell": cell, "fold": f, "macro_iou": d["macro_iou"],
                         "macro_f1": d.get("macro_f1"),
                         "overall_accuracy": d.get("overall_accuracy"), "source_file": str(hits[0])})
        if len(vals) != C.NFOLDS:
            short.append((cell, len(vals)))
            continue
        agg = j["fold_aggregate_diagnostic"]["macro_iou"]["per_fold"]
        if any(a is not None and abs(a - b) > 1e-12 for a, b in zip(agg, vals)):
            disagree.append(cell)
        v = np.array(vals)
        if round(float(v.mean()), 4) != round(float(summary[cell]["fold_macro_iou_mean"]), 4):
            bad_mean.append(cell)
        if round(float(v.std(ddof=1)), 4) != round(float(summary[cell]["fold_macro_iou_std"]), 4):
            bad_sd.append(cell)

    print(f"cells in summary {len(summary)}   rows {len(rows)}")
    print(f"missing pooled JSON: {missing or 'none'}   fewer than 3 folds: {short or 'none'}")
    print(f"per_fold_diagnostic vs aggregate disagreement: {disagree or 'none'}")
    print(f"mean mismatch: {bad_mean or 'none'}   ddof=1 SD mismatch: {bad_sd or 'none'}")
    if missing or short or disagree or bad_mean or bad_sd:
        sys.exit("checks failed -- nothing written")

    if OUT.exists() and not args.force:
        sys.exit(f"{OUT} exists; pass --force to regenerate it")
    rows.sort(key=lambda r: (r["cell"], r["fold"]))
    out = C.assert_writes_are_local(OUT)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    if MAC.is_dir():
        shutil.copy2(out, MAC / out.name)
        print(f"copied to {MAC / out.name}")


if __name__ == "__main__":
    main()
