#!/usr/bin/env python
"""
Great Plan 4, Step 4.3 -- clean rescore of cells that the 2026-08-27 contamination decision did not
cover: the nine later cells, and the five Plan 4 arms as they land.

METHOD (copied from logs_and_models/contamination_decision/raw/scripts/rescore.py, not run in place)
    The pooled out-of-fold confusion matrix is a sum over tiles, so
        CM_clean = CM_all - sum over the 1,281 contaminated tiles of CM_tile
    CM_all is the persisted `global_confusion_matrix`; only the contaminated tiles are re-read.
    Same contaminated-tile list, same scorer (`per_category_metrics.metrics_from_confusion`), same
    three guards:
      1. metrics recomputed from the STORED CM_all must reproduce the published macro_iou exactly
      2. no entry of CM_clean may be negative
      3. --verify_cell recomputes one cell's CM_all from disk tile by tile and compares element-wise

WHY A COPY
    rescore.py writes task2_rescore_table.csv and task2_summary.json inside logs_and_models, so
    running it again would overwrite the decision folder's record. This copy writes ONLY under
    exploratory_data_analysis/results/tables/plan4/clean_rescore/, and reads everything else.

OUTPUTS
    rescore_<cell>.json           per cell: all-tiles, clean and contaminated-only metrics + matrices
    clean_rescore_table.csv       every cell scored HERE, rebuilt from the per-cell JSONs each run
    clean_scores_all_cells.csv    the 28-row 2026-08-27 table (read-only) + the rows scored here,
                                  one clean Macro-IoU per cell for every scored cell

    python plan4_clean_rescore.py                         # the nine unrescored cells
    python plan4_clean_rescore.py --cells convnext_upernet_cir
    python plan4_clean_rescore.py --verify_cell convnext_upernet_ortorgb
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

from ML_sdfi_fastai2.analyse import per_category_metrics as pcm  # noqa: E402  (the frozen scorer)

DECISION = C.LOGS_AND_MODELS / "contamination_decision"
CONTAM = DECISION / "raw" / "contaminated_tiles.txt"
ORIG_TABLE = DECISION / "task2_rescore_table.csv"
OUT = C.TABLES / "plan4" / "clean_rescore"

UNRESCORED = [
    "convnext_upernet_ortorgb", "segformer_b1_ortorgb", "swin_upernet_ortorgb",
    "unet_resnet34_ortorgb", "convnext_upernet_rgb_dsm_dtm_corrected",
    "segformer_b1_rgb_dsm_dtm_corrected", "swin_upernet_rgb_dsm_dtm_corrected",
    "unet_resnet34_rgb_dsm_dtm_corrected", "segformer_b1_6ch_corrected",
]
PREDICTED = ["asfalt", "fliser", "grus", "ubefestet", "green_roof",
             "drivhus", "betonflade", "brosten", "solceller"]


def log(msg=""):
    print(msg, flush=True)


def cell_paths(cell):
    """(pooled json, [pred dir fold 0..2]) from the untouched matrix tree."""
    hits = list(C.SPATIAL_MATRIX.glob(f"*/oof_{cell}/pooled_oof_metrics.json"))
    if len(hits) != 1:
        return None, None
    mdir = hits[0].parent.parent
    base, suf = (cell[:-4], "_unw") if cell.endswith("_unw") else (cell, "")
    return hits[0], [mdir / f"{base}_fold{k}{suf}" / "models" / "example_dataset" for k in range(3)]


def folds_from_split():
    r2f = C.load_fold_assignment(str(C.FOLD_ASSIGNMENT_CSV))
    folds = [[] for _ in range(C.NFOLDS)]
    for t in C.tile_names():
        folds[r2f[C.parse_route(t)]].append(t)
    return folds


def cm_for(tiles, pred_folder, label_cache, n):
    """Pixel-summed CM over `tiles`; labels from the cache. Same arithmetic as rescore.cm_for."""
    cm = np.zeros(n * n, dtype=np.int64)
    for t in tiles:
        pred = np.asarray(Image.open(pred_folder / t), dtype=np.int64)
        label = label_cache[t]
        if label.shape != pred.shape:
            sys.exit(f"shape mismatch {t}: {label.shape} vs {pred.shape}")
        if pred.max() >= n:
            sys.exit(f"pred class index out of range in {t}: {pred.max()}")
        cm += np.bincount(label.ravel() * n + pred.ravel(), minlength=n * n)
    return cm.reshape(n, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(UNRESCORED))
    ap.add_argument("--verify_cell", default="")
    args = ap.parse_args()

    out_dir = C.assert_writes_are_local(OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = C.load_codes(str(C.CODES_TXT))
    n = len(codes)
    folds = folds_from_split()
    contaminated = {l.strip() for l in open(CONTAM) if l.strip()}
    contam_by_fold = [[t for t in folds[k] if t in contaminated] for k in range(C.NFOLDS)]
    assert sum(len(x) for x in contam_by_fold) == len(contaminated) == 1281, \
        f"contaminated tiles {len(contaminated)}, expected 1281"
    log(f"folds {[len(f) for f in folds]}   contaminated {len(contaminated):,} "
        f"per fold {[len(x) for x in contam_by_fold]}")

    t0 = time.time()
    label_cache = {t: np.asarray(Image.open(C.LABEL_DIR / t), dtype=np.int64)
                   for k in range(C.NFOLDS) for t in contam_by_fold[k]}
    log(f"cached {len(label_cache):,} contaminated label rasters in {time.time() - t0:.1f}s\n")

    incons = []
    for cell in [c.strip() for c in args.cells.split(",") if c.strip()]:
        jpath, preds = cell_paths(cell)
        if jpath is None:
            incons.append(f"{cell}: no unique pooled_oof_metrics.json")
            log(f"  SKIP {cell}: not scored yet")
            continue
        if any(not p.is_dir() for p in preds):
            incons.append(f"{cell}: prediction folder missing")
            log(f"  SKIP {cell}: prediction folder missing")
            continue
        stored = json.load(open(jpath, encoding="utf8"))
        cm_all = np.array(stored["global_confusion_matrix"], dtype=np.int64)

        m_all = pcm.metrics_from_confusion(cm_all, codes, split_tag="pooled_oof_all")
        pub = stored["headline_pooled_oof"]["macro_iou"]
        if abs(m_all["macro_iou"] - pub) > 1e-12:                                  # guard 1
            incons.append(f"{cell}: recomputed macro_iou {m_all['macro_iou']!r} != stored {pub!r}")

        t1 = time.time()
        cm_contam = sum(cm_for(contam_by_fold[k], preds[k], label_cache, n) for k in range(C.NFOLDS))
        cm_clean = cm_all - cm_contam
        if (cm_clean < 0).any():                                                    # guard 2
            incons.append(f"{cell}: CM_clean has {int((cm_clean < 0).sum())} negative entries; "
                          f"stored CM does not match the predictions on disk. EXCLUDED.")
            log(f"  {cell}: FAILED the negative-entry guard, excluded")
            continue

        m_clean = pcm.metrics_from_confusion(cm_clean, codes, split_tag="pooled_oof_clean")
        m_contam = pcm.metrics_from_confusion(cm_contam, codes, split_tag="contaminated_only")
        rec = {"cell": cell, "pooled_json": str(jpath), "pred_folders": [str(p) for p in preds],
               "published_macro_iou": pub, "all_tiles": m_all, "clean_tiles": m_clean,
               "contaminated_only": m_contam, "cm_contaminated": cm_contam.tolist(),
               "cm_clean": cm_clean.tolist(), "method": "subtraction, copied from rescore.py"}
        (out_dir / f"rescore_{cell}.json").write_text(json.dumps(rec, indent=2), encoding="utf8")
        log(f"  {cell:<40} mIoU {m_all['macro_iou']:.4f} -> clean {m_clean['macro_iou']:.4f} "
            f"(d={m_clean['macro_iou'] - m_all['macro_iou']:+.4f})  {time.time() - t1:.0f}s")

    if args.verify_cell:                                                            # guard 3
        jpath, preds = cell_paths(args.verify_cell)
        log(f"\nVERIFY {args.verify_cell}: recomputing the full pooled CM from disk ...")
        t1 = time.time()
        cm_re = sum(pcm.pooled_confusion_matrix(folds[k], str(preds[k]), str(C.LABEL_DIR),
                                                n_classes=n) for k in range(C.NFOLDS))
        cm_st = np.array(json.load(open(jpath))["global_confusion_matrix"], dtype=np.int64)
        same = bool(np.array_equal(cm_re, cm_st))
        log(f"  identical to stored: {same}  (max |diff| {int(np.abs(cm_re - cm_st).max())}, "
            f"{time.time() - t1:.0f}s)")
        if not same:
            incons.append(f"{args.verify_cell}: freshly recomputed CM differs from stored")
        (out_dir / f"verify_{args.verify_cell}.json").write_text(json.dumps(
            {"cell": args.verify_cell, "identical_to_stored": same,
             "max_abs_diff": int(np.abs(cm_re - cm_st).max())}, indent=2), encoding="utf8")

    # Rebuild the aggregate tables from every per-cell JSON present, so incremental runs stay whole.
    rows = []
    for p in sorted(out_dir.glob("rescore_*.json")):
        r = json.load(open(p, encoding="utf8"))
        a, c = r["all_tiles"], r["clean_tiles"]
        row = {"cell": r["cell"], "macro_iou_all": a["macro_iou"], "macro_iou_clean": c["macro_iou"],
               "d_macro_iou": c["macro_iou"] - a["macro_iou"],
               "macro_f1_all": a["macro_f1"], "macro_f1_clean": c["macro_f1"],
               "acc_all": a["overall_accuracy"], "acc_clean": c["overall_accuracy"],
               "px_all": a["evaluated_pixels"], "px_clean": c["evaluated_pixels"],
               "n_macro_clean": c["n_macro_classes_evaluated"],
               "excluded_clean": ",".join(c["excluded_absent"]),
               "published_macro_iou": r["published_macro_iou"]}
        for cls in PREDICTED:
            row[f"iou_all_{cls}"] = a["per_class"][cls]["iou"]
            row[f"iou_clean_{cls}"] = c["per_class"][cls]["iou"]
        rows.append(row)
    if rows:
        with open(out_dir / "clean_rescore_table.csv", "w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    combined = {}
    with open(ORIG_TABLE, newline="", encoding="utf8") as fh:                       # read-only
        for r in csv.DictReader(fh):
            combined[r["cell"]] = (float(r["macro_iou_all"]), float(r["macro_iou_clean"]),
                                   "contamination_decision/task2_rescore_table.csv")
    for r in rows:
        if r["cell"] in combined:
            incons.append(f"{r['cell']}: present in BOTH the 2026-08-27 table and here")
        combined[r["cell"]] = (r["macro_iou_all"], r["macro_iou_clean"],
                               "plan4/clean_rescore/clean_rescore_table.csv")
    with open(out_dir / "clean_scores_all_cells.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "macro_iou_all", "macro_iou_clean", "source"])
        for cell in sorted(combined):
            w.writerow([cell, *combined[cell]])

    log(f"\nper-cell JSONs here: {len(rows)}   combined clean table: {len(combined)} cells")
    for s in incons:
        log(f"  INCONSISTENCY {s}")
    sys.exit(1 if incons else 0)


if __name__ == "__main__":
    main()
