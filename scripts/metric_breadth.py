#!/usr/bin/env python
"""
Task 2.5 of the 2026-08-24 work order / Great Plan 3.1 section 5.4: metric breadth over every
scored cell, from the pooled out-of-fold confusion matrices already on disk.

NO new inference and NO new metric code path. The per-class numbers come from
`per_category_metrics.metrics_from_confusion` -- the same self-tested function `pooled_oof_metrics.py`
itself calls -- run over each cell's stored `global_confusion_matrix`. This module only adds
aggregates on top of that output. The recomputed macro-IoU, macro-F1, overall accuracy and evaluated
pixels are asserted against the values already written into each JSON; if an assertion fires, this
script is wrong and the frozen numbers stand.

Reimplementing the scoring here would have been a mistake and nearly was: the convention is that the
ignore-index LABEL ROW is zeroed while the ignore-index PREDICTION COLUMN is kept, so predicting
`unknown` on a real-class pixel is a false negative for that class and is counted. Exactly one of
the 27 cells does this -- `unet_resnet34_10ch_unw`, on 14,671 pixels of 19.3 billion -- which is
few enough that a private reimplementation dropping that column would have passed unnoticed on 26
cells and been silently wrong on the 27th.

WHAT IT PRODUCES, per cell
    per-class IoU, precision, recall, F1 and support            (the over- vs under-prediction story)
    macro-IoU, macro-F1, macro precision/recall                 (recomputed, cross-checked)
    frequency-weighted IoU                                      (Long et al. 2015 convention:
                                                                 sum_c freq_c * IoU_c over present
                                                                 classes, freq from evaluated px)
    overall accuracy                                            (context)
    worst-case metrics                                          (see the caveat below)
    a binary befaestet / ubefaestet collapse                    (bridge to the service's headline)

THE WORST-CASE METRIC, AND AN HONEST CAVEAT
    The thesis cites Wang Z. et al. 2023 (NeurIPS) for "per-category mIoU PLUS worst-case metrics"
    and currently implements only the first half (tracker section 8.4, gap G3.3). The operational
    definition used here is the one this project's own gap analysis records for that gap: the
    minimum single-class IoU, and the mean of the k worst classes. Both are emitted, for k = 1, 2, 3,
    with the responsible class names beside them.

    This has NOT been checked against the paper's own formulation, because the PDF is not on this
    machine. The reference-canon day (Plan 3.1 section 5.5) must verify the exact definition before
    any prose describes this metric as Wang's. If the paper defines it differently, the columns are
    still correct as "min-class IoU" and "mean of the k worst" and the citation is what changes.

THE BINARY COLLAPSE, AND WHAT IT ASSUMES
    sealed   = asfalt, fliser, grus, betonflade, brosten, solceller, drivhus, green_roof
    unsealed = ubefestet
    dropped  = unknown (index 0, the ignore index) and unknown2 (index 9, report-only, 0 px pooled)
    This mapping is the work order's, and it puts solar panels, greenhouses and green roofs on the
    sealed side. That is a defensible reading for an imperviousness proxy but it is a JUDGEMENT, not
    a measurement -- FLAGGED FOR THE AUTHOR'S CONFIRMATION before it reaches the thesis. It exists to
    bridge the 9-class results to the service's roughly 95 percent binary headline and the 0.8747
    trivial floor.

Read-only over logs_and_models/. Writes only under exploratory_data_analysis/.

    python metric_breadth.py [--selftest]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

sys.path.insert(0, r"c:\thesis\ML_sdfi_fastai2\src\ML_sdfi_fastai2")
import analyse.per_category_metrics as pcm  # noqa: E402  (the project's own scored-metric function)

CLASS_ORDER = ["unknown", "asfalt", "fliser", "grus", "ubefestet", "green_roof",
               "drivhus", "betonflade", "brosten", "unknown2", "solceller"]
IGNORE_INDEX = pcm.DEFAULT_IGNORE_INDEX        # 0, unknown
REPORT_ONLY = pcm.DEFAULT_REPORT_ONLY          # (9,), unknown2 -- outside the macro average
SEALED = ["asfalt", "fliser", "grus", "betonflade", "brosten", "solceller", "drivhus", "green_roof"]
UNSEALED = ["ubefestet"]
WORST_K = (1, 2, 3)
TOL = 1e-9


# ----------------------------------------------------------------------------------------------
# core: delegate the scored metrics, add breadth on top
# ----------------------------------------------------------------------------------------------
def metrics_from_cm(cm, names, ignore_index=IGNORE_INDEX, report_only=REPORT_ONLY):
    """cm[true, pred] -> the project's per-class metrics plus the breadth aggregates.

    The per-class block is `per_category_metrics.metrics_from_confusion` verbatim, so the ignore
    convention is the frozen one: the ignore LABEL ROW is zeroed, the ignore PREDICTION COLUMN is
    kept, and predicting the ignore class on a real pixel is a false negative for the true class.
    """
    cm = np.asarray(cm, dtype=np.int64)
    base = pcm.metrics_from_confusion(cm, names, ignore_index=ignore_index, report_only=report_only)

    work = cm.copy()
    work[ignore_index, :] = 0
    per_class = base["per_class"]
    for name, rec in per_class.items():
        rec["predicted_pixels"] = int(work[:, rec["index"]].sum())

    macro = [n_ for n_ in per_class
             if per_class[n_]["in_macro"] and per_class[n_]["present"]]

    def _mean(field):
        vals = [per_class[n_][field] for n_ in macro if per_class[n_][field] is not None]
        return float(np.mean(vals)) if vals else None

    # Frequency-weighted IoU (Long et al. 2015 convention), normalised over the SCORED macro
    # classes so it is comparable with Macro-IoU computed on the same set.
    support_total = sum(per_class[n_]["support_pixels"] for n_ in macro)
    freq_weighted = (sum((per_class[n_]["support_pixels"] / support_total) * per_class[n_]["iou"]
                         for n_ in macro if per_class[n_]["iou"] is not None)
                     if support_total else None)

    ious = {n_: per_class[n_]["iou"] for n_ in macro}
    ranked = sorted(macro, key=lambda n_: (1.0 if ious[n_] is None else ious[n_], n_))
    worst = {}
    for k in WORST_K:
        sel = ranked[:k]
        worst[f"worst{k}_mean_iou"] = float(np.mean([ious[s] for s in sel]))
        worst[f"worst{k}_classes"] = "|".join(sel)

    base.update({
        "n_macro_classes": len(macro),
        "macro_precision": _mean("precision"),
        "macro_recall": _mean("recall"),
        "freq_weighted_iou": float(freq_weighted) if freq_weighted is not None else None,
        "macro_support_pixels": int(support_total),
        **worst,
    })
    return base


def binary_collapse(cm, names, sealed=SEALED, unsealed=UNSEALED,
                    ignore_index=IGNORE_INDEX, report_only=REPORT_ONLY):
    """Fold the 9-class matrix onto sealed / unsealed and score the 2x2 through the same function.

    The collapsed matrix keeps an `unknown` and an `unknown2` slot so the ignore convention is
    identical to the 11-class case: a pixel predicted `unknown` still counts as a false negative for
    whichever group it truly belongs to, rather than quietly leaving the denominator.
    """
    cm = np.asarray(cm, dtype=np.int64)
    idx = {nm: i for i, nm in enumerate(names)}
    groups = {"sealed": [idx[c] for c in sealed], "unsealed": [idx[c] for c in unsealed]}
    other = [i for i, nm in enumerate(names)
             if i == ignore_index or (i in report_only)]
    unassigned = [nm for i, nm in enumerate(names)
                  if nm not in sealed and nm not in unsealed and i not in other]
    assert not unassigned, f"classes in no group and not ignorable: {unassigned}"

    # 4x4: [unknown, sealed, unsealed, unknown2]
    col_groups = [[idx["unknown"]], groups["sealed"], groups["unsealed"], [idx["unknown2"]]]
    row_groups = list(col_groups)
    small = np.zeros((4, 4), dtype=np.int64)
    for a, gr in enumerate(row_groups):
        for b, gc in enumerate(col_groups):
            small[a, b] = cm[np.ix_(gr, gc)].sum()

    m = metrics_from_cm(small, ["unknown", "sealed", "unsealed", "unknown2"],
                        ignore_index=0, report_only=(3,))
    m["confusion_2x2_rows_true_cols_pred"] = small[1:3, 1:3].tolist()
    m["predicted_unknown_pixels"] = int(small[1:3, 0].sum())
    return m


# ----------------------------------------------------------------------------------------------
def selftest():
    print("=" * 88)
    print("metric_breadth selftest (synthetic, no files touched)")
    print("=" * 88)
    names = ["unknown", "a", "b", "c"]
    # true a: 8 right, 2 called b.  true b: 6 right, 4 called a.  true c: 0 right, 10 called a.
    # Row 0 is deliberately non-empty: ignored-label pixels must vanish from every denominator.
    cm = np.array([[0, 7, 7, 7],
                   [0, 8, 2, 0],
                   [0, 4, 6, 0],
                   [0, 10, 0, 0]], dtype=np.int64)
    m = metrics_from_cm(cm, names, ignore_index=0, report_only=())
    pc = m["per_class"]
    # a: tp 8, fp 14, fn 2  -> iou 8/24, prec 8/22, rec 8/10
    assert abs(pc["a"]["iou"] - 8 / 24) < TOL, pc["a"]["iou"]
    assert abs(pc["a"]["precision"] - 8 / 22) < TOL
    assert abs(pc["a"]["recall"] - 8 / 10) < TOL
    assert abs(pc["b"]["iou"] - 6 / 12) < TOL
    assert pc["c"]["iou"] == 0.0 and pc["c"]["precision"] is None and pc["c"]["recall"] == 0.0
    assert m["evaluated_pixels"] == 30, m["evaluated_pixels"]
    print("  ignored-label row dropped from every denominator : OK")
    print("  per-class IoU / precision / recall : OK")
    assert abs(m["macro_iou"] - (8 / 24 + 6 / 12 + 0.0) / 3) < TOL
    assert abs(m["overall_accuracy"] - 14 / 30) < TOL
    print("  macro-IoU, overall accuracy        : OK")
    fw = (10 / 30) * (8 / 24) + (10 / 30) * (6 / 12) + (10 / 30) * 0.0
    assert abs(m["freq_weighted_iou"] - fw) < TOL, (m["freq_weighted_iou"], fw)
    print("  frequency-weighted IoU             : OK")
    assert m["worst1_mean_iou"] == 0.0 and m["worst1_classes"] == "c"
    assert abs(m["worst2_mean_iou"] - (0.0 + 8 / 24) / 2) < TOL
    assert m["worst2_classes"] == "c|a", m["worst2_classes"]
    print("  worst-k (min-class and k-worst mean): OK")

    # THE CASE THAT BROKE THE FIRST DRAFT: predicting the ignore class on a real pixel.
    # It must count as a false negative for the true class and stay in the denominator, which is
    # what `per_category_metrics` does and what unet_resnet34_10ch_unw actually exhibits.
    cm_ig = np.array([[0, 0, 0, 0],
                      [3, 7, 0, 0],      # 3 true-a pixels predicted `unknown`
                      [0, 0, 10, 0],
                      [0, 0, 0, 10]], dtype=np.int64)
    m_ig = metrics_from_cm(cm_ig, names, ignore_index=0, report_only=())
    assert abs(m_ig["per_class"]["a"]["iou"] - 7 / 10) < TOL, m_ig["per_class"]["a"]
    assert abs(m_ig["per_class"]["a"]["precision"] - 1.0) < TOL      # no other class blamed
    assert abs(m_ig["per_class"]["a"]["recall"] - 7 / 10) < TOL
    assert m_ig["evaluated_pixels"] == 30, m_ig["evaluated_pixels"]
    assert abs(m_ig["overall_accuracy"] - 27 / 30) < TOL
    print("  predicted-ignore counts as FN, not dropped : OK")

    # a class with zero support must leave the macro average, not enter it as a zero
    names2 = ["unknown", "a", "b", "z"]
    cm2 = np.array([[0, 0, 0, 0], [0, 5, 0, 0], [0, 0, 5, 0], [0, 0, 0, 0]], dtype=np.int64)
    m2 = metrics_from_cm(cm2, names2, ignore_index=0, report_only=())
    assert m2["n_macro_classes"] == 2 and abs(m2["macro_iou"] - 1.0) < TOL
    assert m2["per_class"]["z"]["present"] is False
    print("  absent class excluded from macro   : OK")

    # binary collapse: two sealed classes that confuse each other must score perfectly when merged
    cm3 = np.zeros((11, 11), dtype=np.int64)
    i = {nm: k for k, nm in enumerate(CLASS_ORDER)}
    cm3[i["asfalt"], i["asfalt"]] = 30
    cm3[i["asfalt"], i["fliser"]] = 20
    cm3[i["fliser"], i["asfalt"]] = 20
    cm3[i["fliser"], i["fliser"]] = 30
    cm3[i["ubefestet"], i["ubefestet"]] = 100
    b = binary_collapse(cm3, CLASS_ORDER)
    assert b["per_class"]["sealed"]["iou"] == 1.0 and b["per_class"]["unsealed"]["iou"] == 1.0
    assert b["overall_accuracy"] == 1.0, b["overall_accuracy"]
    print("  binary collapse absorbs within-group confusion : OK")

    # and it must NOT absorb a genuine sealed/unsealed error
    cm4 = cm3.copy()
    cm4[i["ubefestet"], i["asfalt"]] = 100          # half the unsealed pixels called sealed
    b4 = binary_collapse(cm4, CLASS_ORDER)
    assert abs(b4["per_class"]["unsealed"]["recall"] - 0.5) < TOL, b4["per_class"]["unsealed"]
    assert b4["overall_accuracy"] < 1.0
    print("  binary collapse keeps cross-group error : OK")

    # collapse must preserve the scored pixel count exactly
    assert b4["evaluated_pixels"] == int(cm4[1:, :].sum()), \
        (b4["evaluated_pixels"], int(cm4[1:, :].sum()))
    print("  binary collapse conserves scored pixels : OK")
    print("\nSELFTEST PASSED")


# ----------------------------------------------------------------------------------------------
def discover_cells():
    out = []
    for p in sorted(C.SPATIAL_MATRIX.glob("*/oof_*")):
        if (p / "pooled_oof_metrics.json").is_file():
            out.append((p.parent.name, p.name[len("oof_"):], p / "pooled_oof_metrics.json"))
    return sorted(out, key=lambda t: t[1])


def main():
    ap = argparse.ArgumentParser(description="metric breadth over the scored cells")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    cells = discover_cells()
    print(f"{len(cells)} scored cells found under {C.SPATIAL_MATRIX}")

    wide, long, binary, mismatches = [], [], [], []
    for model_dir, cell, path in cells:
        j = json.loads(path.read_text())
        h = j["headline_pooled_oof"]
        cm = j["global_confusion_matrix"]
        m = metrics_from_cm(cm, CLASS_ORDER)

        # cross-check against what pooled_oof_metrics.py already wrote
        for key in ("macro_iou", "macro_f1", "overall_accuracy", "evaluated_pixels"):
            got, want = m[key], h[key]
            if abs(float(got) - float(want)) > (1e-9 if key != "evaluated_pixels" else 0):
                mismatches.append((cell, key, got, want))

        b = binary_collapse(cm, CLASS_ORDER)
        row = {"cell": cell, "model_dir": model_dir,
               "macro_iou": m["macro_iou"], "macro_f1": m["macro_f1"],
               "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
               "freq_weighted_iou": m["freq_weighted_iou"],
               "overall_accuracy": m["overall_accuracy"],
               "n_macro_classes": m["n_macro_classes"],
               "evaluated_pixels": m["evaluated_pixels"]}
        for k in WORST_K:
            row[f"worst{k}_mean_iou"] = m[f"worst{k}_mean_iou"]
            row[f"worst{k}_classes"] = m[f"worst{k}_classes"]
        row["binary_sealed_iou"] = b["per_class"]["sealed"]["iou"]
        row["binary_unsealed_iou"] = b["per_class"]["unsealed"]["iou"]
        row["binary_macro_iou"] = b["macro_iou"]
        row["binary_overall_accuracy"] = b["overall_accuracy"]
        row["binary_sealed_precision"] = b["per_class"]["sealed"]["precision"]
        row["binary_sealed_recall"] = b["per_class"]["sealed"]["recall"]
        wide.append(row)

        two = b["confusion_2x2_rows_true_cols_pred"]
        binary.append({"cell": cell,
                       "sealed_as_sealed": two[0][0], "sealed_as_unsealed": two[0][1],
                       "unsealed_as_sealed": two[1][0], "unsealed_as_unsealed": two[1][1],
                       "predicted_unknown_pixels": b["predicted_unknown_pixels"],
                       "evaluated_pixels": b["evaluated_pixels"],
                       **{k: b[k] for k in ("macro_iou", "overall_accuracy", "freq_weighted_iou")}})

        for name, rec in m["per_class"].items():
            long.append({"cell": cell, "class": name, "index": rec["index"],
                         "support_pixels": rec["support_pixels"],
                         "predicted_pixels": rec["predicted_pixels"],
                         "present": rec["present"], "in_macro": rec["in_macro"],
                         "iou": rec["iou"], "precision": rec["precision"],
                         "recall": rec["recall"], "f1": rec["f1"]})

    if mismatches:
        for cell, key, got, want in mismatches:
            print(f"  MISMATCH {cell}.{key}: recomputed {got} vs stored {want}")
        sys.exit("recomputation disagrees with pooled_oof_metrics.py -- refusing to write")
    print(f"cross-check: macro_iou, macro_f1, overall_accuracy and evaluated_pixels reproduce the "
          f"stored values on all {len(cells)} cells (exact)")

    w = C.assert_writes_are_local(C.TABLES / "metric_breadth_by_cell.csv")
    with open(w, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(wide[0]))
        wr.writeheader()
        wr.writerows(wide)
    print(f"wrote {w}  ({len(wide)} cells)")

    l = C.assert_writes_are_local(C.TABLES / "metric_breadth_per_class.csv")
    with open(l, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(long[0]))
        wr.writeheader()
        wr.writerows(long)
    print(f"wrote {l}  ({len(long)} rows)")

    bfile = C.assert_writes_are_local(C.TABLES / "metric_breadth_binary_collapse.csv")
    with open(bfile, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(binary[0]))
        wr.writeheader()
        wr.writerows(binary)
    print(f"wrote {bfile}  ({len(binary)} cells)")

    prov = C.assert_writes_are_local(C.TABLES / "metric_breadth_provenance.json")
    prov.write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "n_cells": len(cells),
        "cells": [c for _m, c, _p in cells],
        "source": "each cell's global_confusion_matrix in pooled_oof_metrics.json",
        "ignore_index": IGNORE_INDEX,
        "report_only_classes": sorted(REPORT_ONLY),
        "freq_weighted_iou_definition": "sum_c (support_c / evaluated_px) * IoU_c over macro classes",
        "worst_case_definition": ("min single-class IoU and the mean of the k worst macro classes, "
                                  "k in 1,2,3 -- the operational definition recorded in this "
                                  "project's gap analysis for G3.3. NOT yet checked against "
                                  "Wang Z. et al. 2023; verify on the reference-canon day before "
                                  "attributing the definition to that paper."),
        "binary_collapse": {"sealed": sorted(SEALED), "unsealed": sorted(UNSEALED),
                            "dropped": ["unknown (ignore index)", "unknown2 (report-only)"],
                            "status": "MAPPING AWAITS THE AUTHOR'S CONFIRMATION"},
        "cross_check": "macro_iou / macro_f1 / overall_accuracy / evaluated_pixels reproduce the "
                       "stored pooled values exactly on every cell",
    }, indent=2), encoding="utf-8")
    print(f"wrote {prov}")

    wide.sort(key=lambda r: -r["macro_iou"])
    print("\n=== five metric families, same cells, different stories (sorted by macro-IoU) ===")
    print(f"  {'cell':<40}{'macroIoU':>9}{'FW-IoU':>9}{'acc':>8}{'macroF1':>9}"
          f"{'worst1':>8}{'binAcc':>8}  worst class")
    for r in wide:
        print(f"  {r['cell']:<40}{r['macro_iou']:>9.4f}{r['freq_weighted_iou']:>9.4f}"
              f"{r['overall_accuracy']:>8.4f}{r['macro_f1']:>9.4f}{r['worst1_mean_iou']:>8.4f}"
              f"{r['binary_overall_accuracy']:>8.4f}  {r['worst1_classes']}")
    print("\nNOTE: the binary sealed/unsealed mapping is a judgement, flagged for confirmation.")


if __name__ == "__main__":
    main()
