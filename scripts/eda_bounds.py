#!/usr/bin/env python
"""
E5 -- Performance bounds: what is attainable, so 0.24 to 0.36 Macro-IoU can be read against
something rather than in a vacuum.

WHY THIS EXISTS
    Great Plan 3.0 section 0 asks for "a trivial lower bound and a sense of the attainable upper
    bound, so the 0.24 to 0.36 Macro-IoU is read against what is achievable, not in a vacuum".
    Reporting a model's accuracy without stating what a predictor that does nothing would score is
    the single easiest way to overstate a result, and KDS's headline is an accuracy figure.

THE BOUNDS
    B1  constant majority      predict ubefestet everywhere. The floor. Needs no model at all.
    B2  prior-matched random   draw a class per pixel from the label prior. The floor for a
                               predictor that knows the class frequencies and nothing else.
    B3  per-tile majority      predict each tile's own dominant surface, perfectly. An ORACLE: it
                               uses the answer. Bounds what any model that resolves tiles but not
                               pixels could reach.
    B4  binary constant        the same trivial predictor scored on the sealed/unsealed split KDS
                               actually operates on, which is what its ~95 percent headline refers
                               to.
    Measured models are then listed alongside, straight from the 24 existing pooled OOF cells.

METHOD NOTE
    Every bound is expressed as a CONFUSION MATRIX and pushed through the very same
    per_category_metrics.metrics_from_confusion used for the real results, with the same
    ignore_index, unknown2 and absent-class rules. So the comparison is exact rather than a
    re-derivation that might apply the rules differently.

READ-ONLY over the source data. Writes only under exploratory_data_analysis/.

    python eda_bounds.py [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402


def supports_from_audit() -> np.ndarray:
    """Per-class label pixel counts over the whole pool, from the existing audit."""
    audit = C.load_class_pixel_audit()
    return np.array([audit[name]["pixel_count"] for name in C.CODES], dtype=np.int64)


def score(cm: np.ndarray, tag: str) -> dict:
    """Run a confusion matrix through the project's own metric code."""
    return C.metrics_from_confusion(cm, C.CODES, ignore_index=C.IGNORE_INDEX,
                                    report_only=(C.UNKNOWN2_INDEX,), split_tag=tag)


# ----------------------------------------------------------------------------------------------
# The bounds
# ----------------------------------------------------------------------------------------------
def b1_constant_majority(support: np.ndarray) -> np.ndarray:
    """Predict ubefestet on every pixel. All mass lands in one column."""
    cm = np.zeros((C.NCLASS, C.NCLASS), dtype=np.int64)
    cm[:, C.UBEFESTET_INDEX] = support
    return cm


def b2_prior_matched_random(support: np.ndarray) -> np.ndarray:
    """Expected confusion of drawing each pixel's prediction from the label prior.

    Scored classes only: a random predictor never emits the ignore class, and unknown2 has zero
    support so its prior is zero. Rounding is applied last so row totals stay exact.
    """
    scored = support.copy()
    scored[C.IGNORE_INDEX] = 0
    prior = scored / scored.sum()
    cm = np.outer(support.astype(np.float64), prior)
    return np.rint(cm).astype(np.int64)


def b3_per_tile_majority_oracle(df) -> np.ndarray:
    """Predict, for every tile, that tile's own most common predicted class -- perfectly.

    This is an oracle (it reads the labels), so it is an UPPER bound, not a baseline. It answers:
    how far could a model get if it identified each tile's dominant surface and nothing finer?
    """
    px_cols = [C.px_col(n) for n in C.CODES]
    px = df[px_cols].to_numpy(dtype=np.int64)          # (n_tiles, 11)

    pred_idx = np.array(C.PREDICTED)
    winner_pos = px[:, pred_idx].argmax(axis=1)
    winner = pred_idx[winner_pos]                       # chosen class per tile

    # A tile with no predicted-class pixels at all has nothing to predict; skip it.
    has_any = px[:, pred_idx].sum(axis=1) > 0

    cm = np.zeros((C.NCLASS, C.NCLASS), dtype=np.int64)
    for true_cls in range(C.NCLASS):
        col = px[:, true_cls]
        np.add.at(cm[true_cls], winner[has_any], col[has_any])
    return cm


def b4_binary_constant(support: np.ndarray) -> dict:
    """The trivial predictor scored on the binary sealed/unsealed task KDS reports ~95% on.

    ubefestet is the only unsealed predicted class; every other predicted class is a sealed
    surface or a roof structure. Reported separately because it is a 2-class problem, so the
    9-class macro machinery does not apply.
    """
    scored = support.copy()
    scored[C.IGNORE_INDEX] = 0
    total = scored.sum()
    unsealed = int(scored[C.UBEFESTET_INDEX])
    sealed = int(total - unsealed)
    # Always predicting "unsealed" gets every unsealed pixel right and every sealed pixel wrong.
    return {
        "task": "binary befaestet / ubefaestet",
        "predictor": "constant unsealed",
        "accuracy": unsealed / total,
        "iou_unsealed": unsealed / total,      # TP/(TP+FP+FN) with FN=0, FP=sealed
        "iou_sealed": 0.0,
        "macro_iou_2class": (unsealed / total) / 2.0,
        "unsealed_px": unsealed,
        "sealed_px": sealed,
    }


# ----------------------------------------------------------------------------------------------
# The measured models, for context
# ----------------------------------------------------------------------------------------------
def measured_cells() -> list[dict]:
    """Read the 24 existing pooled OOF cells. Never modifies them."""
    out = []
    for path in sorted(C.SPATIAL_MATRIX.glob("*/oof_*/pooled_oof_metrics.json")):
        with open(path) as fh:
            head = json.load(fh)["headline_pooled_oof"]
        out.append({
            "cell": path.parent.name.replace("oof_", ""),
            "macro_iou": head["macro_iou"],
            "macro_f1": head["macro_f1"],
            "overall_accuracy": head["overall_accuracy"],
        })
    return sorted(out, key=lambda r: -r["macro_iou"])


def main() -> None:
    ap = argparse.ArgumentParser(description="E5 -- performance bounds")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    C.ensure_out_dirs()
    support = supports_from_audit()
    scored_total = int(support.sum() - support[C.IGNORE_INDEX])

    C.banner("E5 -- performance bounds")
    print(f"label pixels total : {support.sum():,}")
    print(f"ignore (unknown)   : {support[C.IGNORE_INDEX]:,} "
          f"({100 * support[C.IGNORE_INDEX] / support.sum():.2f}%)")
    print(f"scored pixels      : {scored_total:,}")

    bounds = {}
    bounds["B1_constant_majority"] = score(b1_constant_majority(support), "B1_constant_majority")
    bounds["B2_prior_matched_random"] = score(b2_prior_matched_random(support),
                                             "B2_prior_matched_random")

    df = C.load_tile_inventory()
    bounds["B3_per_tile_majority_oracle"] = score(b3_per_tile_majority_oracle(df),
                                                  "B3_per_tile_majority_oracle")
    binary = b4_binary_constant(support)

    # ------------------------------------------------------------------------------------------
    C.banner("BOUNDS (9-class task, pooled over the whole labelled pool)")
    print(f"{'bound':<32}{'Macro-IoU':>11}{'macro-F1':>10}{'overall acc':>13}")
    for key, label in [("B1_constant_majority", "B1 constant majority"),
                       ("B2_prior_matched_random", "B2 prior-matched random"),
                       ("B3_per_tile_majority_oracle", "B3 per-tile majority ORACLE")]:
        m = bounds[key]
        print(f"{label:<32}{m['macro_iou']:>11.4f}{m['macro_f1']:>10.4f}"
              f"{m['overall_accuracy']:>13.4f}")

    cells = measured_cells()
    if cells:
        best, worst = cells[0], cells[-1]
        prod = next((c for c in cells if c["cell"] == "unet_resnet34_rgb"), None)
        print(f"{'-' * 66}")
        print(f"{'measured: best cell':<32}{best['macro_iou']:>11.4f}{best['macro_f1']:>10.4f}"
              f"{best['overall_accuracy']:>13.4f}   {best['cell']}")
        if prod:
            print(f"{'measured: production arch':<32}{prod['macro_iou']:>11.4f}"
                  f"{prod['macro_f1']:>10.4f}{prod['overall_accuracy']:>13.4f}   {prod['cell']}")
        print(f"{'measured: worst cell':<32}{worst['macro_iou']:>11.4f}{worst['macro_f1']:>10.4f}"
              f"{worst['overall_accuracy']:>13.4f}   {worst['cell']}")

    # ------------------------------------------------------------------------------------------
    C.banner("THE HEADLINE READING")
    b1 = bounds["B1_constant_majority"]
    if cells:
        acc_gain = best["overall_accuracy"] - b1["overall_accuracy"]
        iou_gain = best["macro_iou"] / b1["macro_iou"]
        print(f"A predictor that always answers 'ubefestet' scores "
              f"{b1['overall_accuracy']:.4f} overall accuracy.")
        print(f"The best of the 24 trained cells scores {best['overall_accuracy']:.4f}.")
        print(f"\n  -> the entire accuracy headline is worth {100 * acc_gain:.1f} percentage points")
        print(f"     over predicting a single class everywhere.")
        print(f"\nOn Macro-IoU the same comparison is {b1['macro_iou']:.4f} against "
              f"{best['macro_iou']:.4f}, a {iou_gain:.1f}x gap.")
        print("Accuracy hides the model's contribution; Macro-IoU exposes it. That asymmetry is")
        print("the argument for the per-category metric, made without appealing to any literature.")

    print(f"\nBinary task ({binary['task']}), which is what KDS's ~95% figure refers to:")
    print(f"  constant 'unsealed' predictor accuracy: {binary['accuracy']:.4f}")
    print(f"  -> a reported ~95% on the binary split is {100 * (0.95 - binary['accuracy']):.1f} "
          f"points above doing nothing.")

    # ------------------------------------------------------------------------------------------
    C.banner("PER-CLASS IoU UNDER EACH BOUND")
    print(f"{'class':<12}{'B1':>9}{'B2':>9}{'B3 oracle':>11}"
          + (f"{'best cell':>11}" if cells else ""))
    best_per_class = {}
    if cells:
        p = next(C.SPATIAL_MATRIX.glob(f"*/oof_{best['cell']}/pooled_oof_metrics.json"), None)
        if p:
            with open(p) as fh:
                best_per_class = json.load(fh)["headline_pooled_oof"]["per_class"]
    for name in C.CODES:
        if name == "unknown":
            continue
        vals = []
        for key in ("B1_constant_majority", "B2_prior_matched_random",
                    "B3_per_tile_majority_oracle"):
            v = bounds[key]["per_class"][name]["iou"]
            vals.append("     n/a" if v is None else f"{v:9.4f}")
        line = f"{name:<12}" + "".join(vals)
        if best_per_class:
            bv = best_per_class.get(name, {}).get("iou")
            line += "       n/a" if bv is None else f"{bv:11.4f}"
        print(line)

    # ------------------------------------------------------------------------------------------
    payload = {
        "scored_pixels": scored_total,
        "ignore_pixels": int(support[C.IGNORE_INDEX]),
        "bounds": {k: {kk: vv for kk, vv in v.items()
                       if kk != "confusion_matrix_rows_label_cols_pred"}
                   for k, v in bounds.items()},
        "binary_constant": binary,
        "measured_cells": cells,
        "note": ("B3 is an oracle and therefore an upper bound, not a baseline. "
                 "The linear-probe floor (module E1) and the annotation ceiling (module E4) "
                 "extend this table once those modules have run."),
    }
    out = C.write_json(payload, C.TABLES / "performance_bounds.json")
    print(f"\nwrote {out}")

    import csv as _csv
    csv_path = C.assert_writes_are_local(C.TABLES / "performance_bounds.csv")
    with open(csv_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["bound", "macro_iou", "macro_f1", "overall_accuracy", "kind"])
        for key, kind in [("B1_constant_majority", "lower bound"),
                          ("B2_prior_matched_random", "lower bound"),
                          ("B3_per_tile_majority_oracle", "oracle / upper bound")]:
            m = bounds[key]
            w.writerow([key, m["macro_iou"], m["macro_f1"], m["overall_accuracy"], kind])
        for c in cells:
            w.writerow([c["cell"], c["macro_iou"], c["macro_f1"], c["overall_accuracy"], "measured"])
    print(f"wrote {csv_path}")


def selftest() -> None:
    """Verify the bound constructions on a tiny hand-checkable case."""
    C.banner("E5 selftest (synthetic, no files touched)")

    # 100 scored pixels: 80 ubefestet (class 4), 20 asfalt (class 1). Plus 50 ignore pixels.
    support = np.zeros(C.NCLASS, dtype=np.int64)
    support[C.IGNORE_INDEX] = 50
    support[1] = 20
    support[C.UBEFESTET_INDEX] = 80

    m = score(b1_constant_majority(support), "t")
    assert abs(m["overall_accuracy"] - 0.8) < 1e-12, m["overall_accuracy"]
    assert abs(m["per_class"]["ubefestet"]["iou"] - 0.8) < 1e-12
    assert m["per_class"]["asfalt"]["iou"] == 0.0
    # macro is over the 9 predicted classes, but only those PRESENT are averaged (absent-class rule)
    assert m["n_macro_classes_evaluated"] == 2, m["n_macro_classes_evaluated"]
    assert abs(m["macro_iou"] - 0.8 / 2) < 1e-12, m["macro_iou"]
    print("B1 constant majority      : OK (acc 0.80, ubefestet IoU 0.80, others 0)")
    print("  absent-class rule honoured: only present classes enter the macro, never counted as 0")

    m2 = score(b2_prior_matched_random(support), "t")
    # prior 0.8/0.2 -> expected TP_ubef = 80*0.8 = 64, FP = 20*0.8 = 16, FN = 16
    assert abs(m2["per_class"]["ubefestet"]["iou"] - 64 / (64 + 16 + 16)) < 1e-3
    print("B2 prior-matched random   : OK (matches the closed-form expectation)")

    # B3 on two synthetic tiles: tile A is mostly ubefestet, tile B mostly asfalt.
    import pandas as pd
    rows = []
    for px1, px4 in [(10, 90), (70, 30)]:
        r = {C.px_col(n): 0 for n in C.CODES}
        r[C.px_col("asfalt")] = px1
        r[C.px_col("ubefestet")] = px4
        rows.append(r)
    cm3 = b3_per_tile_majority_oracle(pd.DataFrame(rows))
    # tile A -> all 100 px predicted ubefestet; tile B -> all 100 px predicted asfalt
    assert cm3[C.UBEFESTET_INDEX, C.UBEFESTET_INDEX] == 90 and cm3[1, C.UBEFESTET_INDEX] == 10
    assert cm3[1, 1] == 70 and cm3[C.UBEFESTET_INDEX, 1] == 30
    print("B3 per-tile majority      : OK (each tile's dominant class wins that whole tile)")

    b = b4_binary_constant(support)
    assert abs(b["accuracy"] - 0.8) < 1e-12
    print("B4 binary constant        : OK")

    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
