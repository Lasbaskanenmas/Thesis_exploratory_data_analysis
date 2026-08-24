#!/usr/bin/env python
"""
Score ONE held-out fold. Needed for the Plan 3.0 section 4 learning curve.

`pooled_oof_metrics.py` requires all three folds because it pools them into one out-of-fold matrix.
The learning curve trains on subsets of a single fold's training pool and always evaluates on that
same held-out fold, so there is nothing to pool -- but the number must be directly comparable to the
frozen cell's per-fold diagnostic, which is what anchors the 100% point.

It is: this reuses `per_category_metrics.pooled_confusion_matrix` and `metrics_from_confusion`, the
same functions `pooled_oof_metrics` calls per fold. `--validate` proves it by reproducing the frozen
`unet_resnet34_rgb` fold-0 diagnostic from its predictions on disk.

Read-only except for its own --out JSON.

    python score_single_fold.py --validate
    python score_single_fold.py --fold 0 --pred_folder <...> --out <...>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402
import per_category_metrics as pcm  # noqa: E402


def score(fold, pred_folder, out=None, quiet=False):
    codes = pcm.load_codes(str(C.CODES_TXT))
    route_fold = C.load_fold_assignment(str(C.FOLD_ASSIGNMENT_CSV))
    folds = C.heldout_tiles_by_fold(str(C.ALL_TXT), route_fold)
    tiles = folds[fold]
    if not quiet:
        print(f"  fold {fold}: {len(tiles):,} held-out tiles")
        print(f"  predictions: {pred_folder}")

    cm = pcm.pooled_confusion_matrix(tiles, str(pred_folder), str(C.LABEL_DIR),
                                     n_classes=len(codes))
    m = pcm.metrics_from_confusion(cm, codes, ignore_index=pcm.DEFAULT_IGNORE_INDEX,
                                   report_only=pcm.DEFAULT_REPORT_ONLY,
                                   split_tag=f"fold_{fold}")
    result = {"fold": fold, "pred_folder": str(pred_folder), "n_tiles": len(tiles),
              "metrics": m, "confusion_matrix": cm.tolist()}
    if not quiet:
        print(f"\n  Macro-IoU {m['macro_iou']:.4f}   macro-F1 {m['macro_f1']:.4f}   "
              f"accuracy {m['overall_accuracy']:.4f}   "
              f"({m['n_macro_classes_evaluated']}/{m['n_macro_classes_defined']} classes)")
        for name, row in m["per_class"].items():
            if row["in_macro"]:
                iou = "n/a" if row["iou"] is None else f"{row['iou']:.4f}"
                print(f"    {name:<12} IoU={iou}  support_px={row['support_pixels']:,}")
    if out:
        C.write_json(result, Path(out) / "single_fold_metrics.json")
        print(f"\n  wrote {Path(out) / 'single_fold_metrics.json'}")
    return result


def validate():
    """Reproduce the frozen unet_resnet34_rgb fold-0 diagnostic. This is the 100% anchor."""
    C.banner("score_single_fold validation against the frozen fold-0 diagnostic")
    frozen = json.load(open(C.SPATIAL_MATRIX / "unet_resnet34" / "oof_unet_resnet34_rgb"
                            / "pooled_oof_metrics.json"))
    ref = frozen["per_fold_diagnostic"][0]
    pred = (C.SPATIAL_MATRIX / "unet_resnet34" / "unet_resnet34_rgb_fold0"
            / "models" / "example_dataset")
    got = score(0, pred, quiet=False)["metrics"]

    print(f"\n  {'quantity':<22}{'recomputed':>14}{'frozen':>14}{'delta':>12}")
    ok = True
    for k in ("macro_iou", "macro_f1", "overall_accuracy"):
        d = abs(got[k] - ref[k])
        ok &= d < 1e-9
        print(f"  {k:<22}{got[k]:>14.6f}{ref[k]:>14.6f}{d:>12.2e}")
    for cls, row in ref["per_class"].items():
        if not row["in_macro"] or row["iou"] is None:
            continue
        d = abs(got["per_class"][cls]["iou"] - row["iou"])
        ok &= d < 1e-9
        if d >= 1e-9:
            print(f"  MISMATCH {cls}: {got['per_class'][cls]['iou']} vs {row['iou']}")
    print(f"\n  per-class IoU identical for all {sum(1 for r in ref['per_class'].values() if r['in_macro'])} "
          f"macro classes: {ok}")
    print("\n  VALIDATION PASS -- learning-curve points are directly comparable to the anchor"
          if ok else "\n  VALIDATION FAILED")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fold", type=int)
    ap.add_argument("--pred_folder")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.validate:
        validate()
    elif a.fold is not None and a.pred_folder:
        score(a.fold, a.pred_folder, a.out)
    else:
        ap.error("pass --validate, or --fold and --pred_folder")


if __name__ == "__main__":
    main()
