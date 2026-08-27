#!/usr/bin/env python
"""
Task 2.1 of the 2026-08-24 work order: give the per-class learning curves their 100 % points.

lc25 / lc50 / lc75 were scored with `score_single_fold.py` and each wrote a
`single_fold_metrics.json`. The 100 % point needs no compute -- it is the frozen
`unet_resnet34_rgb` cell's fold-0 entry, which already sits inside that cell's
`pooled_oof_metrics.json` under `per_fold_diagnostic`. This lifts it out into the same shape as the
three curve points so every consumer reads four files with one schema.

Two guards, because the curve is only readable if all four points are the same measurement:
  - the extracted fold-0 macro-IoU / accuracy must reproduce the anchor recorded in the
    2026-08-23 launch block (0.3292 / 0.8720);
  - `evaluated_pixels` and every class's `support_pixels` must match the three curve points exactly,
    which is what "scored on a byte-identical held-out set" means in practice.

Also writes a tidy per-class curve table (long form) for the results-chapter figure, regenerable
from these four JSONs alone.

Read-only apart from its own outputs under exploratory_data_analysis/. CPU, seconds.

    python extract_lc100_per_class.py
"""
import csv
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

FROZEN_OOF = (C.SPATIAL_MATRIX / "unet_resnet34" / "oof_unet_resnet34_rgb" /
              "pooled_oof_metrics.json")
LC_DIR = C.TABLES / "learning_curve"
POINTS = [("lc25", 3222, 25.0), ("lc50", 6444, 50.1), ("lc75", 9245, 71.8), ("lc100", 12875, 100.0)]
ANCHOR = {"macro_iou": 0.3292, "overall_accuracy": 0.8720}   # 2026-08-23 launch block, 4 dp
PRED_FOLDER_LC100 = ("../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_fold0/"
                     "models/example_dataset")


def main():
    if not FROZEN_OOF.is_file():
        sys.exit(f"missing frozen cell scores: {FROZEN_OOF}")
    pooled = json.loads(FROZEN_OOF.read_text())

    diags = {d["split"]: d for d in pooled["per_fold_diagnostic"]}
    if "fold_0" not in diags:
        sys.exit(f"no fold_0 entry in per_fold_diagnostic (have {sorted(diags)})")
    m = diags["fold_0"]

    # --- guard 1: the anchor the curve was launched against ---
    for key, want in ANCHOR.items():
        got = m[key]
        if round(got, 4) != want:
            sys.exit(f"anchor mismatch on {key}: extracted {got:.6f}, launch block says {want}")
    print(f"anchor reproduced: macro_iou {m['macro_iou']:.6f}  accuracy {m['overall_accuracy']:.6f}")

    # --- guard 2: identical held-out set as the three curve points ---
    curves = {}
    for tag, _tiles, _pct in POINTS[:-1]:
        p = LC_DIR / tag / "single_fold_metrics.json"
        if not p.is_file():
            sys.exit(f"missing curve point: {p}")
        curves[tag] = json.loads(p.read_text())

    for tag, doc in curves.items():
        cm = doc["metrics"]
        if cm["evaluated_pixels"] != m["evaluated_pixels"]:
            sys.exit(f"{tag}: evaluated_pixels {cm['evaluated_pixels']} != lc100 "
                     f"{m['evaluated_pixels']} -- points are not on the same held-out set")
        for cls, rec in cm["per_class"].items():
            if rec["support_pixels"] != m["per_class"][cls]["support_pixels"]:
                sys.exit(f"{tag}/{cls}: support {rec['support_pixels']} != lc100 "
                         f"{m['per_class'][cls]['support_pixels']}")
    print(f"held-out set identical across all four points: "
          f"{m['evaluated_pixels']:,} evaluated pixels, {POINTS[0][0]}..lc100")

    # --- write lc100 in the curve-point schema ---
    out_dir = LC_DIR / "lc100"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fold": 0,
        "pred_folder": PRED_FOLDER_LC100,
        "n_tiles": pooled["fold_tile_counts"]["fold_0"] if isinstance(
            pooled.get("fold_tile_counts"), dict) else C.EXPECTED_FOLD_COUNTS[0],
        "metrics": m,
        "source": str(FROZEN_OOF),
        "note": ("lc100 is the FROZEN unet_resnet34_rgb cell's fold-0 diagnostic, lifted out of its "
                 "pooled_oof_metrics.json. No compute was run for this point; it is the anchor the "
                 "three trained points were measured against."),
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    target = C.assert_writes_are_local(LC_DIR / "lc100_per_class.json")
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    also = C.assert_writes_are_local(out_dir / "single_fold_metrics.json")
    also.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {target}")
    print(f"wrote {also}  (same content, curve-point layout)")

    # --- tidy long-form curve table for the figure ---
    curves["lc100"] = payload
    rows = []
    classes = [c for c, r in m["per_class"].items() if r["in_macro"]]
    for tag, tiles, pct in POINTS:
        cm = curves[tag]["metrics"]
        rows.append({"point": tag, "train_tiles": tiles, "train_pct_of_fold0_pool": pct,
                     "scope": "aggregate", "class": "MACRO", "metric": "macro_iou",
                     "value": cm["macro_iou"], "support_pixels": cm["evaluated_pixels"]})
        rows.append({"point": tag, "train_tiles": tiles, "train_pct_of_fold0_pool": pct,
                     "scope": "aggregate", "class": "OVERALL", "metric": "overall_accuracy",
                     "value": cm["overall_accuracy"], "support_pixels": cm["evaluated_pixels"]})
        for cls in classes:
            rec = cm["per_class"][cls]
            rows.append({"point": tag, "train_tiles": tiles, "train_pct_of_fold0_pool": pct,
                         "scope": "per_class", "class": cls, "metric": "iou",
                         "value": rec["iou"], "support_pixels": rec["support_pixels"]})

    tidy = C.assert_writes_are_local(LC_DIR / "learning_curve_per_class.csv")
    with open(tidy, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {tidy}  ({len(rows)} rows, {len(classes)} macro classes x 4 points + aggregates)")

    print("\n=== per-class IoU, lc25 -> lc50 -> lc75 -> lc100 (fold 0, descriptive) ===")
    print(f"  {'class':<12}" + "".join(f"{t:>10}" for t, _, _ in POINTS))
    for cls in classes:
        vals = [curves[t]["metrics"]["per_class"][cls]["iou"] for t, _, _ in POINTS]
        print(f"  {cls:<12}" + "".join(f"{v:>10.4f}" for v in vals))
    for name, key in (("MACRO-IoU", "macro_iou"), ("accuracy", "overall_accuracy")):
        vals = [curves[t]["metrics"][key] for t, _, _ in POINTS]
        print(f"  {name:<12}" + "".join(f"{v:>10.4f}" for v in vals))


if __name__ == "__main__":
    main()
