#!/usr/bin/env python
"""
E6 -- Per-route out-of-fold confusion matrices: 24 cells x 16 routes = 384 matrices, cached once.

WHY THIS EXISTS
    Two jobs, one scan.

    For SQ1 it answers: how much does performance vary by GEOGRAPHY, and is that variation driven
    by place or merely by the class mix each route happens to contain? The existing artifacts stop
    at per-FOLD granularity, so this cannot currently be asked at all.

    For the Great Plan 3.0 section 5 statistics it is step 1. The route is the replication unit the
    whole blocking design rests on, and the formal test is a route-level Wilcoxon over N = 16 pairs.
    Folds cannot carry it: three folds partition the same data, are not independent, and a
    two-sided Wilcoxon at N = 3 cannot drop below p ~ 0.25 no matter what the effect is. Caching
    the 384 matrices means Part B never has to re-read the 463,536 predictions.

METHOD
    Each route's tiles were held out in exactly one fold, so route -> fold -> that cell's
    prediction directory is unambiguous. The loop is TILE-OUTER, CELL-INNER: every label is read
    once and every prediction once, roughly halving the IO against the obvious cell-outer ordering.

VALIDATION (the strongest check in the whole EDA)
    Summing a cell's 16 route matrices must reproduce that cell's `global_confusion_matrix` from
    the existing pooled_oof_metrics.json ELEMENT BY ELEMENT. That one assertion simultaneously
    validates the route-to-fold-to-directory routing, the tile partition, and the accumulation.
    Run --cells on a single cell first to confirm it before committing to the full sweep.

READ-ONLY over the source data -- the 24 pooled_oof_metrics.json files are read for validation and
never rewritten. Writes only under exploratory_data_analysis/.

    python eda_route_cell_metrics.py [--cells NAME[,NAME...]] [--procs N] [--limit N]
    python eda_route_cell_metrics.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

_CELLDIRS: list[dict[int, str]] = []


def discover_cells() -> list[str]:
    """Every scored cell, from the oof_* folders in the untouched matrix tree."""
    cells = []
    for p in sorted(C.SPATIAL_MATRIX.glob("*/oof_*")):
        if (p / "pooled_oof_metrics.json").is_file():
            cells.append(p.name[len("oof_"):])
    return sorted(cells)


def cell_fold_dirs(cell: str) -> dict[int, str]:
    out = {}
    for f in range(C.NFOLDS):
        hits = list(C.SPATIAL_MATRIX.glob(f"*/{cell}_fold{f}/models/example_dataset"))
        if len(hits) != 1:
            raise RuntimeError(f"expected exactly one prediction dir for {cell} fold {f}; "
                               f"found {[str(h) for h in hits]}")
        out[f] = str(hits[0])
    return out


def _init(celldirs):
    global _CELLDIRS
    _CELLDIRS = celldirs


def worker(task):
    """One tile, all cells. Returns a (n_cells, 121) matrix of pixel counts."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    fname, fold = task
    try:
        label = C.read_label(C.LABEL_DIR / fname).astype(np.int64).ravel()
        n = C.NCLASS
        out = np.zeros((len(_CELLDIRS), n * n), dtype=np.int64)
        base = label * n
        for ci, dirs in enumerate(_CELLDIRS):
            pred = np.asarray(Image.open(Path(dirs[fold]) / fname), dtype=np.int64).ravel()
            if pred.shape != label.shape:
                return ("FAIL", fname, f"shape mismatch in cell {ci}")
            if pred.max() >= n:
                return ("FAIL", fname, f"prediction class {pred.max()} out of range")
            out[ci] = np.bincount(base + pred, minlength=n * n)
        return ("OK", fname, out)
    except Exception as exc:                       # noqa: BLE001
        return ("FAIL", fname, f"{type(exc).__name__}: {exc}")


def validate_against_pooled(cells, routes, cms) -> bool:
    """Sum each cell's route matrices and compare to the existing pooled matrix, exactly."""
    C.banner("VALIDATION vs the existing pooled_oof_metrics.json")
    ok = True
    for ci, cell in enumerate(cells):
        hits = list(C.SPATIAL_MATRIX.glob(f"*/oof_{cell}/pooled_oof_metrics.json"))
        if not hits:
            print(f"{cell:<34} no pooled file found -- cannot validate")
            ok = False
            continue
        with open(hits[0]) as fh:
            ref = np.array(json.load(fh)["global_confusion_matrix"], dtype=np.int64)
        got = cms[ci].sum(axis=0).reshape(C.NCLASS, C.NCLASS)
        same = np.array_equal(got, ref)
        ok &= same
        if same:
            print(f"{cell:<34} EXACT MATCH  ({got.sum():,} px)")
        else:
            diff = int(np.abs(got - ref).sum())
            print(f"{cell:<34} MISMATCH  total |diff| = {diff:,}  "
                  f"(ours {got.sum():,} vs ref {ref.sum():,})")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="E6 -- per-route out-of-fold confusion matrices")
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell names; default is all 24")
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    C.ensure_out_dirs()
    cells = args.cells.split(",") if args.cells else discover_cells()
    if not cells:
        sys.exit(f"no scored cells found under {C.SPATIAL_MATRIX}")
    celldirs = [cell_fold_dirs(c) for c in cells]

    df = C.load_tile_inventory()
    if args.limit:
        df = df.head(args.limit)
    routes = sorted(df["route"].unique().tolist())
    ridx = {r: i for i, r in enumerate(routes)}

    C.banner(f"E6 -- {len(cells)} cells x {len(routes)} routes over {len(df):,} tiles")
    for c in cells:
        print(f"  {c}")
    print(f"\nreads about {len(df) * (len(cells) + 1) / 1000:.0f} GB; tile-outer/cell-inner so")
    print(f"each label is read once. {args.procs} processes.\n")

    cms = np.zeros((len(cells), len(routes), C.NCLASS * C.NCLASS), dtype=np.int64)
    tasks = [(r.filename, int(r.fold)) for r in df.itertuples()]
    route_of = dict(zip(df["filename"], df["route"]))
    fails = []

    with Pool(processes=args.procs, initializer=_init, initargs=(celldirs,)) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, tasks, chunksize=4), 1):
            if res[0] != "OK":
                fails.append((res[1], res[2]))
                continue
            cms[:, ridx[route_of[res[1]]], :] += res[2]
            if i % 500 == 0:
                print(f"  {i:,} / {len(tasks):,}")

    if fails:
        print(f"\n{len(fails)} tiles failed:")
        for f, e in fails[:20]:
            print(f"  {f}: {e}")

    cms4 = cms.reshape(len(cells), len(routes), C.NCLASS, C.NCLASS)
    ok = validate_against_pooled(cells, routes, cms) if not args.limit else False
    if args.limit:
        print("(--limit used, so the pooled comparison is skipped)")

    # ------------------------------------------------------------------------------------------
    C.banner("PER-ROUTE PERFORMANCE")
    rows = []
    for ci, cell in enumerate(cells):
        for rj, route in enumerate(routes):
            m = C.metrics_from_confusion(cms4[ci, rj], C.CODES, ignore_index=C.IGNORE_INDEX,
                                         report_only=(C.UNKNOWN2_INDEX,),
                                         split_tag=f"{cell}|{route}")
            rec = {"cell": cell, "route": route,
                   "tiles": int((df["route"] == route).sum()),
                   "scored_px": int(m["evaluated_pixels"]),
                   "macro_iou_present_classes": m["macro_iou"],
                   "macro_f1_present_classes": m["macro_f1"],
                   "overall_accuracy": m["overall_accuracy"],
                   "n_classes_present": m["n_macro_classes_evaluated"]}
            for name in C.CODES:
                rec[f"iou_{name}"] = m["per_class"][name]["iou"] if name != "unknown" else None
            rows.append(rec)

    # spread across routes, per cell
    print(f"{'cell':<34}{'median':>9}{'min':>9}{'max':>9}{'spread':>9}   route-level macro-IoU")
    summary = []
    for cell in cells:
        vals = [r["macro_iou_present_classes"] for r in rows
                if r["cell"] == cell and r["macro_iou_present_classes"] is not None]
        if not vals:
            continue
        v = np.array(vals)
        print(f"{cell:<34}{np.median(v):>9.4f}{v.min():>9.4f}{v.max():>9.4f}"
              f"{v.max() - v.min():>9.4f}")
        summary.append({"cell": cell, "route_median_macro_iou": float(np.median(v)),
                        "route_min": float(v.min()), "route_max": float(v.max()),
                        "route_spread": float(v.max() - v.min()),
                        "route_std": float(v.std(ddof=1)) if len(v) > 1 else None,
                        "n_routes_scored": int(len(v))})

    print("\nThe spread is the point. A wide range across routes means the pooled headline is an")
    print("average over places that behave very differently, which is precisely what a single")
    print("national accuracy figure conceals.")

    # ------------------------------------------------------------------------------------------
    if len(cells) > 1:
        C.banner("WHICH ROUTES ARE HARD  (averaged over all cells)")
        print(f"{'route':<9}{'fold':>5}{'tiles':>8}{'mean macro-IoU':>17}{'mean acc':>11}"
              f"{'classes':>9}")
        for route in routes:
            rr = [r for r in rows if r["route"] == route
                  and r["macro_iou_present_classes"] is not None]
            if not rr:
                continue
            fold = int(df.loc[df["route"] == route, "fold"].iloc[0])
            print(f"{route:<9}{fold:>5}{rr[0]['tiles']:>8}"
                  f"{np.mean([r['macro_iou_present_classes'] for r in rr]):>17.4f}"
                  f"{np.mean([r['overall_accuracy'] for r in rr]):>11.4f}"
                  f"{rr[0]['n_classes_present']:>9}")
        print("\nRoute-level scores are averaged over the classes PRESENT in that route, so routes")
        print("are not directly comparable to each other; they are comparable BETWEEN MODELS,")
        print("paired route by route, which is exactly what the Wilcoxon test needs.")

    # ------------------------------------------------------------------------------------------
    C.banner("WRITING ARTIFACTS")
    npz = C.assert_writes_are_local(C.TABLES / "route_cell_cms.npz")
    np.savez_compressed(npz, cms=cms4, cells=np.array(cells), routes=np.array(routes),
                        codes=np.array(C.CODES))
    print(f"wrote {npz}  ({npz.stat().st_size / 1e3:.0f} KB)  shape {cms4.shape}")
    print("  -> this is the durable asset: Part B's route-level Wilcoxon, Holm-Bonferroni and")
    print("     block-bootstrap all read from here instead of the 463,536 prediction tiles.")

    mcsv = C.assert_writes_are_local(C.TABLES / "route_cell_metrics.csv")
    with open(mcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {mcsv}  ({len(rows)} rows)")

    # cross-cell summary -- the table the 2026-07-28 handoff section 9.3 flags as missing
    cross = []
    for cell in cells:
        hits = list(C.SPATIAL_MATRIX.glob(f"*/oof_{cell}/pooled_oof_metrics.json"))
        rec = {"cell": cell}
        if hits:
            with open(hits[0]) as fh:
                d = json.load(fh)
            h = d["headline_pooled_oof"]
            rec.update({"macro_iou": h["macro_iou"], "macro_f1": h["macro_f1"],
                        "overall_accuracy": h["overall_accuracy"],
                        "evaluated_pixels": h["evaluated_pixels"]})
            for name in C.CODES:
                rec[f"iou_{name}"] = h["per_class"].get(name, {}).get("iou")
            fa = d.get("fold_aggregate_diagnostic", {})
            rec["fold_macro_iou_mean"] = fa.get("macro_iou", {}).get("mean")
            rec["fold_macro_iou_std"] = fa.get("macro_iou", {}).get("std")
        s = next((x for x in summary if x["cell"] == cell), {})
        rec.update({k: v for k, v in s.items() if k != "cell"})
        cross.append(rec)
    ccsv = C.assert_writes_are_local(C.TABLES / "cross_cell_summary.csv")
    with open(ccsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cross[0].keys()))
        w.writeheader(); w.writerows(cross)
    print(f"wrote {ccsv}  ({len(cross)} cells)")

    C.write_json({"generated_utc": datetime.now(timezone.utc).isoformat(),
                  "cells": cells, "routes": routes, "n_tiles": int(len(df)),
                  "validated_against_pooled": bool(ok), "failures": fails[:20],
                  "per_cell_route_spread": summary},
                 C.TABLES / "route_cell_provenance.json")

    if not args.limit and not ok:
        sys.exit("\nVALIDATION FAILED -- the route matrices do not reproduce the pooled matrices")
    if ok:
        print("\nVALIDATION PASSED -- route matrices sum exactly to the existing pooled matrices")


def selftest() -> None:
    C.banner("E6 selftest (synthetic, no files touched)")
    n = C.NCLASS

    # The accumulation identity the real validation relies on: per-route matrices built from
    # disjoint tile sets must sum to the matrix built from all tiles at once.
    rng = np.random.default_rng(3)
    label = rng.integers(0, n, size=4000)
    pred = rng.integers(0, n, size=4000)
    route = rng.integers(0, 4, size=4000)

    total = np.bincount(label * n + pred, minlength=n * n)
    per_route = np.zeros((4, n * n), dtype=np.int64)
    for r in range(4):
        m = route == r
        per_route[r] = np.bincount(label[m] * n + pred[m], minlength=n * n)
    assert np.array_equal(per_route.sum(axis=0), total)
    print("route decomposition : OK (disjoint routes sum to the pooled matrix exactly)")

    cm = total.reshape(n, n)
    m1 = C.metrics_from_confusion(cm, C.CODES, ignore_index=C.IGNORE_INDEX,
                                  report_only=(C.UNKNOWN2_INDEX,))
    assert m1["macro_iou"] is not None
    print(f"metric path         : OK (macro-IoU {m1['macro_iou']:.4f} from the pooled matrix)")

    # A route lacking a class must exclude it from that route's macro rather than score it 0.
    sparse = np.zeros((n, n), dtype=np.int64)
    sparse[4, 4] = 900
    sparse[1, 1] = 100
    m2 = C.metrics_from_confusion(sparse, C.CODES, ignore_index=C.IGNORE_INDEX,
                                  report_only=(C.UNKNOWN2_INDEX,))
    assert m2["n_macro_classes_evaluated"] == 2, m2["n_macro_classes_evaluated"]
    assert abs(m2["macro_iou"] - 1.0) < 1e-12
    print("absent-class rule   : OK (a route scores only the classes it actually contains,")
    print("                      so small routes are not punished for missing classes)")

    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
