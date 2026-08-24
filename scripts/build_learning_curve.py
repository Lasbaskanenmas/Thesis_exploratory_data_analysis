#!/usr/bin/env python
"""
Great Plan 3.0 §4, foundational analysis 4: the data-quantity ablation / learning curve.

resnet34+UNet (the production architecture), weighted CE, the frozen spatial protocol, trained on
nested whole-route subsets of the training pool, evaluated every time on the SAME held-out fold.

DESIGN DECISIONS, and why:

  * Channel config = **rgb**. Plan 3.0 §4 says "the production model (resnet34+UNet)" and does NOT
    name a channel config. rgb is chosen because (a) it is the only resnet cell untouched by all
    three elevation defects -- placeholder constants, the uint8 truncation, and the 671 misregistered
    DSM/DTM tiles -- so the curve measures data volume rather than channel damage, and (b) it gives a
    frozen 100% anchor in `unet_resnet34_rgb`.

  * Held-out fold = **0**. Measured, not assumed. Fold 0's training pool has 13 routes with the
    largest at 28.2%, the only fold that supports a 25/50/75 ladder by whole routes (folds 1 and 2
    are both dominated by route 84-40 at 40.9%, which makes a 25% point unreachable). Fold 0's
    held-out set also carries all 9 predicted classes, so per-class curves are measurable.
    Trade-off stated honestly: `unet_resnet34_rgb` scores 0.3292 on fold 0 against a pooled 0.2695,
    so the curve sits above the headline level. That is fine -- a learning curve is read for SHAPE,
    and every point here uses an identical held-out set.

  * Subsets are nested PREFIXES of the training routes sorted by descending tile count. Deterministic,
    reproducible, nesting guaranteed by construction, and the biggest (most class-rich) routes enter
    first so the small subsets are not degenerate.

MECHANISM: the trainer computes train = path_to_all_txt - path_to_valid_txt. So each curve point gets
its own `all_*.txt` containing (subset training tiles) + (the fold-0 held-out tiles), with
path_to_valid_txt unchanged. The held-out set is therefore byte-identical at every point.

ADDITIVE. Writes only under configs/matrix_configs/ and never overwrites an existing file.
logs_and_models/ is read only.

    python build_learning_curve.py            # dry run: subsets, nesting, class coverage
    python build_learning_curve.py --write    # also emit the tile lists and configs
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

REPO = Path(r"c:\thesis\ML_sdfi_fastai2")
sys.path.insert(0, str(REPO / "src" / "ML_sdfi_fastai2" / "analyse"))
import generate_matrix_configs as G  # noqa: E402  (reuse the exact frozen config templates)

HELD_OUT_FOLD = 0
CHANNEL = "rgb"
MODEL_KEY = "resnet"
TARGETS = [0.25, 0.50, 0.75]

LC_DIR = REPO / "configs" / "matrix_configs" / "learning_curve"
OUT_TRAIN = REPO / "configs" / "matrix_configs" / "train"
OUT_INFER = REPO / "configs" / "matrix_configs" / "infer"
# path the configs will use, relative to c:\thesis\ML_sdfi_fastai2
LC_REL = "configs/matrix_configs/learning_curve"


def load_routes():
    tiles_by_route = {}
    for name in C.tile_names():
        tiles_by_route.setdefault(C.parse_filename(name)["route"], []).append(name)
    route_fold = C.load_fold_assignment(str(C.FOLD_ASSIGNMENT_CSV))
    return tiles_by_route, route_fold


def class_coverage(tile_names):
    """Which of the 9 predicted classes appear in this tile set? Uses the existing route audit."""
    with open(C.CLASS_ROUTES_JSON) as fh:
        cr = json.load(fh)["class_routes"]
    routes = {C.parse_filename(t)["route"] for t in tile_names}
    return {cls: bool(set(v["any_pixel"]) & routes) for cls, v in cr.items()}


def build():
    tiles_by_route, route_fold = load_routes()
    held_routes = sorted(r for r, f in route_fold.items() if f == HELD_OUT_FOLD)
    train_routes = sorted((r for r, f in route_fold.items() if f != HELD_OUT_FOLD),
                          key=lambda r: -len(tiles_by_route[r]))
    held_tiles = [t for r in held_routes for t in tiles_by_route[r]]
    pool = sum(len(tiles_by_route[r]) for r in train_routes)

    C.banner(f"Learning curve -- resnet34+UNet, {CHANNEL}, held-out fold {HELD_OUT_FOLD}")
    print(f"  held-out routes  : {held_routes}  ({len(held_tiles):,} tiles)")
    print(f"  training pool    : {len(train_routes)} routes, {pool:,} tiles")
    print(f"  routes descending: " + ", ".join(f"{r}({len(tiles_by_route[r])})" for r in train_routes))

    # Subset selection: exhaustive over the 2^13 subsets of the training pool, chosen sequentially so
    # each subset contains the previous one. Ranked by (class coverage DESC, |fraction - target| ASC).
    #
    # Coverage is the primary key deliberately. Plan 3.0 section 4 wants to test whether rare classes
    # stay near zero "regardless of quantity" -- that question is only answerable if the rare classes
    # are PRESENT in training at every point. Size-ordered prefixes leave solceller absent until 84%
    # of the pool, which would measure class presence rather than data volume.
    with open(C.CLASS_ROUTES_JSON) as fh:
        class_routes = json.load(fh)["class_routes"]
    n_classes = len(class_routes)

    def coverage_of(routes):
        rs = set(routes)
        return sum(1 for v in class_routes.values() if set(v["any_pixel"]) & rs)

    points, must_contain = [], set()
    for target in TARGETS:
        best = None
        for mask in range(1, 1 << len(train_routes)):
            routes = [train_routes[i] for i in range(len(train_routes)) if mask >> i & 1]
            if not must_contain <= set(routes):
                continue
            n = sum(len(tiles_by_route[r]) for r in routes)
            key = (-coverage_of(routes), abs(n / pool - target), len(routes))
            if best is None or key < best[0]:
                best = (key, routes, n)
        _key, routes, n = best
        must_contain = set(routes)
        tiles = [t for r in routes for t in tiles_by_route[r]]
        points.append({"target": target, "k": len(routes), "routes": sorted(routes),
                       "tiles": tiles, "n": n, "frac": n / pool,
                       "coverage": coverage_of(routes), "n_classes": n_classes})

    print(f"\n  {'point':<8}{'routes':>8}{'tiles':>9}{'actual':>9}{'target':>9}   route list")
    for p in points:
        print(f"  lc{int(p['target']*100):<6}{p['k']:>8}{p['n']:>9,}{100*p['frac']:>8.1f}%"
              f"{100*p['target']:>8.0f}%   {', '.join(p['routes'])}")
    print(f"  {'lc100':<8}{len(train_routes):>8}{pool:>9,}{100.0:>8.1f}%{100:>8.0f}%   "
          f"(frozen cell unet_resnet34_rgb, fold {HELD_OUT_FOLD} diagnostic)")

    # nesting proof
    print("\n  nesting check:")
    ok = True
    for a, b in zip(points, points[1:]):
        sub = set(a["routes"]) <= set(b["routes"])
        ok &= sub
        print(f"    lc{int(a['target']*100)} routes subset of lc{int(b['target']*100)} : {sub}")
    all_in_pool = all(set(p["routes"]) <= set(train_routes) for p in points)
    print(f"    every subset within the fold-{HELD_OUT_FOLD} training pool : {all_in_pool}")
    print(f"    no held-out route in any subset : "
          f"{all(not (set(p['routes']) & set(held_routes)) for p in points)}")
    if not (ok and all_in_pool):
        sys.exit("NESTING CHECK FAILED")

    print("\n  class coverage (9 predicted classes):")
    for p in points:
        cov = class_coverage(p["tiles"])
        missing = [c for c, v in cov.items() if not v]
        print(f"    lc{int(p['target']*100):<4} {sum(cov.values())}/9 present"
              + (f"   MISSING: {', '.join(missing)}" if missing else "   all present"))
    return points, held_tiles, held_routes


def emit(points, held_tiles):
    model_str, exp_dirname, prefix, trainer = G.MODELS[MODEL_KEY]
    dtypes, chans, means, stds = G.CHANNELS[CHANNEL]
    exp_root = f"{G.MATRIX_ROOT}/{exp_dirname}"
    valid_txt = f"{G.SPLIT_DIR}/fold_{HELD_OUT_FOLD}_valid.txt"

    LC_DIR.mkdir(parents=True, exist_ok=True)
    written, refused = [], []

    for p in points:
        tag = f"lc{int(p['target']*100)}"
        job = f"{prefix}_{CHANNEL}_{tag}_fold{HELD_OUT_FOLD}"
        infer_job = f"infer_{job}"

        # the pool this point sees: its training subset + the untouched held-out fold
        list_path = LC_DIR / f"all_{tag}_fold{HELD_OUT_FOLD}.txt"
        body = "\n".join(sorted(p["tiles"]) + sorted(held_tiles)) + "\n"
        for path, text in ((list_path, body),):
            if path.exists():
                refused.append(path); continue
            path.write_text(text, encoding="utf8")
            written.append(path)

        train_ini = G.train_config(model_str, dtypes, chans, means, stds, valid_txt,
                                   exp_root, job, G.CLASS_WEIGHTS)
        # point path_to_all_txt at this curve point's pool instead of the full all.txt
        old = f"path_to_all_txt = {G.DATA}/data/all.txt"
        new = f"path_to_all_txt = ../ML_sdfi_fastai2/{LC_REL}/{list_path.name}"
        assert old in train_ini, "template changed: path_to_all_txt line not found"
        train_ini = train_ini.replace(old, new)
        train_ini = ("#LEARNING CURVE point %s (Plan 3.0 section 4). Training pool = %d routes, "
                     "%d tiles (%.1f%% of the fold-%d training pool).\n"
                     "#Held-out fold %d is UNCHANGED and identical at every curve point.\n"
                     % (tag, p["k"], p["n"], 100 * p["frac"], HELD_OUT_FOLD, HELD_OUT_FOLD)) + train_ini

        infer_ini = G.infer_config(model_str, dtypes, chans, means, stds, valid_txt,
                                   exp_root, job, infer_job)

        for path, text in ((OUT_TRAIN / f"{job}.ini", train_ini),
                           (OUT_INFER / f"{infer_job}.ini", infer_ini)):
            if path.exists():
                refused.append(path); continue
            path.write_text(text, encoding="utf8")
            written.append(path)

    if refused:
        print("\nREFUSED to overwrite:")
        for r in refused:
            print("   " + str(r))
        sys.exit("aborting rather than overwriting")

    print(f"\n  wrote {len(written)} files:")
    for w in written:
        print(f"    {w}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    points, held_tiles, _held_routes = build()
    if args.write:
        emit(points, held_tiles)
    else:
        print("\n  dry run -- pass --write to emit tile lists and configs")


if __name__ == "__main__":
    main()
