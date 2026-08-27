#!/usr/bin/env python
"""
Task 2.6 of the 2026-08-24 work order / Great Plan 3.1 section 5.4: Boundary IoU.

WHY
    Three of the weakest classes -- brosten, fliser, betonflade -- are thin, edge-dominated paved
    surfaces. Mask IoU cannot tell "wrong class" from "right class, sloppy edge" on them, and E4 has
    just measured a 42 percent error rate ON annotation boundaries against 2.4 percent at 50 px or
    more. Boundary IoU separates the two, which is what gap G3.2 asks for and what the qualitative
    taxonomy's boundary-bleeding mode needs a number beside.

DEFINITION (Cheng, Girshick, Dollar, Berg, Kirillov, CVPR 2021)
    For a binary mask M, let Md be the set of pixels of M within distance d of M's own boundary,
    i.e. M minus its erosion by a disc of radius d. Then

        Boundary IoU(G, P) = |Gd AND Pd| / |Gd OR Pd|

    Both masks are restricted to their own boundary regions before intersecting. d is set relative
    to the image diagonal; the paper uses 2 percent, which on a 1000x1000 tile is 28 px (2.8 m at
    0.1 m GSD). `--d_px` overrides it and the value used is recorded in the output.

    NOTE ON THE CITATION: the definition above is implemented from the formula as stated in this
    project's own gap analysis (G3.2 / L4). The PDF is not on this machine, so the 2 percent
    diagonal convention and the erosion-by-disc detail must be checked against the paper on the
    reference-canon day (Plan 3.1 section 5.5) before prose attributes specifics to Cheng et al.
    The quantity computed here is well defined and reported either way.

SCOPE
    The four declared cells (Plan 3.1 section 5.4, decision C), no others:
        unet_resnet34_rgb, convnext_upernet_rgb, convnext_upernet_rgb_ndsm,
        convnext_upernet_6ch_corrected
    Scored on held-out predictions only, which is all these folders contain -- each fold directory
    holds exactly the tiles that fold held out, so the union over three folds is one out-of-fold
    pass over the pool, the same pixels the pooled metric uses.

    `--tiles N` scores a fixed-seed sample stratified by fold and reports the sample size beside
    every number; `--tiles 0` scores the full pool as an overnight batch. Boundary IoU is a
    per-class ratio of pixel counts, so counts accumulate across tiles and the sample estimates the
    same quantity, with sampling error that shrinks in N.

Read-only over logs_and_models/ and the dataset. Writes only under exploratory_data_analysis/.

    python boundary_iou.py --selftest
    python boundary_iou.py --tiles 600            # the sampled run
    python boundary_iou.py --tiles 0 --procs 16   # the full overnight batch
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

CELLS = [
    ("unet_resnet34", "unet_resnet34_rgb"),
    ("convnext_upernet", "convnext_upernet_rgb"),
    ("convnext_upernet", "convnext_upernet_rgb_ndsm"),
    ("convnext_upernet", "convnext_upernet_6ch_corrected"),
]
CLASS_ORDER = ["unknown", "asfalt", "fliser", "grus", "ubefestet", "green_roof",
               "drivhus", "betonflade", "brosten", "unknown2", "solceller"]
IGNORE_INDEX = 0
SCORED = [i for i, n in enumerate(CLASS_ORDER) if i != IGNORE_INDEX and n != "unknown2"]
DIAG_FRACTION = 0.02
SEED = 20260824


def boundary_region(mask, d, cache={}):
    """mask minus its erosion by a disc of radius d -- the pixels within d of the mask's boundary.

    Erosion is done with scipy's binary_erosion on a disc structuring element. The mask is padded
    with its own border value so that objects touching the tile edge are not given a spurious
    boundary there: a road running off the tile edge has no annotation boundary at the edge.
    """
    from scipy.ndimage import binary_erosion
    if not mask.any():
        return mask
    key = int(d)
    if key not in cache:
        r = np.arange(-d, d + 1)
        yy, xx = np.meshgrid(r, r, indexing="ij")
        cache[key] = (yy * yy + xx * xx) <= d * d
    eroded = binary_erosion(mask, structure=cache[key], border_value=1)
    return mask & ~eroded


def tile_counts(name, pred_dir, label_dir, d):
    """Per-class (intersection, union) of the two boundary regions, plus plain-mask counts."""
    import rasterio
    with rasterio.open(Path(label_dir) / name) as src:
        gt = src.read(1)
    with rasterio.open(Path(pred_dir) / name) as src:
        pr = src.read(1)
    if gt.shape != pr.shape:
        return ("SHAPE", name, gt.shape, pr.shape)

    valid = gt != IGNORE_INDEX                      # the ignore convention, unchanged
    out = np.zeros((len(CLASS_ORDER), 4), dtype=np.int64)   # b_inter, b_union, m_inter, m_union
    for c in SCORED:
        g = (gt == c)
        p = (pr == c) & valid                       # a prediction on an ignored pixel is not scored
        if not g.any() and not p.any():
            continue
        gb = boundary_region(g, d)
        pb = boundary_region(p, d)
        out[c, 0] = int((gb & pb).sum())
        out[c, 1] = int((gb | pb).sum())
        out[c, 2] = int((g & p).sum())
        out[c, 3] = int((g | p).sum())
    return ("OK", name, out, None)


def selftest():
    print("=" * 88)
    print("boundary_iou selftest (synthetic, no files touched)")
    print("=" * 88)
    d = 3

    # 1. a solid square thicker than 2d: its boundary region is a ring, not the whole square
    m = np.zeros((40, 40), dtype=bool)
    m[10:30, 10:30] = True
    br = boundary_region(m, d)
    assert br.sum() < m.sum(), (br.sum(), m.sum())
    assert not br[20, 20], "the interior of a thick object must not be in its boundary region"
    assert br[10, 15], "the edge of the object must be in its boundary region"
    ring = 20 * 20 - 14 * 14                       # 20x20 square eroded by radius 3 -> 14x14
    assert br.sum() == ring, (br.sum(), ring)
    print(f"  boundary region of a thick square is a ring ({ring} px) : OK")

    # 2. an object thinner than 2d is entirely boundary -- the brosten / fliser case
    thin = np.zeros((40, 40), dtype=bool)
    thin[20:23, 5:35] = True                       # 3 px wide, thinner than 2d = 6
    assert boundary_region(thin, d).sum() == thin.sum()
    print("  a thin object is entirely its own boundary : OK")

    # 3. identical masks -> Boundary IoU 1; disjoint masks -> 0
    gb, pb = boundary_region(m, d), boundary_region(m.copy(), d)
    assert (gb & pb).sum() / (gb | pb).sum() == 1.0
    other = np.zeros((40, 40), dtype=bool)
    other[0:5, 0:5] = True
    assert (gb & boundary_region(other, d)).sum() == 0
    print("  identical -> 1.0, disjoint -> 0.0 : OK")

    # 4. the discriminating case: a mask shifted by 1 px keeps almost all of its MASK IoU but
    #    loses a large share of its BOUNDARY IoU. This is the whole point of the metric.
    shifted = np.zeros((40, 40), dtype=bool)
    shifted[11:31, 10:30] = True                   # same square, one row down
    m_iou = (m & shifted).sum() / (m | shifted).sum()
    sb = boundary_region(shifted, d)
    b_iou = (gb & sb).sum() / (gb | sb).sum()
    assert m_iou > 0.9, m_iou
    assert b_iou < m_iou, (b_iou, m_iou)
    print(f"  1 px shift: mask IoU {m_iou:.4f} vs boundary IoU {b_iou:.4f}  (boundary is stricter) : OK")

    # 5. THE PROPERTY THE METRIC EXISTS FOR. Under the same 1 px boundary error, mask IoU improves
    #    as the object grows -- it dilutes the error against the interior -- while Boundary IoU
    #    stays put, because it only ever looks at a band of fixed width along the perimeter. That
    #    size-insensitivity is why Boundary IoU is the right instrument for thin paved surfaces.
    big = np.zeros((200, 200), dtype=bool)
    big[20:180, 20:180] = True
    big_s = np.zeros((200, 200), dtype=bool)
    big_s[21:181, 20:180] = True
    big_m_iou = (big & big_s).sum() / (big | big_s).sum()
    bb, bbs = boundary_region(big, d), boundary_region(big_s, d)
    big_b_iou = (bb & bbs).sum() / (bb | bbs).sum()
    assert big_m_iou > m_iou, (big_m_iou, m_iou)          # mask IoU rewards size
    assert abs(big_b_iou - b_iou) < 0.05, (big_b_iou, b_iou)   # boundary IoU does not
    print(f"  same 1 px error, 8x larger object: mask IoU {m_iou:.4f} -> {big_m_iou:.4f} "
          f"(dilutes), boundary IoU {b_iou:.4f} -> {big_b_iou:.4f} (holds) : OK")

    # 6. an object touching the tile edge must not gain a boundary there
    edge = np.zeros((40, 40), dtype=bool)
    edge[0:20, 0:20] = True
    be = boundary_region(edge, d)
    assert not be[0, 0], "the tile border is not an annotation boundary"
    assert be[19, 10] and be[10, 19], "the object's real edges are still boundary"
    print("  tile border is not treated as an object boundary : OK")
    print("\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description="Boundary IoU on the four declared cells")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tiles", type=int, default=600,
                    help="tiles per fold-sample; 0 = the full pool (overnight batch)")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--d_px", type=int, default=None,
                    help=f"boundary width; default {DIAG_FRACTION:.0%} of the 1000x1000 diagonal")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    d = args.d_px or int(round(DIAG_FRACTION * np.hypot(1000, 1000)))
    label_dir = C.LABEL_DIR
    r2f = C.load_fold_assignment(str(C.FOLD_ASSIGNMENT_CSV))

    by_fold = {0: [], 1: [], 2: []}
    for n in C.tile_names():
        by_fold[r2f[C.parse_route(n)]].append(n)

    rng = random.Random(SEED)
    if args.tiles:
        sample = []
        for f in sorted(by_fold):
            pool = sorted(by_fold[f])
            sample.append((f, rng.sample(pool, min(args.tiles, len(pool)))))
        mode = f"sample of {args.tiles} tiles per fold, seed {SEED}"
    else:
        sample = [(f, sorted(by_fold[f])) for f in sorted(by_fold)]
        mode = "full pool"
    n_total = sum(len(v) for _f, v in sample)
    print(f"Boundary IoU  d = {d} px ({d * 0.1:.1f} m at 0.1 m GSD)   {mode}   "
          f"{n_total:,} tiles x {len(CELLS)} cells")

    rows, per_cell = [], {}
    for model_dir, cell in CELLS:
        t0 = time.time()
        acc = np.zeros((len(CLASS_ORDER), 4), dtype=np.int64)
        n_ok, bad = 0, []
        for fold, names in sample:
            pred_dir = (C.SPATIAL_MATRIX / model_dir / f"{cell}_fold{fold}" /
                        "models" / "example_dataset")
            if not pred_dir.is_dir():
                sys.exit(f"missing predictions: {pred_dir}")
            fn = partial(tile_counts, pred_dir=str(pred_dir), label_dir=str(label_dir), d=d)
            with Pool(args.procs) as pool:
                for status, name, payload, extra in pool.imap_unordered(fn, names, chunksize=8):
                    if status != "OK":
                        bad.append((name, payload, extra))
                        continue
                    acc += payload
                    n_ok += 1
        per_cell[cell] = acc
        for c in SCORED:
            b_i, b_u, m_i, m_u = acc[c]
            rows.append({
                "cell": cell, "class": CLASS_ORDER[c], "index": c,
                "boundary_iou": (b_i / b_u) if b_u else None,
                "mask_iou_sampled": (m_i / m_u) if m_u else None,
                "boundary_intersection_px": int(b_i), "boundary_union_px": int(b_u),
                "mask_intersection_px": int(m_i), "mask_union_px": int(m_u),
                "tiles_scored": n_ok, "d_px": d,
            })
        print(f"  {cell:<34} {n_ok:>6} tiles  {time.time() - t0:>7.1f} s"
              + (f"   {len(bad)} FAILED" if bad else ""))
        if bad:
            for b in bad[:5]:
                print(f"      {b}")

    out = C.assert_writes_are_local(C.TABLES / "boundary_iou.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} rows)")

    prov = C.assert_writes_are_local(C.TABLES / "boundary_iou_provenance.json")
    prov.write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "definition": "Cheng et al. 2021: |Gd AND Pd| / |Gd OR Pd|, Md = M minus erosion(M, disc d)",
        "citation_status": "formula implemented from the project gap analysis (G3.2/L4); the paper "
                           "itself is not on this machine -- verify d convention on the "
                           "reference-canon day before attributing specifics",
        "d_px": d, "d_metres": d * 0.1, "diag_fraction": DIAG_FRACTION,
        "mode": mode, "seed": SEED, "tiles_scored_per_cell": n_total,
        "cells": [c for _m, c in CELLS],
        "ignore_index": IGNORE_INDEX,
        "note": "mask_iou_sampled is the plain IoU over the SAME tiles, so the two columns are "
                "directly comparable; on a sample it will not equal the pooled figure.",
    }, indent=2), encoding="utf-8")
    print(f"wrote {prov}")

    print(f"\n=== Boundary IoU vs mask IoU (same tiles, d = {d} px) ===")
    classes = [CLASS_ORDER[c] for c in SCORED]
    short = {"unet_resnet34_rgb": "rn34_rgb", "convnext_upernet_rgb": "cnx_rgb",
             "convnext_upernet_rgb_ndsm": "cnx_ndsm",
             "convnext_upernet_6ch_corrected": "cnx_6chcorr"}
    print(f"  {'class':<12}" + "".join(f"{short.get(c, c[:11]):>13}" for _m, c in CELLS))
    for cls in classes:
        line = f"  {cls:<12}"
        for _m, cell in CELLS:
            r = next(x for x in rows if x["cell"] == cell and x["class"] == cls)
            bi = r["boundary_iou"]
            mi = r["mask_iou_sampled"]
            line += f"{(f'{bi:.3f}/{mi:.3f}' if bi is not None else '-/-'):>13}"
        print(line)
    print("  (boundary IoU / mask IoU on the same tiles; boundary is stricter by construction)")


if __name__ == "__main__":
    main()
