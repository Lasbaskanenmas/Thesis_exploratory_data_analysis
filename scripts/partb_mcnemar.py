#!/usr/bin/env python
"""
Part B, McNemar on pooled out-of-fold pixels -- CONTEXT ONLY, for the ten declared family pairs.

The locked declaration (section 4) specifies this and simultaneously refuses to interpret it:
"reported as context -- discordant-pixel direction and magnitude -- with its p-value explicitly not
interpreted (n ~ 12.2 billion spatially autocorrelated pixels makes it near-automatic by
construction)". This module computes exactly that and prints the disclaimer beside every row, so
the number cannot be quoted out of its declared role.

WHY IT NEEDS ITS OWN PASS. McNemar needs per-pixel AGREEMENT between two cells -- how often A is
right where B is wrong. A confusion matrix marginalises that away, so `route_cell_cms.npz` cannot
supply it. This streams the prediction rasters instead: for each tile, the label plus the seven
distinct cells that appear in the declared pairs, all read from the fold in which that tile was
held out.

The ignore convention is the frozen one: pixels whose LABEL is the ignore index are dropped; a
prediction of the ignore index on a real pixel counts as wrong.

Worker-capped by default so the training dataloaders are not starved.

    python partb_mcnemar.py --declaration ..\\2026-08-25_pre_declarations.md [--procs 8] [--limit N]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402
import partb_statistics as PB  # noqa: E402

OUT_DIR = C.TABLES / "part_b"
_CELLDIRS: dict[str, dict[int, str]] = {}


def _init(celldirs):
    global _CELLDIRS
    _CELLDIRS = celldirs


def worker(task):
    """-> (status, name, correct_matrix) with correct_matrix shape (n_cells, n_scored_px) as bits."""
    import rasterio
    name, fold = task
    try:
        with rasterio.open(Path(C.LABEL_DIR) / name) as src:
            lab = src.read(1)
        valid = lab != C.IGNORE_INDEX
        n_valid = int(valid.sum())
        if n_valid == 0:
            return ("EMPTY", name, None)
        labv = lab[valid]
        out = np.zeros((len(_CELLDIRS), n_valid), dtype=bool)
        for i, (_cell, dirs) in enumerate(sorted(_CELLDIRS.items())):
            with rasterio.open(Path(dirs[fold]) / name) as src:
                pred = src.read(1)
            out[i] = pred[valid] == labv
        return ("OK", name, out)
    except Exception as exc:                                   # noqa: BLE001
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(description="McNemar, context only, declared family pairs")
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dec = PB.load_declaration(args.declaration)          # refuses unless LOCKED
    D = PB.normalised(dec)
    pairs = [(a, b) for pairs_ in D["families"].values() for a, b in pairs_]
    cells = sorted({c for p in pairs for c in p})
    print(f"declaration {dec.get('declaration_version')} status {dec.get('status')}")
    print(f"{len(pairs)} declared family pairs over {len(cells)} distinct cells")

    # cell -> fold -> prediction directory
    celldirs = {}
    for cell in cells:
        hit = list(C.SPATIAL_MATRIX.glob(f"*/oof_{cell}"))
        if not hit:
            sys.exit(f"no scored cell folder for {cell}")
        model_dir = hit[0].parent
        dirs = {}
        for f in range(3):
            d = model_dir / f"{cell}_fold{f}" / "models" / "example_dataset"
            if not d.is_dir():
                sys.exit(f"missing predictions: {d}")
            dirs[f] = str(d)
        celldirs[cell] = dirs
    order = sorted(celldirs)
    ci = {c: i for i, c in enumerate(order)}

    r2f = C.load_fold_assignment(str(C.FOLD_ASSIGNMENT_CSV))
    names = C.tile_names()
    if args.limit:
        names = names[:args.limit]
    tasks = [(n, r2f[C.parse_route(n)]) for n in names]
    print(f"streaming {len(tasks):,} tiles x {len(order) + 1} rasters, {args.procs} processes\n")

    # b[i][j] = pixels where cell i is correct and cell j is wrong
    b = np.zeros((len(order), len(order)), dtype=np.int64)
    n_px = 0
    fails, empty = [], 0

    with Pool(args.procs, initializer=_init, initargs=(celldirs,)) as pool:
        for k, (status, name, payload) in enumerate(
                pool.imap_unordered(worker, tasks, chunksize=8), 1):
            if status == "FAIL":
                fails.append((name, payload))
                continue
            if status == "EMPTY":
                empty += 1
                continue
            n_px += payload.shape[1]
            for i in range(len(order)):
                ok_i = payload[i]
                for j in range(len(order)):
                    if i != j:
                        b[i, j] += int(np.count_nonzero(ok_i & ~payload[j]))
            if k % 2000 == 0:
                print(f"  {k:,} / {len(tasks):,}")

    print(f"\nscored pixels: {n_px:,}   tiles with no scored pixel: {empty}   failures: {len(fails)}")
    for nm, err in fails[:5]:
        print(f"  FAIL {nm}: {err}")

    rows = []
    print(f"\n{'pair':<44}{'b (A right,B wrong)':>21}{'c (A wrong,B right)':>21}"
          f"{'chi2':>12}{'p':>12}")
    for a, bb in pairs:
        i, j = ci[a], ci[bb]
        m = PB.mcnemar(int(b[i, j]), int(b[j, i]))
        rows.append({"cell_a": a, "cell_b": bb, "b_a_right_b_wrong": int(b[i, j]),
                     "c_a_wrong_b_right": int(b[j, i]),
                     "discordant_total": int(b[i, j] + b[j, i]),
                     "favours": a if b[i, j] > b[j, i] else bb,
                     "chi2": m["chi2"], "p_chi2": m["p_chi2"],
                     "scored_pixels": n_px,
                     "role": "CONTEXT ONLY -- p-value not interpreted, per declaration section 4"})
        short = f"{a} - {bb}".replace("convnext_upernet", "cnx").replace("unet_resnet34", "rn34") \
                             .replace("swin_upernet", "swin").replace("segformer_b1", "segf")
        print(f"{short:<44}{b[i, j]:>21,}{b[j, i]:>21,}{m['chi2']:>12.1f}{m['p_chi2']:>12.3e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = C.assert_writes_are_local(OUT_DIR / "mcnemar_family_pairs.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    prov = C.assert_writes_are_local(OUT_DIR / "mcnemar_provenance.json")
    prov.write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "declaration_version": str(dec.get("declaration_version")),
        "declaration_status": dec.get("status"),
        "cells": order, "n_pairs": len(pairs), "scored_pixels": int(n_px),
        "tiles": len(tasks), "tiles_empty": empty, "failures": fails[:20],
        "ignore_convention": "label==ignore_index dropped; predicting ignore on a real pixel is wrong",
        "role": "context only; the declaration explicitly does not interpret the p-value",
    }, indent=2), encoding="utf-8")
    print(f"wrote {prov}")
    print("\nREAD AS CONTEXT ONLY. At this pixel count with spatial autocorrelation, significance")
    print("is near-automatic by construction; the direction and the discordant magnitude are the")
    print("only parts of this table the thesis uses.")


if __name__ == "__main__":
    main()
