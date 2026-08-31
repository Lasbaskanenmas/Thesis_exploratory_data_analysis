#!/usr/bin/env python
"""
WO-2 -- build the pre-registered qualitative error-analysis pack (locked declaration D4).

EXECUTES D4 EXACTLY. Nothing here is a design choice; every number, stratum, cell and taxonomy mode
comes from `2026-08-25_pre_declarations.md` section 5, which is LOCKED. The one place this module
makes a judgement is the panel layout, which the declaration leaves open, and that judgement is
described in INSTRUCTIONS.md so it travels with the results.

ORDER OF OPERATIONS IS PART OF THE PROTOCOL. The tile list is drawn and written to disk BEFORE any
image is rendered, because D4 says so: a sample that could be redrawn after seeing a panel is not a
sample. The rendering step reads that file back rather than recomputing the draw.

SEED DISCIPLINE. D4 declares one seed, 20260825, "for every random step". A single
`random.Random(20260825)` is therefore used for all four draws in a fixed, documented order --
(1) the per-fold stratified sample, (2) the pass-1 item shuffle, (3) the double-scored tile subset,
(4) the pass-2 item shuffle. Deriving separate streams per step would have been tidier but would
have been a deviation, so it was not done. The order is recorded in the provenance JSON so the
whole pack is reproducible from the seed alone.

BLINDING. A panel carries an item id and nothing else: no tile name, no route, no cell name, in the
image or in the filename. The item -> (tile, cell) mapping goes to SEALED_item_key.csv, which the
author does not open until scoring is complete.

    python wo2_qualitative_pack.py --declaration ..\\2026-08-25_pre_declarations.md [--procs 4]
    python wo2_qualitative_pack.py --declaration ... --sample-only    # step 1 only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402
import partb_statistics as PB  # noqa: E402

PACK = C.EDA_ROOT / "results" / "qualitative_pack"
PANELS = PACK / "panels"
PANELS2 = PACK / "panels_pass2"
INVENTORY = C.TABLES / "tile_inventory.csv"

# D4 section 5 item 5, verbatim. Order is the declared order; the column names follow it.
MODES = [
    ("m1_speckle", "salt-and-pepper speckle (isolated single-class pixels/specks)"),
    ("m2_boundary_bleed", "boundary bleeding (class runs across a visible surface edge)"),
    ("m3_spurious_patch", "spurious minority-class patch (false-positive island of a class absent "
                          "or negligible in the tile's ground truth)"),
    ("m4_paved_swap", "paved-class swap (asfalt/fliser/grus/brosten/betonflade confusion over "
                      "paved ground)"),
    ("m5_road_continuation", "road/driveway continuation error (carriageway class extended into "
                             "entrances, or cut)"),
    ("m6_veg_occlusion", "vegetation-occlusion error (surface under canopy mislabelled)"),
    ("m7_misregistration", "misregistration echo (prediction pattern visibly offset from image "
                           "structure)"),
    ("m8_large_object", "large-object failure (solar farm, greenhouse, green roof missed, "
                        "fragmented or hallucinated)"),
    ("m9_other", "other (one-line free note)"),
]

# 11-class palette. Index 0 (unknown / ignore) is deliberately NOT painted: leaving it as raw
# imagery is the clearest possible signal of "no ground truth exists here", which matters because
# E4 measured 37 percent of the label mass as unannotated coverage rather than uncertainty.
PALETTE = {
    0: None,                    # unknown -> unpainted
    1: (55, 71, 92),            # asfalt        slate
    2: (255, 140, 0),           # fliser        orange
    3: (200, 170, 60),          # grus          ochre
    4: (60, 150, 70),           # ubefestet     green
    5: (150, 255, 60),          # green_roof    lime
    6: (0, 220, 220),           # drivhus       cyan
    7: (230, 60, 220),          # betonflade    magenta
    8: (220, 40, 40),           # brosten       red
    9: (255, 255, 255),         # unknown2      white (0 px in this pool)
    10: (60, 90, 240),          # solceller     blue
}
ALPHA = 0.45
GUTTER = 10
HEADER = 46


def weak_mask(row, weak_classes):
    return any(int(row[f"px_{c}"]) > 0 for c in weak_classes)


def draw_sample(dec_q, rng):
    """Step 1: the stratified per-fold draw. Returns rows in a stable, documented order."""
    with open(INVENTORY, newline="", encoding="utf-8") as fh:
        inv = list(csv.DictReader(fh))
    weak = list(dec_q["weak_classes"])
    per_fold = int(dec_q["per_fold"])
    n_weak = int(dec_q["strata"]["weak_present"])
    n_comp = int(dec_q["strata"]["complement"])
    assert n_weak + n_comp == per_fold, f"strata {n_weak}+{n_comp} != per_fold {per_fold}"

    picked = []
    print(f"weak-class stratum defined as any pixel of {weak}")
    for fold in range(C.NFOLDS):
        pool = [r for r in inv if int(r["fold"]) == fold]
        w = sorted([r for r in pool if weak_mask(r, weak)], key=lambda r: r["filename"])
        c = sorted([r for r in pool if not weak_mask(r, weak)], key=lambda r: r["filename"])
        print(f"  fold {fold}: {len(pool):,} held-out tiles -> weak-present {len(w):,}, "
              f"complement {len(c):,}")
        if len(w) < n_weak or len(c) < n_comp:
            sys.exit(f"fold {fold}: stratum too small for the declared draw")
        for stratum, k, tag in ((w, n_weak, "weak_present"), (c, n_comp, "complement")):
            for r in rng.sample(stratum, k):
                picked.append({"filename": r["filename"], "route": r["route"], "fold": fold,
                               "stratum": tag,
                               **{f"px_{cl}": r[f"px_{cl}"] for cl in weak},
                               "n_scored_px": r["n_scored_px"],
                               "n_classes_present": r["n_predicted_classes_present"]})
    return picked


def colourise(labels, base):
    """Alpha-blend class colours over the base imagery. Index 0 is left unpainted."""
    out = base.astype(np.float32).copy()
    for idx, col in PALETTE.items():
        if col is None:
            continue
        m = labels == idx
        if not m.any():
            continue
        for ch in range(3):
            out[..., ch][m] = (1 - ALPHA) * out[..., ch][m] + ALPHA * col[ch]
    return np.clip(out, 0, 255).astype(np.uint8)


def render_item(task, out_dir):
    """One blinded panel: [imagery | ground truth | prediction]. No identifying text anywhere."""
    import rasterio
    from PIL import Image, ImageDraw
    item_id, tile, pred_dir = task
    try:
        with rasterio.open(Path(C.SPLITTED_DIR) / "rgb" / tile) as src:
            rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
        with rasterio.open(Path(C.LABEL_DIR) / tile) as src:
            gt = src.read(1)
        with rasterio.open(Path(pred_dir) / tile) as src:
            pr = src.read(1)

        h, w = rgb.shape[:2]
        panels = [rgb, colourise(gt, rgb), colourise(pr, rgb)]
        canvas = Image.new("RGB", (w * 3 + GUTTER * 2, h + HEADER), (255, 255, 255))
        for i, arr in enumerate(panels):
            canvas.paste(Image.fromarray(arr), (i * (w + GUTTER), HEADER))
        d = ImageDraw.Draw(canvas)
        d.text((6, 8), f"ITEM {item_id}", fill=(0, 0, 0))
        for i, cap in enumerate(("IMAGERY", "GROUND TRUTH", "PREDICTION")):
            d.text((i * (w + GUTTER) + 6, 26), cap, fill=(0, 0, 0))
        canvas.save(Path(out_dir) / f"item_{item_id}.png", optimize=True)
        return ("OK", item_id, None)
    except Exception as exc:                                   # noqa: BLE001
        return ("FAIL", item_id, f"{type(exc).__name__}: {exc}")


def write_legend(path):
    from PIL import Image, ImageDraw
    names = {v: k for k, v in enumerate(C.CODES)}
    rows = [(i, C.CODES[i]) for i in range(len(C.CODES))]
    img = Image.new("RGB", (420, 26 * len(rows) + 40), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((10, 10), "class colours (overlay alpha 0.45)", fill=(0, 0, 0))
    for n, (i, nm) in enumerate(rows):
        y = 34 + n * 26
        col = PALETTE[i]
        if col is None:
            d.rectangle([10, y, 40, y + 18], outline=(0, 0, 0))
            d.text((48, y + 3), f"{i}  {nm}  (not painted - no ground truth)", fill=(0, 0, 0))
        else:
            d.rectangle([10, y, 40, y + 18], fill=col, outline=(0, 0, 0))
            d.text((48, y + 3), f"{i}  {nm}", fill=(0, 0, 0))
    img.save(path)
    _ = names


def main():
    ap = argparse.ArgumentParser(description="WO-2 qualitative sampling pack (declaration D4)")
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--sample-only", action="store_true")
    args = ap.parse_args()

    dec = PB.load_declaration(args.declaration)        # refuses unless status is LOCKED
    q = dec["qualitative"]
    print(f"declaration {dec.get('declaration_version')}  status {dec.get('status')}")
    print(f"D4: n={q['n_tiles']} tiles, {q['per_fold']}/fold, cells {q['cells']}, "
          f"seed {q['seed']}, {q['taxonomy_modes']} modes, "
          f"{q['double_scored_tiles']} double-scored\n")
    assert len(MODES) == int(q["taxonomy_modes"]), "taxonomy size differs from the declaration"

    PACK.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(q["seed"]))                # ONE stream, four documented draws

    # ---------------- step 1: the draw, persisted before anything is rendered ----------------
    picked = draw_sample(q, rng)                       # draw (1)
    assert len(picked) == int(q["n_tiles"]), f"{len(picked)} != declared {q['n_tiles']}"
    assert len({p["filename"] for p in picked}) == len(picked), "duplicate tile in the sample"

    sample_csv = C.assert_writes_are_local(PACK / "sample_tiles.csv")
    with open(sample_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(picked[0]))
        w.writeheader()
        w.writerows(picked)
    print(f"\nwrote {sample_csv}  ({len(picked)} tiles)  -- persisted BEFORE any rendering")
    if args.sample_only:
        return

    # ---------------- items, shuffled, blinded ----------------
    cells = list(q["cells"])
    celldir = {}
    for cell in cells:
        hit = list(C.SPATIAL_MATRIX.glob(f"*/oof_{cell}"))
        if not hit:
            sys.exit(f"no scored cell {cell}")
        celldir[cell] = {f: hit[0].parent / f"{cell}_fold{f}" / "models" / "example_dataset"
                         for f in range(C.NFOLDS)}
        for f, d_ in celldir[cell].items():
            if not d_.is_dir():
                sys.exit(f"missing predictions: {d_}")

    items = [{"tile": p["filename"], "fold": p["fold"], "cell": cell, "stratum": p["stratum"]}
             for p in picked for cell in cells]
    rng.shuffle(items)                                  # draw (2)
    for n, it in enumerate(items, 1):
        it["item_id"] = f"{n:04d}"

    adjacent = sum(1 for a, b in zip(items, items[1:]) if a["tile"] == b["tile"])
    print(f"{len(items)} items shuffled. Adjacent same-tile pairs: {adjacent} "
          f"(a plain shuffle is what D4 declares; reported, not engineered away)")

    key = C.assert_writes_are_local(PACK / "SEALED_item_key.csv")
    with open(key, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item_id", "tile", "cell", "fold", "stratum"])
        w.writeheader()
        w.writerows({k: it[k] for k in ("item_id", "tile", "cell", "fold", "stratum")}
                    for it in items)

    # ---------------- scoring sheets ----------------
    def sheet(path, item_ids, note):
        cols = ["item_id"] + [m for m, _ in MODES] + ["dominant_mode", "note"]
        with open(C.assert_writes_are_local(path), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for i in item_ids:
                w.writerow([i] + [""] * (len(cols) - 1))
        print(f"wrote {path}  ({len(item_ids)} rows)  {note}")

    sheet(PACK / "scoring_sheet.csv", [it["item_id"] for it in items], "pass 1")

    dbl_tiles = rng.sample(sorted({p["filename"] for p in picked}),
                           int(q["double_scored_tiles"]))          # draw (3)
    p2 = [dict(it) for it in items if it["tile"] in set(dbl_tiles)]
    rng.shuffle(p2)                                                 # draw (4)
    for n, it in enumerate(p2, 1):
        it["pass2_id"] = f"R{n:03d}"
    key2 = C.assert_writes_are_local(PACK / "SEALED_item_key_pass2.csv")
    with open(key2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pass2_id", "item_id", "tile", "cell", "fold"])
        w.writeheader()
        w.writerows({"pass2_id": it["pass2_id"], "item_id": it["item_id"], "tile": it["tile"],
                     "cell": it["cell"], "fold": it["fold"]} for it in p2)
    sheet(PACK / "scoring_sheet_pass2.csv", [it["pass2_id"] for it in p2],
          "pass 2, self-consistency, score at least 24 h after pass 1")

    # ---------------- render ----------------
    PANELS.mkdir(parents=True, exist_ok=True)
    PANELS2.mkdir(parents=True, exist_ok=True)
    write_legend(PACK / "legend.png")

    tasks = [(it["item_id"], it["tile"], str(celldir[it["cell"]][it["fold"]])) for it in items]
    tasks2 = [(it["pass2_id"], it["tile"], str(celldir[it["cell"]][it["fold"]])) for it in p2]
    for label, tk, out in (("pass 1", tasks, PANELS), ("pass 2", tasks2, PANELS2)):
        print(f"\nrendering {label}: {len(tk)} panels, {args.procs} processes -> {out}")
        fails = []
        with Pool(args.procs) as pool:
            for n, (st, iid, err) in enumerate(
                    pool.imap_unordered(partial(render_item, out_dir=str(out)), tk, chunksize=4), 1):
                if st != "OK":
                    fails.append((iid, err))
                if n % 48 == 0:
                    print(f"  {n} / {len(tk)}")
        if fails:
            print(f"  {len(fails)} FAILED")
            for iid, err in fails[:5]:
                print(f"    {iid}: {err}")
        else:
            print(f"  {len(tk)} panels written, 0 failures")

    prov = C.assert_writes_are_local(PACK / "pack_provenance.json")
    prov.write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "declaration_version": str(dec.get("declaration_version")),
        "declaration_status": dec.get("status"),
        "seed": int(q["seed"]),
        "seed_usage_order": ["1 stratified per-fold sample", "2 pass-1 item shuffle",
                             "3 double-scored tile subset", "4 pass-2 item shuffle"],
        "n_tiles": len(picked), "n_items": len(items), "n_pass2_items": len(p2),
        "cells": cells, "weak_classes": list(q["weak_classes"]),
        "strata": dict(q["strata"]),
        "adjacent_same_tile_pairs": adjacent,
        "modes": [{"column": m, "definition": d} for m, d in MODES],
        "blinding": "panels carry an item id only; no tile, route or cell in image or filename",
        "sealed_keys": ["SEALED_item_key.csv", "SEALED_item_key_pass2.csv"],
        "ignore_index_rendering": "class 0 left unpainted, so unannotated ground is visible as raw "
                                  "imagery in the ground-truth panel",
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {prov}")
    print(f"\nPACK READY: {PACK}")
    print("SEALED_item_key.csv and SEALED_item_key_pass2.csv must not be opened until scoring ends.")


if __name__ == "__main__":
    main()
