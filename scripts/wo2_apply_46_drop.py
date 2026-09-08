#!/usr/bin/env python
"""
Apply Great Plan 3.1 section 4.6's pre-agreed drop to the WO-2 qualitative pack: 96 -> 64 tiles,
"with the double-scored share kept".

WHY THIS IS A FILTER AND NOT A REDRAW. The 96-tile sample was drawn under the locked seed and
persisted before any panel was rendered. Redrawing 64 tiles would replace that sample with a new
one and throw away the property that makes it pre-registered. Instead this takes a SUBSET of the
already-persisted 96, in the original per-stratum draw order, so:

  - no new randomness enters anywhere;
  - the 96-tile list, the 216 rendered panels and both sealed keys stay untouched on disk as the
    parent record;
  - the reduction is reproducible from `sample_tiles.csv` alone, with no seed needed.

THE ARITHMETIC, stated because 64 does not divide evenly into the declared structure. D4 fixes
32 tiles per fold, half weak-class-present and half complement. 64 tiles cannot be 3 equal folds,
so the allocation is 22 / 22 / 20 across folds 0 / 1 / 2, and the weak/complement balance is held
exactly within every fold (11+11, 11+11, 10+10). Totals stay 32 weak and 32 complement. The
reduction falls on the highest fold index by a stated deterministic rule rather than on whichever
fold would flatter the result.

THE DOUBLE-SCORED SHARE. D4 sets 12 tiles of 96, and states that as "12.5 %". Section 4.6 says the
*share* is kept, so at 64 tiles that is 8 tiles and 16 items, not 12. Those 8 are the first 8 of the
original 12, in their original draw order, that survive into the retained 64; if fewer than 8
survive, the remainder is topped up from the retained tiles in draw order. Every pass-2 panel needed
already exists.

DISCLOSURE. Building the subset requires mapping item ids to tiles, so this module reads
`SEALED_item_key.csv`. That does not affect the author's blinding -- the author has not opened it,
and nothing in this module's output or console reveals any item -> (tile, cell) mapping. The sheets
it writes contain item ids only.

    python wo2_apply_46_drop.py [--n 64] [--no-hardlinks]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

PACK = C.EDA_ROOT / "results" / "qualitative_pack"
SEED = 20260825          # the declared seed, reused for the rebuilt pass-2 shuffle
MODES = ["m1_speckle", "m2_boundary_bleed", "m3_spurious_patch", "m4_paved_swap",
         "m5_road_continuation", "m6_veg_occlusion", "m7_misregistration",
         "m8_large_object", "m9_other"]


def allocate(n_total, n_folds=3):
    """Per-fold tile counts, stratum-balanced within each fold, reduction on the highest index."""
    per_stratum = n_total // (2 * n_folds)           # 64 -> 10
    rem = (n_total - per_stratum * 2 * n_folds) // 2  # leftover PAIRS to distribute -> 2
    out = {}
    for f in range(n_folds):
        k = per_stratum + (1 if f < rem else 0)
        out[f] = k
    return out                                        # {0: 11, 1: 11, 2: 10} per stratum


def main():
    ap = argparse.ArgumentParser(description="apply the section 4.6 drop to the WO-2 pack")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--no-hardlinks", action="store_true")
    args = ap.parse_args()

    with open(PACK / "sample_tiles.csv", newline="", encoding="utf-8") as fh:
        sample = list(csv.DictReader(fh))
    print(f"parent sample: {len(sample)} tiles (untouched)")

    per_stratum = allocate(args.n)
    print(f"section 4.6 target n = {args.n}")
    for f, k in per_stratum.items():
        print(f"  fold {f}: {k} weak-present + {k} complement = {2 * k}")
    total = sum(2 * k for k in per_stratum.values())
    if total != args.n:
        sys.exit(f"allocation sums to {total}, not {args.n}")

    # Prefix of the original draw order within each (fold, stratum). The file preserves that order.
    kept, seen = [], {}
    for row in sample:
        key = (int(row["fold"]), row["stratum"])
        seen.setdefault(key, 0)
        if seen[key] < per_stratum[int(row["fold"])]:
            kept.append(row)
            seen[key] += 1
    kept_names = {r["filename"] for r in kept}
    assert len(kept) == args.n, f"{len(kept)} != {args.n}"
    print(f"retained {len(kept)} tiles as a prefix of the parent draw order (no new randomness)")

    out = C.assert_writes_are_local(PACK / f"sample_tiles_{args.n}.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0]))
        w.writeheader()
        w.writerows(kept)
    print(f"wrote {out}")

    # ---- pass 1 sheet: the in-scope item ids, in the existing shuffled order ----
    with open(PACK / "SEALED_item_key.csv", newline="", encoding="utf-8") as fh:
        key1 = list(csv.DictReader(fh))          # read to filter; never echoed
    in_scope = [r["item_id"] for r in key1 if r["tile"] in kept_names]
    print(f"pass 1: {len(in_scope)} of {len(key1)} items in scope")

    cols = ["item_id"] + MODES + ["dominant_mode", "note"]
    s1 = C.assert_writes_are_local(PACK / f"scoring_sheet_{args.n}.csv")
    with open(s1, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in in_scope:
            w.writerow([i] + [""] * (len(cols) - 1))
    print(f"wrote {s1}  ({len(in_scope)} rows)")

    # ---- pass 2: keep the SHARE, 12.5 % of n ----
    n_dbl = round(args.n * 12 / 96)
    with open(PACK / "SEALED_item_key_pass2.csv", newline="", encoding="utf-8") as fh:
        key2 = list(csv.DictReader(fh))
    orig_dbl = []
    for r in key2:                                # preserve first-appearance order
        if r["tile"] not in orig_dbl:
            orig_dbl.append(r["tile"])
    survivors = [t for t in orig_dbl if t in kept_names]
    chosen = survivors[:n_dbl]
    if len(chosen) < n_dbl:
        for r in kept:                            # top up in retained draw order
            if r["filename"] not in chosen and len(chosen) < n_dbl:
                chosen.append(r["filename"])
    chosen_set = set(chosen)
    n_topped = max(0, n_dbl - len(survivors))
    print(f"pass 2: double-scored share held at 12.5 % -> {n_dbl} tiles, {2 * n_dbl} items")
    print(f"        {len(survivors)} of the original 12 survived the truncation; "
          f"{n_topped} topped up from the retained set")

    # A topped-up tile has no pass-2 panel, because pass 2 was only ever rendered for the original
    # twelve. Reusing just the survivors would quietly drop the share to 10.9 %, so pass 2 is
    # rebuilt for all `n_dbl` chosen tiles: both cells per tile, reshuffled under the declared seed,
    # into a fresh id space. Nothing from the parent pack is modified -- the original 24-item pass 2
    # stays on disk unused. The author has scored nothing yet, so no work is invalidated.
    import random
    tile_cell = {}
    for r in key1:
        tile_cell.setdefault(r["tile"], []).append(r)
    p2_items = [dict(r) for t in chosen for r in tile_cell[t]]
    random.Random(SEED).shuffle(p2_items)
    for n, it in enumerate(p2_items, 1):
        it["pass2_id"] = f"S{n:03d}"
    p2_ids = [it["pass2_id"] for it in p2_items]
    assert len(p2_ids) == 2 * n_dbl, f"{len(p2_ids)} != {2 * n_dbl}"

    k2 = C.assert_writes_are_local(PACK / f"SEALED_item_key_pass2_{args.n}.csv")
    with open(k2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pass2_id", "item_id", "tile", "cell", "fold"])
        w.writeheader()
        w.writerows({"pass2_id": it["pass2_id"], "item_id": it["item_id"], "tile": it["tile"],
                     "cell": it["cell"], "fold": it["fold"]} for it in p2_items)

    s2 = C.assert_writes_are_local(PACK / f"scoring_sheet_pass2_{len(p2_ids)}.csv")
    with open(s2, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i in p2_ids:
            w.writerow([i] + [""] * (len(cols) - 1))
    print(f"wrote {s2}  ({len(p2_ids)} rows)")

    # render the rebuilt pass 2 -- 16 panels, the only rendering this module does
    import wo2_qualitative_pack as W
    dest2 = PACK / f"panels_pass2_{args.n}"
    dest2.mkdir(exist_ok=True)
    celldir = {}
    for cell in sorted({it["cell"] for it in p2_items}):
        hit = list(C.SPATIAL_MATRIX.glob(f"*/oof_{cell}"))
        celldir[cell] = {f: str(hit[0].parent / f"{cell}_fold{f}" / "models" / "example_dataset")
                         for f in range(C.NFOLDS)}
    fails = 0
    for it in p2_items:
        st, _iid, err = W.render_item(
            (it["pass2_id"], it["tile"], celldir[it["cell"]][int(it["fold"])]), out_dir=str(dest2))
        if st != "OK":
            fails += 1
            print(f"    FAIL {it['pass2_id']}: {err}")
    print(f"rendered {len(p2_items) - fails} / {len(p2_items)} pass-2 panels -> {dest2.name}/"
          + ("" if not fails else f"   {fails} FAILED"))

    # ---- an in-scope panel folder, hardlinked so it costs no disk ----
    made = 0
    if not args.no_hardlinks:
        dest = PACK / f"panels_{args.n}"
        dest.mkdir(exist_ok=True)
        for i in in_scope:
            src = PACK / "panels" / f"item_{i}.png"
            dst = dest / f"item_{i}.png"
            if dst.exists():
                made += 1
                continue
            try:
                os.link(src, dst)
                made += 1
            except OSError:
                subprocess.run(["cmd", "/c", "mklink", "/H", str(dst), str(src)],
                               capture_output=True, check=False)
                if dst.exists():
                    made += 1
        print(f"{made} of {len(in_scope)} in-scope panels linked into {dest.name}/ "
              f"(hardlinks, no extra disk)")

    prov = C.assert_writes_are_local(PACK / f"drop_46_provenance_{args.n}.json")
    prov.write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "authority": "Great Plan 3.1 section 4.6 pre-agreed drop, invoked by the author 2026-09-07",
        "parent_sample": "sample_tiles.csv (96 tiles), untouched",
        "method": "prefix of the parent per-(fold, stratum) draw order; no new randomness",
        "n_tiles": args.n,
        "per_fold_per_stratum": {str(k): v for k, v in per_stratum.items()},
        "fold_totals": {str(k): 2 * v for k, v in per_stratum.items()},
        "weak_total": sum(per_stratum.values()), "complement_total": sum(per_stratum.values()),
        "pass1_items": len(in_scope), "pass2_tiles": n_dbl, "pass2_items": len(p2_ids),
        "double_scored_share": "12.5 % held constant, per section 4.6 wording",
        "panels_rendered": 0,
        "blinding": "no item -> (tile, cell) mapping is disclosed by this module or its outputs",
        "d4_item_9": "this is a departure from D4's n = 96 and is recorded as a dated amendment",
    }, indent=2), encoding="utf-8")
    print(f"wrote {prov}")
    print(f"\nDONE. Score {len(in_scope)} items from scoring_sheet_{args.n}.csv, then "
          f"{len(p2_ids)} from scoring_sheet_pass2_{len(p2_ids)}.csv at least 24 h later.")


if __name__ == "__main__":
    main()
