#!/usr/bin/env python
"""
E0 -- Tile inventory: the per-tile join table every other EDA module groups by.

WHY THIS EXISTS
    Two audits already cover this dataset, but both stored only AGGREGATES:
      logs_and_models/class_pixel_audit/   -> global per-class totals (11 rows)
      logs_and_models/route_class_audit/   -> per-route per-class totals (16 rows)
    Neither kept a per-tile record, so questions like "how do classes co-occur within a tile",
    "where is this tile on the ground", "is the ignore mask geographically biased" cannot be
    answered from anything on disk. This script builds the missing primitive: one row per tile.

WHAT IT PRODUCES
    results/tables/tile_inventory.csv  -- 19,314 rows:
      filename, route, year, parent, tile_i, tile_j, fold,
      easting, northing, centroid_e, centroid_n,          (EPSG:25832, metres)
      px_<class> x 11, present_<class> x 11,
      n_scored_px, n_predicted_classes_present

    Geography comes from the label GeoTIFF's affine transform. All seven layers (six channels plus
    the label) are pixel-aligned on an identical transform, so one header read per tile is enough
    to place it on the ground. Tiles are 1000 x 1000 at 0.1 m GSD, i.e. 100 m x 100 m.

VALIDATION (this is the point -- a silent error must not be able to pass)
    Per-class pixel and tile-presence totals are asserted EXACTLY against class_pixel_audit.json,
    per-route pixel totals EXACTLY against route_class_audit.csv, and the fold sizes against the
    frozen split. Any drift means this inventory is wrong and the script exits non-zero.

READ-ONLY over the source data. Writes only under exploratory_data_analysis/.

    python eda_tile_inventory.py [--procs N] [--limit N] [--selftest]
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402


# ----------------------------------------------------------------------------------------------
# Worker: one tile -> class counts + ground position. Module level so Windows spawn can pickle it.
# ----------------------------------------------------------------------------------------------
def worker(fname: str):
    import rasterio
    path = C.LABEL_DIR / fname
    try:
        with rasterio.open(path) as src:
            lab = src.read(1).astype(np.int32)
            tr = src.transform
            easting, northing = float(tr.c), float(tr.f)
    except Exception as exc:                      # noqa: BLE001 -- report, never repair
        return ("FAIL", fname, f"{type(exc).__name__}: {exc}")

    # Fold out-of-range values into unknown, exactly as class_pixel_audit.worker and the trainer do.
    lab[(lab < 0) | (lab >= C.NCLASS)] = 0
    px = np.bincount(lab.ravel(), minlength=C.NCLASS).astype(np.int64)
    return ("OK", fname, px, easting, northing, int(lab.size))


def build_rows(results, route_to_fold):
    """Turn worker output into inventory rows. Pure, so it can be self-tested without file I/O."""
    rows = []
    for status, fname, px, easting, northing, size in results:
        meta = C.parse_filename(fname)
        route = meta["route"]
        if route not in route_to_fold:
            raise RuntimeError(f"tile {fname} has route {route}, absent from the frozen split")
        present = (px > 0).astype(np.int64)
        row = {
            "filename": fname,
            "route": route,
            "year": meta["year"],
            "parent": meta["parent"],
            "tile_i": meta["tile_i"],
            "tile_j": meta["tile_j"],
            "fold": route_to_fold[route],
            "easting": easting,
            "northing": northing,
            "centroid_e": easting + C.TILE_GROUND_M / 2.0,
            "centroid_n": northing - C.TILE_GROUND_M / 2.0,   # north-up: row 0 is the top
            "n_scored_px": int(size - px[C.IGNORE_INDEX]),
            "n_predicted_classes_present": int(sum(present[c] for c in C.PREDICTED)),
        }
        for i, name in enumerate(C.CODES):
            row[C.px_col(name)] = int(px[i])
            row[C.present_col(name)] = int(present[i])
        rows.append(row)
    return rows


FIELDS = (["filename", "route", "year", "parent", "tile_i", "tile_j", "fold",
           "easting", "northing", "centroid_e", "centroid_n",
           "n_scored_px", "n_predicted_classes_present"]
          + [C.px_col(n) for n in C.CODES]
          + [C.present_col(n) for n in C.CODES])


# ----------------------------------------------------------------------------------------------
# Validation against the artifacts that already exist
# ----------------------------------------------------------------------------------------------
def validate(rows, partial: bool) -> bool:
    import pandas as pd
    df = pd.DataFrame(rows)
    ok = True

    C.banner("VALIDATION")

    if partial:
        print("--limit was used, so totals cannot match the full-pool audits. Structural checks only.\n")
    else:
        audit = C.load_class_pixel_audit()
        print(f"{'class':<12}{'px (inventory)':>18}{'px (audit)':>18}  {'':2}"
              f"{'tiles (inv)':>12}{'tiles (audit)':>14}")
        for name in C.CODES:
            inv_px = int(df[C.px_col(name)].sum())
            inv_tl = int(df[C.present_col(name)].sum())
            ref = audit[name]
            match = (inv_px == ref["pixel_count"]) and (inv_tl == ref["tiles_present"])
            ok &= match
            print(f"{name:<12}{inv_px:>18,}{ref['pixel_count']:>18,}  {'OK' if match else 'FAIL':<2}"
                  f"{inv_tl:>12,}{ref['tiles_present']:>14,}")

        # Per-route pixel totals against route_class_audit.csv
        ref_routes = {}
        with open(C.ROUTE_CLASS_AUDIT_CSV) as fh:
            for rec in csv.DictReader(fh):
                ref_routes[rec["route"]] = rec
        route_ok = True
        for route, grp in df.groupby("route"):
            ref = ref_routes.get(route)
            if ref is None:
                print(f"  route {route} missing from route_class_audit.csv")
                route_ok = False
                continue
            if int(grp.shape[0]) != int(ref["route_tiles"]):
                print(f"  route {route}: {grp.shape[0]} tiles vs audit {ref['route_tiles']}")
                route_ok = False
            for name in C.CODES:
                if int(grp[C.px_col(name)].sum()) != int(ref[f"px_{name}"]):
                    print(f"  route {route} class {name}: pixel total differs from the audit")
                    route_ok = False
        ok &= route_ok
        print(f"\nper-route totals vs route_class_audit.csv : {'OK' if route_ok else 'FAIL'}")

        total_ok = len(df) == C.EXPECTED_TILES
        ok &= total_ok
        print(f"tile count {len(df)} == {C.EXPECTED_TILES}                 : "
              f"{'OK' if total_ok else 'FAIL'}")

        fold_counts = df.groupby("fold").size().to_dict()
        fold_ok = fold_counts == C.EXPECTED_FOLD_COUNTS
        ok &= fold_ok
        print(f"fold sizes {fold_counts} : {'OK' if fold_ok else 'FAIL'} "
              f"(expected {C.EXPECTED_FOLD_COUNTS})")

    # Structural checks that hold regardless of --limit
    dup = df["filename"].duplicated().sum()
    print(f"duplicate filenames: {dup} {'OK' if dup == 0 else 'FAIL'}")
    ok &= dup == 0

    geo_ok = bool(df["easting"].notna().all() and (df["easting"] > 0).all())
    print(f"all tiles georeferenced: {'OK' if geo_ok else 'FAIL'}")
    ok &= geo_ok
    return ok


def summarize(rows) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    C.banner("INVENTORY SUMMARY")
    print(f"tiles                : {len(df):,}")
    print(f"routes               : {df['route'].nunique()}   parents: {df['parent'].nunique():,}")
    print(f"years                : {sorted(df['year'].unique().tolist())}")
    e0, e1 = df["centroid_e"].min(), df["centroid_e"].max()
    n0, n1 = df["centroid_n"].min(), df["centroid_n"].max()
    print(f"extent (EPSG:{C.EPSG}) : easting {e0:,.0f} to {e1:,.0f} "
          f"({(e1 - e0) / 1000:,.0f} km)")
    print(f"                       northing {n0:,.0f} to {n1:,.0f} "
          f"({(n1 - n0) / 1000:,.0f} km)")

    print("\nclasses present per tile (of the 9 predicted):")
    vc = df["n_predicted_classes_present"].value_counts().sort_index()
    for k, v in vc.items():
        print(f"  {k} classes : {v:6,} tiles  ({100 * v / len(df):5.2f}%)")

    ign = df[C.px_col('unknown')].sum() / df[[C.px_col(n) for n in C.CODES]].sum().sum()
    print(f"\nignore (unknown) share of all label pixels: {100 * ign:.3f}%")
    print("  -> module E4 decomposes this; it is the largest single property of the label set")


def main() -> None:
    ap = argparse.ArgumentParser(description="E0 -- build the per-tile inventory")
    ap.add_argument("--procs", type=int, default=min(32, __import__("os").cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=None, help="debug: only the first N tiles")
    ap.add_argument("--selftest", action="store_true", help="check the pure logic, touch no files")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    C.ensure_out_dirs()
    names = C.tile_names()
    if args.limit:
        names = names[: args.limit]
    route_to_fold = C.load_fold_assignment(str(C.FOLD_ASSIGNMENT_CSV))

    C.banner(f"E0 -- tile inventory over {len(names):,} tiles, {args.procs} processes")
    print(f"labels : {C.LABEL_DIR}  (read-only)")
    print(f"output : {C.TILE_INVENTORY_CSV}\n")

    results, failures = [], []
    with Pool(processes=args.procs) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, names, chunksize=16), 1):
            if res[0] == "OK":
                results.append(res)
            else:
                failures.append((res[1], res[2]))
            if i % 2000 == 0:
                print(f"  {i:,} / {len(names):,}")

    if failures:
        print(f"\n{len(failures)} tiles failed to read:")
        for fname, err in failures[:20]:
            print(f"  {fname}: {err}")
        sys.exit("aborting -- the inventory would be incomplete")

    rows = build_rows(results, route_to_fold)
    rows.sort(key=lambda r: r["filename"])

    out = C.assert_writes_are_local(C.TILE_INVENTORY_CSV)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")

    summarize(rows)
    ok = validate(rows, partial=bool(args.limit))

    C.write_json({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_tiles": len(rows),
        "source_list": str(C.ALL_TXT),
        "label_dir": str(C.LABEL_DIR),
        "fold_assignment": str(C.FOLD_ASSIGNMENT_CSV),
        "validated_against_existing_audits": bool(ok and not args.limit),
    }, C.TABLES / "tile_inventory_provenance.json")

    if not ok:
        sys.exit("\nVALIDATION FAILED -- do not use this inventory")
    print("\nVALIDATION PASSED")


def selftest() -> None:
    """Exercise the pure aggregation logic on synthetic data. No file I/O."""
    C.banner("E0 selftest (synthetic, no files touched)")

    meta = C.parse_filename("O2021_82_11_1_0023_00003029_3000_2000.tif")
    assert meta["route"] == "82-11", meta
    assert meta["year"] == 2021 and meta["tile_i"] == 3000 and meta["tile_j"] == 2000
    assert meta["parent"] == "O2021_82_11_1_0023_00003029"
    print("parse_filename            : OK")

    # Two synthetic tiles on the same route; one holds a rare class, one does not.
    px_a = np.zeros(C.NCLASS, dtype=np.int64); px_a[0] = 400_000; px_a[4] = 600_000
    px_b = np.zeros(C.NCLASS, dtype=np.int64); px_b[4] = 900_000; px_b[5] = 100_000
    fake = [
        ("OK", "O2021_82_11_1_0023_00003029_0_0.tif", px_a, 460909.0, 6257502.0, 1_000_000),
        ("OK", "O2021_82_11_1_0023_00003029_0_1000.tif", px_b, 460909.0, 6257402.0, 1_000_000),
    ]
    rows = build_rows(fake, {"82-11": 1})

    assert rows[0]["n_scored_px"] == 600_000, rows[0]["n_scored_px"]
    assert rows[1]["n_scored_px"] == 1_000_000
    assert rows[0]["n_predicted_classes_present"] == 1     # ubefestet only (unknown is not predicted)
    assert rows[1]["n_predicted_classes_present"] == 2     # ubefestet + green_roof
    assert rows[0][C.present_col("green_roof")] == 0
    assert rows[1][C.px_col("green_roof")] == 100_000
    assert rows[0]["fold"] == 1
    print("scored-pixel accounting   : OK (ignore_index excluded, never counted as a class)")
    print("class presence per tile   : OK")

    # centroid is half a tile in from the upper-left corner, north-up
    assert abs(rows[0]["centroid_e"] - (460909.0 + 50.0)) < 1e-9
    assert abs(rows[0]["centroid_n"] - (6257502.0 - 50.0)) < 1e-9
    print("centroid geometry         : OK (UL + 50 m east, 50 m south)")

    try:
        build_rows(fake, {"99-99": 0})
    except RuntimeError:
        print("unknown-route guard       : OK (fails loud)")
    else:
        raise AssertionError("expected a RuntimeError for a route outside the frozen split")

    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
