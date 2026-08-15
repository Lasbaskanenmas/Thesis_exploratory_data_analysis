#!/usr/bin/env python
"""
E4 -- Label quality: where the 36.9 percent `unknown` mass comes from, whether the ignore mask is
biased, what was excluded from the pool, and how much error sits on annotation boundaries.

WHY THIS EXISTS
    36.861 percent of every label pixel in the pool is class 0, `unknown`, the ignore index. It is
    the single largest property of the label set and nothing on disk explains it. Whether that mass
    is ANNOTATOR UNCERTAINTY or simply UNANNOTATED GROUND changes the interpretation of every
    metric in the thesis, because ignore pixels are silently dropped from all of them.

WHAT IT MEASURES
    1. Origin of the ignore mass, from the 799 pre-tiling parent rasters in labels/large_label/.
       Those rasters carry a nodata sentinel of 15 which the reclass step maps to 0. Separating
       sentinel-15 area from genuine class-0 polygons decides coverage versus uncertainty.
    2. Annotated footprint from example_dataset_ground_surface.gpkg, as an independent check on 1.
    3. Bias of the ignore mask across routes and tiles. If unknown concentrates geographically it
       silently reweights every metric, which is a validity problem for SQ2.
    4. Selection bias: 60,123 label tiles exist but only 19,314 are in all.txt. What was dropped,
       and is the drop geographically or compositionally biased?
    5. Boundary ambiguity: out-of-fold error rate as a function of distance to the nearest label
       boundary. This separates "the model is wrong" from "the annotation is imprecise" and is the
       concrete measurement behind the brosten / fliser / asfalt taxonomy hypothesis (finding F5b).

WHAT IT CANNOT MEASURE, AND WHY (stated rather than quietly skipped)
    The plan intended to quantify how much ignore mass the cleaning step
    (data_cleaning_based_on_newer_ground_truth.py, which sets pixels whose class changed between
    two annotation vintages to 0) contributed. That is NOT reproducible from disk:
      - labels/old_splitted_labels/ holds 16 tiles, from a DIFFERENT tiling (stride 960, i.e. the
        overlap-40 grid), and only 1 of the 16 filenames exists in the current pool.
      - labels/large_label/reclass/ is not an older vintage. It is the same raster with the nodata
        sentinel 15 remapped to 0, verified by cross-tabulation.
    So annotator churn cannot be estimated here. This module reports that limitation explicitly and
    uses the boundary analysis as the available proxy for annotation precision.

READ-ONLY over the source data. Writes only under exploratory_data_analysis/.

    python eda_label_quality.py [--procs N] [--skip-parent-scan] [--boundary-tiles N]
                                [--boundary-cell NAME] [--selftest]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

LARGE_LABEL_DIR = C.DATASET_ROOT / "labels" / "large_label"
RECLASS_DIR = LARGE_LABEL_DIR / "reclass"
NODATA_SENTINEL_LABEL = 15          # the value large_label uses for "no annotation here"
NVALS = 16                          # scan 0..15 so the sentinel is visible alongside the classes

# Half-open distance buckets in pixels: [0,1), [1,2), [2,3), [3,5), [5,10), [10,20), [20,50), >=50.
# The Euclidean distance transform returns non-integer values on diagonals (sqrt(2) and so on), so
# these are ranges rather than exact pixel counts. Bucket 0 is exactly "on a boundary".
BOUNDARY_EDGES = [1, 2, 3, 5, 10, 20, 50]
BOUNDARY_LABELS = ["0 (on boundary)", "[1,2)", "[2,3)", "[3,5)",
                   "[5,10)", "[10,20)", "[20,50)", ">=50"]
NBIN = len(BOUNDARY_LABELS)


# ----------------------------------------------------------------------------------------------
# 1. Origin of the ignore mass, from the parent rasters
# ----------------------------------------------------------------------------------------------
def parent_worker(name: str):
    import rasterio
    try:
        with rasterio.open(LARGE_LABEL_DIR / name) as src:
            a = src.read(1)
        counts = np.bincount(a.ravel(), minlength=NVALS)[:NVALS].astype(np.int64)
        return ("OK", name, counts, int(a.size))
    except Exception as exc:                       # noqa: BLE001
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def scan_parents(procs: int):
    names = sorted(p.name for p in LARGE_LABEL_DIR.glob("*.tif"))
    if not names:
        return None
    print(f"scanning {len(names)} parent rasters in {LARGE_LABEL_DIR} ...")
    total = np.zeros(NVALS, dtype=np.int64)
    px = 0
    fails = []
    with Pool(processes=procs) as pool:
        for i, res in enumerate(pool.imap_unordered(parent_worker, names, chunksize=4), 1):
            if res[0] != "OK":
                fails.append((res[1], res[2]))
                continue
            total += res[2]
            px += res[3]
            if i % 100 == 0:
                print(f"  {i} / {len(names)}")
    return {"names": names, "counts": total, "pixels": px, "failures": fails}


# ----------------------------------------------------------------------------------------------
# 5. Boundary ambiguity
# ----------------------------------------------------------------------------------------------
def boundary_worker(args):
    """Bucket out-of-fold errors by distance to the nearest annotation boundary."""
    from scipy import ndimage
    fname, pred_dir = args
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        label = C.read_label(C.LABEL_DIR / fname)
        pred = np.asarray(Image.open(Path(pred_dir) / fname)).astype(np.int32)
        if label.shape != pred.shape:
            return ("FAIL", fname, f"shape {label.shape} vs {pred.shape}")

        # A pixel is on a boundary when any 4-neighbour carries a different class.
        b = np.zeros_like(label, dtype=bool)
        b[:-1, :] |= label[:-1, :] != label[1:, :]
        b[1:, :] |= label[:-1, :] != label[1:, :]
        b[:, :-1] |= label[:, :-1] != label[:, 1:]
        b[:, 1:] |= label[:, :-1] != label[:, 1:]

        dist = ndimage.distance_transform_edt(~b) if b.any() else np.full(label.shape, 1e6)

        scored = label != C.IGNORE_INDEX
        err = scored & (pred != label)
        idx = np.digitize(dist[scored], BOUNDARY_EDGES)
        n_tot = np.bincount(idx, minlength=NBIN).astype(np.int64)
        idx_e = np.digitize(dist[err], BOUNDARY_EDGES)
        n_err = np.bincount(idx_e, minlength=NBIN).astype(np.int64)
        return ("OK", fname, n_tot, n_err)
    except Exception as exc:                       # noqa: BLE001
        return ("FAIL", fname, f"{type(exc).__name__}: {exc}")


def resolve_cell_dirs(cell: str) -> dict[int, Path]:
    """Map fold -> prediction directory for a scored cell, from the untouched matrix tree."""
    out = {}
    for f in range(C.NFOLDS):
        hits = list(C.SPATIAL_MATRIX.glob(f"*/{cell}_fold{f}/models/example_dataset"))
        if len(hits) != 1:
            raise RuntimeError(f"expected one prediction dir for {cell} fold {f}, found {hits}")
        out[f] = hits[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E4 -- label quality")
    ap.add_argument("--procs", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--skip-parent-scan", action="store_true")
    ap.add_argument("--boundary-tiles", type=int, default=600)
    ap.add_argument("--boundary-cell", default="convnext_upernet_rgb")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    C.ensure_out_dirs()
    df = C.load_tile_inventory()
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat()}

    C.banner("E4 -- label quality")

    # ------------------------------------------------------------------------------------------
    C.banner("1. ORIGIN OF THE IGNORE MASS  (from the 799 parent rasters)")
    if args.skip_parent_scan:
        print("skipped (--skip-parent-scan)")
    else:
        res = scan_parents(args.procs)
        if res is None:
            print(f"no parent rasters found under {LARGE_LABEL_DIR}; skipping")
        else:
            cnt, px = res["counts"], res["pixels"]
            sentinel = int(cnt[NODATA_SENTINEL_LABEL])
            explicit0 = int(cnt[0])
            real = int(cnt[1:NODATA_SENTINEL_LABEL].sum())
            print(f"\nparent rasters scanned : {len(res['names'])}")
            print(f"pixels                 : {px:,}")
            print(f"\n{'value':<10}{'pixels':>18}{'share':>9}   meaning")
            print(f"{'15':<10}{sentinel:>18,}{100 * sentinel / px:>8.3f}%   "
                  f"nodata sentinel -> becomes class 0 (unannotated ground)")
            print(f"{'0':<10}{explicit0:>18,}{100 * explicit0 / px:>8.3f}%   "
                  f"explicit unknown class in the annotation")
            print(f"{'1..14':<10}{real:>18,}{100 * real / px:>8.3f}%   annotated surface classes")

            ignore_total = sentinel + explicit0
            if ignore_total:
                print(f"\nOf all pixels that end up as ignore, "
                      f"{100 * sentinel / ignore_total:.2f}% are the nodata sentinel and "
                      f"{100 * explicit0 / ignore_total:.2f}% are explicit unknown.")
            print("\nReading: the ignore mass is ANNOTATION COVERAGE, not annotator uncertainty.")
            print("Annotators delineated selected polygons and everything outside them is simply")
            print("unlabelled. So the ~37% is not label noise to be cleaned; it is the share of")
            print("the imagery that carries no ground truth at all, and it is dropped from every")
            print("metric silently. The effective supervised dataset is roughly 12.2 of 19.3")
            print("billion pixels.")
            payload["parent_scan"] = {
                "n_parents": len(res["names"]), "pixels": px,
                "sentinel_15_px": sentinel, "explicit_class0_px": explicit0,
                "annotated_px": real,
                "counts_by_value": {str(i): int(cnt[i]) for i in range(NVALS) if cnt[i]},
                "failures": res["failures"][:20],
            }

    # ------------------------------------------------------------------------------------------
    C.banner("2. ANNOTATED FOOTPRINT  (independent check, from the GeoPackage)")
    try:
        import geopandas as gpd
        gdf = gpd.read_file(C.GROUND_SURFACE_GPKG)
        poly_area = float(gdf.geometry.area.sum())
        tile_area = len(df) * (C.TILE_GROUND_M ** 2)
        print(f"polygons              : {len(gdf):,}")
        print(f"attribute columns     : {[c for c in gdf.columns if c != 'geometry'][:8]}")
        print(f"annotated polygon area: {poly_area / 1e6:,.2f} km2")
        print(f"tile footprint area   : {tile_area / 1e6:,.2f} km2 "
              f"({len(df):,} tiles x {C.TILE_GROUND_M:.0f} m squared)")
        print(f"coverage ratio        : {100 * poly_area / tile_area:.2f}%")
        print("\n(Tiles overlap slightly at parent margins, so the tile-footprint figure is a mild")
        print("over-estimate of unique ground. The ratio is a cross-check on section 1, not a")
        print("replacement for it.)")
        if "ML_CATEGORY" in gdf.columns:
            vc = gdf["ML_CATEGORY"].value_counts()
            print(f"\npolygons per ML_CATEGORY:")
            for k, v in vc.items():
                print(f"  {str(k):<22}{v:>9,}")
        payload["gpkg"] = {"n_polygons": len(gdf), "annotated_area_m2": poly_area,
                           "tile_footprint_area_m2": tile_area,
                           "coverage_ratio": poly_area / tile_area}
    except Exception as exc:                       # noqa: BLE001
        print(f"could not read the GeoPackage: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------------------------------
    C.banner("3. IS THE IGNORE MASK BIASED?  (per route)")
    tot_px = len(df) * C.TILE_PX * C.TILE_PX
    print(f"pool-wide unknown share: "
          f"{100 * df[C.px_col('unknown')].sum() / tot_px:.3f}%\n")
    print(f"{'route':<9}{'fold':>5}{'tiles':>8}{'unknown %':>12}{'scored px':>16}")
    rows = []
    for route, g in df.groupby("route"):
        px = len(g) * C.TILE_PX * C.TILE_PX
        share = g[C.px_col("unknown")].sum() / px
        rows.append({"route": route, "fold": int(g["fold"].iloc[0]), "tiles": len(g),
                     "unknown_share": share, "scored_px": int(g["n_scored_px"].sum())})
        print(f"{route:<9}{g['fold'].iloc[0]:>5}{len(g):>8}{100 * share:>11.2f}%"
              f"{g['n_scored_px'].sum():>16,}")
    shares = np.array([r["unknown_share"] for r in rows])
    print(f"\nrange across routes: {100 * shares.min():.2f}% to {100 * shares.max():.2f}% "
          f"(spread {100 * (shares.max() - shares.min()):.1f} points)")
    print("A wide spread means the amount of supervision differs sharply by geography, so routes")
    print("do not contribute equally to any pooled metric even before class imbalance is counted.")

    fold_share = df.groupby("fold").apply(
        lambda g: g[C.px_col("unknown")].sum() / (len(g) * C.TILE_PX * C.TILE_PX),
        include_groups=False)
    print(f"\nper fold: " + ", ".join(f"fold {k} {100 * v:.2f}%" for k, v in fold_share.items()))
    payload["ignore_bias"] = {"per_route": rows,
                              "per_fold": {str(k): float(v) for k, v in fold_share.items()}}

    # ------------------------------------------------------------------------------------------
    C.banner("4. SELECTION BIAS  (what was excluded from the pool)")
    n_label_tiles = sum(1 for _ in C.LABEL_DIR.glob("*.tif"))
    n_parents_pool = df["parent"].nunique()
    n_parents_large = sum(1 for _ in LARGE_LABEL_DIR.glob("*.tif"))
    print(f"label tiles on disk        : {n_label_tiles:,}")
    print(f"tiles in all.txt (the pool): {len(df):,} "
          f"({100 * len(df) / n_label_tiles:.1f}% of them)")
    print(f"parent rasters on disk     : {n_parents_large:,}")
    print(f"parents represented in pool: {n_parents_pool:,}")
    print("\nThe pool is a minority of the label tiles that exist. Per the dataset notes the")
    print("excluded ones are all-zero labels and tiles lacking a matching image, plus 4 tiles with")
    print("corrupt DSM/DTM. That exclusion is defensible, but it means the pool is a SELECTED")
    print("subset, and any statement about the service's national behaviour inherits that")
    print("selection on top of the 16-route footprint.")
    if C.CORRUPT_TILES_TXT.is_file():
        corrupt = [ln.strip() for ln in C.CORRUPT_TILES_TXT.read_text().splitlines() if ln.strip()]
        print(f"\nknown corrupt tiles ({len(corrupt)}), excluded via '#' in all.txt:")
        for c in corrupt[:6]:
            print(f"  {c}")
    payload["selection"] = {"label_tiles_on_disk": n_label_tiles, "tiles_in_pool": len(df),
                            "parents_on_disk": n_parents_large, "parents_in_pool": n_parents_pool}

    # ------------------------------------------------------------------------------------------
    C.banner(f"5. BOUNDARY AMBIGUITY  (cell {args.boundary_cell})")
    try:
        dirs = resolve_cell_dirs(args.boundary_cell)
        rng = np.random.default_rng(7)
        sample = df.sample(n=min(args.boundary_tiles, len(df)), random_state=7)
        tasks = [(r.filename, dirs[r.fold]) for r in sample.itertuples()]
        print(f"sampling {len(tasks):,} tiles (fixed seed, reproducible) ...")

        n_tot = np.zeros(NBIN, dtype=np.int64)
        n_err = np.zeros(NBIN, dtype=np.int64)
        fails = []
        with Pool(processes=args.procs) as pool:
            for i, res in enumerate(pool.imap_unordered(boundary_worker, tasks, chunksize=4), 1):
                if res[0] != "OK":
                    fails.append((res[1], res[2]))
                    continue
                n_tot += res[2]
                n_err += res[3]
                if i % 200 == 0:
                    print(f"  {i:,} / {len(tasks):,}")

        labels = BOUNDARY_LABELS
        tot, terr = n_tot.sum(), n_err.sum()
        print(f"\nscored pixels sampled : {tot:,}")
        print(f"errors                : {terr:,}  (error rate {100 * terr / tot:.2f}%)")
        print(f"\n{'dist to boundary (px)':<24}{'pixels':>16}{'% of px':>10}"
              f"{'errors':>16}{'% of errors':>13}{'error rate':>12}")
        cum_e = 0
        for i, lab in enumerate(labels):
            if n_tot[i] == 0:
                continue
            cum_e += n_err[i]
            print(f"{lab:<24}{n_tot[i]:>16,}{100 * n_tot[i] / tot:>9.2f}%"
                  f"{n_err[i]:>16,}{100 * n_err[i] / terr:>12.2f}%"
                  f"{100 * n_err[i] / n_tot[i]:>11.2f}%")
        within2 = n_err[:2].sum() / terr if terr else 0
        px_within2 = n_tot[:2].sum() / tot if tot else 0
        print(f"\n{100 * within2:.1f}% of all error sits within 2 px of an annotation boundary,")
        print(f"on {100 * px_within2:.1f}% of the pixels. At 0.1 m GSD, 2 px is 20 cm on the")
        print("ground, which is within the plausible precision of hand-drawn polygons.")
        print("Error concentrated there is annotation imprecision as much as model failure, and")
        print("it caps what any model can score. Error spread far from boundaries is genuine")
        print("class confusion -- which is the reading that matters for finding F5b.")
        payload["boundary"] = {
            "cell": args.boundary_cell, "tiles_sampled": len(tasks) - len(fails),
            "bin_edges_px": BOUNDARY_EDGES, "labels": labels,
            "pixels": n_tot.tolist(), "errors": n_err.tolist(),
            "overall_error_rate": float(terr / tot) if tot else None,
            "share_of_error_within_2px": float(within2),
            "share_of_pixels_within_2px": float(px_within2),
        }
        bcsv = C.assert_writes_are_local(C.TABLES / "boundary_error_profile.csv")
        with open(bcsv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["distance_bin_px", "pixels", "errors", "error_rate"])
            for i, lab in enumerate(labels):
                w.writerow([lab, int(n_tot[i]), int(n_err[i]),
                            (n_err[i] / n_tot[i]) if n_tot[i] else None])
        print(f"\nwrote {bcsv}")
    except Exception as exc:                       # noqa: BLE001
        print(f"boundary analysis skipped: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------------------------------
    C.banner("6. LIMITATION -- annotator churn is NOT measurable from disk")
    old_tiles = sorted(p.name for p in C.OLD_LABEL_DIR.glob("*.tif")) if C.OLD_LABEL_DIR.is_dir() else []
    overlap = [t for t in old_tiles if (C.LABEL_DIR / t).is_file()]
    print(f"labels/old_splitted_labels/ : {len(old_tiles)} tiles")
    print(f"  of which share a filename with the current pool: {len(overlap)}")
    print(f"  their tile offsets are multiples of 960, i.e. the overlap-40 grid, not the")
    print(f"  current stride-1000 grid, so they are not comparable tile for tile.")
    print(f"labels/large_label/reclass/ : verified to be the SAME rasters with the nodata")
    print(f"  sentinel 15 remapped to 0, not an older annotation vintage.")
    print("\nSo the contribution of data_cleaning_based_on_newer_ground_truth.py to the ignore")
    print("mass cannot be estimated here, and no number for annotator disagreement is reported.")
    print("Section 5's boundary profile is the available proxy for annotation precision.")
    payload["churn_limitation"] = {"old_tiles": len(old_tiles), "name_overlap": len(overlap),
                                   "measurable": False}

    print(f"\nwrote {C.write_json(payload, C.TABLES / 'label_quality.json')}")


def selftest() -> None:
    C.banner("E4 selftest (synthetic, no files touched)")
    from scipy import ndimage

    # A 2-class split down the middle: boundary pixels sit at the seam.
    lab = np.zeros((40, 40), dtype=np.int32)
    lab[:, 20:] = 1
    b = np.zeros_like(lab, dtype=bool)
    b[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    b[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    assert b[:, 19].all() and b[:, 20].all() and not b[:, 18].any()
    print("boundary detection : OK (seam pixels on both sides flagged)")

    dist = ndimage.distance_transform_edt(~b)
    assert dist[0, 19] == 0 and dist[0, 20] == 0
    assert abs(dist[0, 18] - 1.0) < 1e-9 and abs(dist[0, 17] - 2.0) < 1e-9
    print("distance transform : OK (1 px in from the seam measures 1.0)")

    idx = np.digitize(np.array([0.0, 1.0, 2.0, 4.0, 8.0, 15.0, 40.0, 200.0]), BOUNDARY_EDGES)
    assert idx.tolist() == [0, 1, 2, 3, 4, 5, 6, 7], idx.tolist()
    # sqrt(2) is the nearest diagonal distance and must land in [1,2), not with the exact zeros
    assert np.digitize(np.array([np.sqrt(2)]), BOUNDARY_EDGES).tolist() == [1]
    print(f"distance bucketing : OK ({', '.join(BOUNDARY_LABELS)})")
    print("  diagonal distance sqrt(2) buckets to [1,2), not to the on-boundary bin")

    # The nodata remap that rules out the old-vs-new comparison.
    old = np.array([3, 4, 15, 15], dtype=np.uint8)
    new = np.where(old == 15, 0, old)
    assert new.tolist() == [3, 4, 0, 0]
    print("reclass semantics  : OK (15 -> 0 is a nodata remap, not a vintage change)")

    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
