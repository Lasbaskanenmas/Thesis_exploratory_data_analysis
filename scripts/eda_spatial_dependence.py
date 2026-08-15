#!/usr/bin/env python
"""
E3 -- Spatial dependence: the empirical warrant for spatial cross-validation, measured on KDS's
own data instead of cited from the literature.

WHY THIS EXISTS
    Great Plan 3.0 section 2 names the precise weakness of the thesis as it stands: the need for
    route-blocked spatial CV is justified by CITING Roberts 2017, Karasiak 2022 and Kattenborn
    2022, not by DEMONSTRATING it on Befaestelsesdata. This module supplies the demonstration, so
    SQ2's instrument is argued from measurement rather than from authority.

WHAT IT MEASURES
    1. Empirical semivariogram of tile composition against distance. The distance at which the
       variogram reaches its sill is the AUTOCORRELATION RANGE -- the single number the blocking
       design has to respect (Roberts 2017's design rule).
    2. Moran's I across the same distance bands, as an independent read on the same structure.
    3. The variogram split three ways: pairs within one parent orthophoto, pairs in the same
       flight route but different parents, and pairs in different routes. This tests the specific
       claim behind the route-over-parent decision -- that parent-level blocking leaves residual
       correlation because parents in a route share flight, day, light and season (Karasiak 2022).
    4. The naive-split leakage demonstration: how far each of the 25 hand-picked valid tiles sits
       from its nearest TRAINING tile, against the same distance under route blocking. This is
       Sub-1's claim reduced to one figure.
    5. Per-route composition, so the covariate shift between folds is visible.

    Tiles are 1000 x 1000 px at 0.1 m GSD, so a tile is 100 m x 100 m and adjacent tiles are 100 m
    apart centre to centre. Read every distance below against that scale.

READ-ONLY over the source data. Writes only under exploratory_data_analysis/.

    python eda_spatial_dependence.py [--chunk N] [--selftest]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402


# ----------------------------------------------------------------------------------------------
# Distance bins: fine at the tile-adjacency scale, geometric out to national extent.
# ----------------------------------------------------------------------------------------------
def make_bins() -> np.ndarray:
    fine = np.arange(0.0, 2001.0, 100.0)                    # 0 to 2 km in tile-width steps
    coarse = np.geomspace(2000.0, 400_000.0, 40)[1:]        # 2 km out to 400 km
    return np.concatenate([fine, coarse])


STRATA = ["same_parent", "same_route_diff_parent", "diff_route"]


def pairwise_scan(xy: np.ndarray, z: np.ndarray, route_id: np.ndarray, parent_id: np.ndarray,
                  bin_edges: np.ndarray, chunk: int = 512, verbose: bool = True) -> dict:
    """One chunked pass over all j>i pairs, accumulating the variogram and Moran cross-products.

    Both statistics come from the same pass, stratified by spatial relationship, so the whole of
    E3's core is a single sweep over the ~186 million tile pairs.
    """
    n = len(xy)
    nb = len(bin_edges) - 1
    zc = z - z.mean()

    acc = {
        "count": np.zeros(nb, dtype=np.int64),
        "gamma_sum": np.zeros(nb, dtype=np.float64),        # sum of (zi - zj)^2
        "cross_sum": np.zeros(nb, dtype=np.float64),        # sum of zc_i * zc_j
    }
    strat = {s: {"count": np.zeros(nb, dtype=np.int64),
                 "gamma_sum": np.zeros(nb, dtype=np.float64)} for s in STRATA}

    idx = np.arange(n)
    xy32 = xy.astype(np.float32)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        rows = idx[start:end]

        d = np.sqrt(((xy32[rows][:, None, :] - xy32[None, :, :]) ** 2).sum(axis=-1))
        upper = idx[None, :] > rows[:, None]                # keep each pair once

        b = np.digitize(d, bin_edges) - 1
        keep = upper & (b >= 0) & (b < nb)
        if not keep.any():
            continue

        bk = b[keep]
        dz = (z[rows][:, None] - z[None, :])[keep]
        cz = (zc[rows][:, None] * zc[None, :])[keep]

        acc["count"] += np.bincount(bk, minlength=nb)
        acc["gamma_sum"] += np.bincount(bk, weights=dz.astype(np.float64) ** 2, minlength=nb)
        acc["cross_sum"] += np.bincount(bk, weights=cz.astype(np.float64), minlength=nb)

        same_r = (route_id[rows][:, None] == route_id[None, :])
        same_p = (parent_id[rows][:, None] == parent_id[None, :])
        for name, m in (("same_parent", same_p),
                        ("same_route_diff_parent", same_r & ~same_p),
                        ("diff_route", ~same_r)):
            mk = keep & m
            if not mk.any():
                continue
            bm = b[mk]
            dzm = (z[rows][:, None] - z[None, :])[mk]
            strat[name]["count"] += np.bincount(bm, minlength=nb)
            strat[name]["gamma_sum"] += np.bincount(bm, weights=dzm.astype(np.float64) ** 2,
                                                    minlength=nb)

        if verbose and (start // chunk) % 5 == 0:
            print(f"  pairs scanned through tile {end:,} / {n:,}")

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = np.where(acc["count"] > 0, acc["gamma_sum"] / (2.0 * acc["count"]), np.nan)
        moran = np.where(acc["count"] > 0,
                         n * acc["cross_sum"] / (acc["count"] * (zc ** 2).sum()), np.nan)
        strat_gamma = {s: np.where(strat[s]["count"] > 0,
                                   strat[s]["gamma_sum"] / (2.0 * strat[s]["count"]), np.nan)
                       for s in STRATA}
    return {
        "bin_edges": bin_edges, "centers": centers,
        "count": acc["count"], "gamma": gamma, "moran": moran,
        "strata_gamma": strat_gamma,
        "strata_count": {s: strat[s]["count"] for s in STRATA},
        "sill": float(zc.var()),
    }


def autocorrelation_range(centers: np.ndarray, gamma: np.ndarray, sill: float,
                          frac: float = 0.95) -> float | None:
    """First distance at which the variogram reaches `frac` of the sill (the practical range)."""
    ok = np.isfinite(gamma)
    if not ok.any():
        return None
    hit = np.where(ok & (gamma >= frac * sill))[0]
    return float(centers[hit[0]]) if len(hit) else None


def nearest_other_distance(xy_a: np.ndarray, xy_b: np.ndarray, chunk: int = 256) -> np.ndarray:
    """For each point in A, the distance to the nearest point in B."""
    out = np.empty(len(xy_a), dtype=np.float64)
    b32 = xy_b.astype(np.float32)
    for s in range(0, len(xy_a), chunk):
        e = min(s + chunk, len(xy_a))
        d = np.sqrt(((xy_a[s:e].astype(np.float32)[:, None, :] - b32[None, :, :]) ** 2).sum(-1))
        out[s:e] = d.min(axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="E3 -- spatial dependence")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    C.ensure_out_dirs()
    df = C.load_tile_inventory()
    if len(df) < C.EXPECTED_TILES:
        print(f"WARNING: inventory holds {len(df)} tiles, expected {C.EXPECTED_TILES}. "
              f"Re-run eda_tile_inventory.py without --limit for the real numbers.\n")

    C.banner(f"E3 -- spatial dependence over {len(df):,} tiles")

    xy = df[["centroid_e", "centroid_n"]].to_numpy(dtype=np.float64)
    scored = df["n_scored_px"].to_numpy(dtype=np.float64)
    ubef = df[C.px_col("ubefestet")].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        sealed_frac = np.where(scored > 0, 1.0 - ubef / scored, np.nan)

    # Tiles with no scored pixels carry no composition signal; drop them from the variogram only.
    keep = np.isfinite(sealed_frac)
    print(f"tiles with scored pixels : {keep.sum():,} of {len(df):,} "
          f"({100 * keep.sum() / len(df):.2f}%)")
    print(f"variable                 : sealed fraction (1 - ubefestet share of scored pixels)")
    print(f"  mean {np.nanmean(sealed_frac):.4f}   sd {np.nanstd(sealed_frac):.4f}")

    route_id = df["route"].astype("category").cat.codes.to_numpy()
    parent_id = df["parent"].astype("category").cat.codes.to_numpy()

    print(f"\nscanning {keep.sum() * (keep.sum() - 1) // 2:,} tile pairs "
          f"(chunk={args.chunk}) ...")
    res = pairwise_scan(xy[keep], sealed_frac[keep], route_id[keep], parent_id[keep],
                        make_bins(), chunk=args.chunk)

    sill = res["sill"]
    rng95 = autocorrelation_range(res["centers"], res["gamma"], sill, 0.95)
    rng50 = autocorrelation_range(res["centers"], res["gamma"], sill, 0.50)

    # ------------------------------------------------------------------------------------------
    C.banner("SEMIVARIOGRAM  (sill = a priori variance; range = where gamma reaches it)")
    print(f"sill (variance of sealed fraction) : {sill:.5f}")
    print(f"practical range at 50% of sill     : "
          f"{'not reached' if rng50 is None else f'{rng50:,.0f} m'}")
    print(f"practical range at 95% of sill     : "
          f"{'not reached' if rng95 is None else f'{rng95:,.0f} m'}")
    print(f"tile size for reference            : {C.TILE_GROUND_M:,.0f} m")
    print(f"\n{'lag (m)':>12}{'pairs':>14}{'gamma':>10}{'gamma/sill':>12}{'Moran I':>10}")
    for i, c in enumerate(res["centers"]):
        if res["count"][i] == 0 or c > 120_000:
            continue
        if c <= 2000 or i % 3 == 0:
            print(f"{c:>12,.0f}{res['count'][i]:>14,}{res['gamma'][i]:>10.5f}"
                  f"{res['gamma'][i] / sill:>12.3f}{res['moran'][i]:>10.4f}")

    # ------------------------------------------------------------------------------------------
    C.banner("ROUTE vs PARENT BLOCKING  (the Karasiak argument, tested here)")
    print("Semivariance by spatial relationship. Values BELOW the sill mean the pair is still")
    print("correlated, so splitting at that level would leak.\n")
    print(f"{'stratum':<26}{'pairs':>14}{'mean gamma':>13}{'/ sill':>9}")
    for s in STRATA:
        cnt = res["strata_count"][s]
        g = res["strata_gamma"][s]
        ok = cnt > 0
        if not ok.any():
            continue
        mg = float(np.average(g[ok], weights=cnt[ok]))
        print(f"{s:<26}{cnt.sum():>14,}{mg:>13.5f}{mg / sill:>9.3f}")
    print("\nReading: if same_route_diff_parent sits clearly below the sill, then parents inside a")
    print("route remain correlated, and blocking on the parent orthophoto would leave that")
    print("correlation straddling the split. That is the measured case for route-level blocking.")

    # ------------------------------------------------------------------------------------------
    C.banner("NAIVE SPLIT vs ROUTE BLOCKING  (Sub-1 in one figure)")
    naive_valid = set(C.read_tile_list(str(C.NAIVE_VALID_TXT)))
    is_naive_valid = df["filename"].isin(naive_valid).to_numpy()
    leak = {}
    if is_naive_valid.sum():
        d_naive = nearest_other_distance(xy[is_naive_valid], xy[~is_naive_valid])
        leak["naive"] = d_naive
        print(f"naive 25-tile valid set ({is_naive_valid.sum()} tiles found in the pool)")
        print(f"  distance to the nearest TRAINING tile:")
        print(f"    min {d_naive.min():>10,.0f} m   median {np.median(d_naive):>10,.0f} m"
              f"   max {d_naive.max():>10,.0f} m")
        print(f"    tiles whose nearest training tile is adjacent (<= 150 m): "
              f"{int((d_naive <= 150).sum())} of {len(d_naive)}")
    else:
        print("no valid.txt tiles found in the pool -- skipping the naive leg")

    for f in range(C.NFOLDS):
        held = (df["fold"] == f).to_numpy()
        d_sp = nearest_other_distance(xy[held], xy[~held])
        leak[f"fold_{f}"] = d_sp
        print(f"\nroute-blocked fold {f} ({held.sum():,} held-out tiles)")
        print(f"  distance to the nearest TRAINING tile:")
        print(f"    min {d_sp.min():>10,.0f} m   median {np.median(d_sp):>10,.0f} m"
              f"   max {d_sp.max():>10,.0f} m")

    # ------------------------------------------------------------------------------------------
    C.banner("PER-ROUTE COMPOSITION  (the covariate shift the folds inherit)")
    rows = []
    print(f"{'route':<9}{'fold':>5}{'tiles':>8}{'year':>10}{'sealed%':>9}   dominant classes")
    for route, g in df.groupby("route"):
        sc = g["n_scored_px"].sum()
        sealed = 1.0 - g[C.px_col("ubefestet")].sum() / sc if sc else float("nan")
        shares = {n: g[C.px_col(n)].sum() / sc for n in C.CODES
                  if n not in ("unknown", "unknown2") and sc}
        top = sorted(shares.items(), key=lambda kv: -kv[1])[:3]
        yr = "+".join(str(y) for y in sorted(g["year"].unique()))
        print(f"{route:<9}{g['fold'].iloc[0]:>5}{len(g):>8}{yr:>10}{100 * sealed:>9.2f}   "
              + ", ".join(f"{k} {100 * v:.1f}%" for k, v in top))
        rec = {"route": route, "fold": int(g["fold"].iloc[0]), "tiles": len(g), "years": yr,
               "scored_px": int(sc), "sealed_fraction": sealed,
               "centroid_e": float(g["centroid_e"].mean()),
               "centroid_n": float(g["centroid_n"].mean())}
        rec.update({f"share_{n}": shares.get(n, 0.0) for n in C.CODES})
        rows.append(rec)

    route_csv = C.assert_writes_are_local(C.TABLES / "route_composition.csv")
    with open(route_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {route_csv}")

    # ------------------------------------------------------------------------------------------
    vario_csv = C.assert_writes_are_local(C.TABLES / "semivariogram.csv")
    with open(vario_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lag_center_m", "n_pairs", "gamma", "gamma_over_sill", "moran_i"]
                   + [f"gamma_{s}" for s in STRATA] + [f"n_pairs_{s}" for s in STRATA])
        for i, c in enumerate(res["centers"]):
            w.writerow([c, int(res["count"][i]), res["gamma"][i],
                        res["gamma"][i] / sill if sill else None, res["moran"][i]]
                       + [res["strata_gamma"][s][i] for s in STRATA]
                       + [int(res["strata_count"][s][i]) for s in STRATA])
    print(f"wrote {vario_csv}")

    payload = {
        "n_tiles": int(len(df)),
        "n_tiles_with_scored_px": int(keep.sum()),
        "variable": "sealed_fraction = 1 - ubefestet / scored pixels",
        "tile_ground_m": C.TILE_GROUND_M,
        "sill_variance": sill,
        "range_50pct_sill_m": rng50,
        "range_95pct_sill_m": rng95,
        "mean_gamma_over_sill_by_stratum": {
            s: (float(np.average(res["strata_gamma"][s][res["strata_count"][s] > 0],
                                 weights=res["strata_count"][s][res["strata_count"][s] > 0]))
                / sill if (res["strata_count"][s] > 0).any() else None)
            for s in STRATA},
        "nearest_training_tile_distance_m": {
            k: {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max()),
                "n": int(len(v)), "n_within_150m": int((v <= 150).sum())}
            for k, v in leak.items()},
    }
    out = C.write_json(payload, C.TABLES / "spatial_dependence.json")
    print(f"wrote {out}")

    np.savez_compressed(C.assert_writes_are_local(C.TABLES / "spatial_dependence_curves.npz"),
                        centers=res["centers"], gamma=res["gamma"], moran=res["moran"],
                        count=res["count"], sill=np.array([sill]),
                        **{f"gamma_{s}": res["strata_gamma"][s] for s in STRATA},
                        **{f"count_{s}": res["strata_count"][s] for s in STRATA},
                        **{f"nearest_{k}": v for k, v in leak.items()})
    print(f"wrote {C.TABLES / 'spatial_dependence_curves.npz'}")


def selftest() -> None:
    """Verify the estimators against cases with a known answer."""
    C.banner("E3 selftest (synthetic, no files touched)")
    rng = np.random.default_rng(0)

    bins = np.array([0.0, 150.0, 500.0, 5000.0, 1e6])

    # Case 1: pure noise on a grid -> no spatial structure. gamma should sit at the sill at every
    # lag and Moran's I should hover around zero.
    g = np.stack(np.meshgrid(np.arange(24) * 100.0, np.arange(24) * 100.0), -1).reshape(-1, 2)
    z = rng.normal(size=len(g))
    rid = np.zeros(len(g), dtype=int)
    pid = np.arange(len(g))
    r = pairwise_scan(g, z, rid, pid, bins, chunk=128, verbose=False)
    ratio = r["gamma"][0] / r["sill"]
    assert 0.85 < ratio < 1.15, f"white noise gamma/sill at the shortest lag = {ratio}"
    assert abs(r["moran"][0]) < 0.15, f"white noise Moran I = {r['moran'][0]}"
    print(f"white noise      : gamma/sill = {ratio:.3f}, Moran I = {r['moran'][0]:+.3f}  OK")
    assert autocorrelation_range(r["centers"], r["gamma"], r["sill"], 0.95) == r["centers"][0]
    print("  range detected at the first lag, i.e. no autocorrelation           OK")

    # Case 2: a smooth east-west gradient -> strong short-range correlation. gamma must rise with
    # distance and Moran's I must be clearly positive nearby.
    z2 = g[:, 0] / 1000.0
    r2 = pairwise_scan(g, z2, rid, pid, bins, chunk=128, verbose=False)
    assert r2["gamma"][0] < r2["gamma"][1] < r2["gamma"][2], r2["gamma"][:3]
    assert r2["moran"][0] > 0.5, r2["moran"][0]
    assert r2["gamma"][0] / r2["sill"] < 0.2, r2["gamma"][0] / r2["sill"]
    print(f"smooth gradient  : gamma rises with lag, Moran I = {r2['moran'][0]:+.3f}  OK")
    print("  gamma at the shortest lag is far below the sill, i.e. neighbours are")
    print("  strongly correlated -- which is what the real data is expected to show   OK")

    # Case 3: stratification. Two clusters far apart, constant within each -> within-cluster pairs
    # have zero semivariance, across-cluster pairs carry all of it.
    a = np.stack([np.arange(20) * 100.0, np.zeros(20)], -1)
    b = a + np.array([500_000.0, 0.0])
    xy = np.vstack([a, b])
    z3 = np.concatenate([np.zeros(20), np.ones(20)])
    rid3 = np.concatenate([np.zeros(20, int), np.ones(20, int)])
    r3 = pairwise_scan(xy, z3, rid3, rid3, bins, chunk=16, verbose=False)
    within = r3["strata_gamma"]["same_parent"]
    cnt = r3["strata_count"]["same_parent"]
    assert np.nansum(within[cnt > 0]) == 0.0, within[cnt > 0]
    print("stratification   : within-cluster semivariance is exactly 0            OK")

    # Case 4: nearest-neighbour distance
    d = nearest_other_distance(np.array([[0.0, 0.0]]), np.array([[300.0, 400.0], [0.0, 900.0]]))
    assert abs(d[0] - 500.0) < 1e-6, d
    print("nearest-distance : 3-4-5 triangle gives 500 m                          OK")

    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
