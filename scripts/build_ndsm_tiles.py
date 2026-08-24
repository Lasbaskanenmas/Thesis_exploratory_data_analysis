#!/usr/bin/env python
"""
Build the nDSM channel:  nDSM = max(DSM - DTM, 0), one tile per entry in all.txt.

Work order 2026-08-13 section 3. Object height above ground, as a directly supplied quantity, so the
network does not have to recover it as a small difference between two large, 92.6% correlated
channels (see results/findings/2026-08-14_preflight_ndsm.md, P0.2).

CLAMP POLICY (decided in the work order, and binding):
  * Floor clamp at 0. The surface model cannot sit below the terrain model, so negatives are noise,
    water or misregistration. It also neutralises the -9999 -> 0 m nodata artefact (SQ1 F9) without
    a separate mask.
  * NO ceiling clamp. A [0,40] m cutoff is an arbitrary parameter, it clips more than the stated 1%
    given p99 ~ 55 m, and the classes of interest sit at 4-5 m so the tail carries no class signal.
  * ORDER IS BINDING: clamp first, then measure statistics on the clamped rasters. The figures
    quoted on 2026-08-13 (0.01264671 / 0.03532309) were measured UNCLAMPED and must not be used.

SAFETY (this script is additive only):
  * Writes exclusively into data/splitted/nDSM/ and its own log under exploratory_data_analysis/.
  * Never deletes, moves or overwrites anything. Each tile is written to a .tmp file and then
    atomically renamed into place ONLY if the destination does not already exist, so a partially
    written file can never appear at a final path.
  * IDEMPOTENT: an existing destination is validated and skipped, so an interrupted run restarts
    safely without duplicating or corrupting output.
  * Refuses to continue if the projected write would exceed the 40 GB budget.

    python build_ndsm_tiles.py --selftest
    python build_ndsm_tiles.py --build            # generate (resumable)
    python build_ndsm_tiles.py --validate         # the section 3 assertions
    python build_ndsm_tiles.py --stats            # full-pool clamped statistics for the configs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

NDSM_DIR = C.SPLITTED_DIR / "nDSM"
DSM_DIR = C.SPLITTED_DIR / "DSM"
DTM_DIR = C.SPLITTED_DIR / "DTM"

LOG_DIR = C.EDA_ROOT / "logs"
STATS_JSON = C.TABLES / "ndsm_clamped_stats.json"
PER_TILE_CSV = C.TABLES / "ndsm_tile_stats.csv"

DISK_BUDGET_BYTES = 40 * 1024 ** 3          # hard stop, per the operating rules
NODATA_THRESHOLD = C.NODATA_THRESHOLD        # -100.0, matches ImageBlockReplacement.py:242


# --------------------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------------------
def build_one(name):
    """Create data/splitted/nDSM/<name> from the DSM/DTM pair. Returns a status dict.

    Never overwrites: writes <name>.tmp then renames only onto a free path.
    """
    import rasterio

    dst = NDSM_DIR / name
    if dst.is_file():
        return {"tile": name, "status": "skipped", "bytes": dst.stat().st_size}

    src_dsm, src_dtm = DSM_DIR / name, DTM_DIR / name
    if not src_dsm.is_file() or not src_dtm.is_file():
        return {"tile": name, "status": "missing_source"}

    with rasterio.open(src_dsm) as s_dsm:
        dsm = s_dsm.read(1).astype(np.float32)
        profile = s_dsm.profile.copy()
        transform, crs = s_dsm.transform, s_dsm.crs
    with rasterio.open(src_dtm) as s_dtm:
        dtm = s_dtm.read(1).astype(np.float32)
        if s_dtm.transform != transform:
            return {"tile": name, "status": "geotransform_mismatch"}

    if dsm.shape != dtm.shape:
        return {"tile": name, "status": "shape_mismatch"}

    # nodata -> 0 on BOTH inputs first, exactly as the loader does, then difference, then floor clamp.
    dsm = np.where(dsm < NODATA_THRESHOLD, 0.0, dsm)
    dtm = np.where(dtm < NODATA_THRESHOLD, 0.0, dtm)
    ndsm = np.maximum(dsm - dtm, 0.0).astype(np.float32)

    profile.update(dtype="float32", count=1, compress="deflate", predictor=3,
                   zlevel=6, tiled=False, nodata=None)
    profile.pop("photometric", None)

    tmp = NDSM_DIR / (name + ".tmp")
    with rasterio.open(tmp, "w", **profile) as out:
        out.write(ndsm, 1)
    try:
        # os.link + unlink of the tmp would also work; rename is atomic on NTFS for same-volume moves.
        if dst.is_file():                       # another worker won the race -- keep theirs
            tmp.unlink()
            return {"tile": name, "status": "skipped", "bytes": dst.stat().st_size}
        os.rename(tmp, dst)
    except OSError as exc:
        if tmp.is_file():
            tmp.unlink()
        return {"tile": name, "status": f"rename_failed: {exc}"}

    return {"tile": name, "status": "created", "bytes": dst.stat().st_size,
            "mean": float(ndsm.mean()), "std": float(ndsm.std()),
            "min": float(ndsm.min()), "max": float(ndsm.max()),
            "median": float(np.median(ndsm))}


# --------------------------------------------------------------------------------------------------
def build(workers=8, log_every=250):
    import multiprocessing as mp

    C.ensure_out_dirs()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    NDSM_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = C.assert_writes_are_local(LOG_DIR / f"ndsm_build_{stamp}.log")

    names = C.tile_names()
    free = _free_bytes(NDSM_DIR)

    def log(msg):
        line = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf8") as fh:
            fh.write(line + "\n")

    log(f"nDSM build start | {len(names):,} tiles | workers={workers}")
    log(f"destination {NDSM_DIR}")
    log(f"free disk {free / 1024**3:.1f} GB | budget {DISK_BUDGET_BYTES / 1024**3:.0f} GB")
    already = sum(1 for n in names if (NDSM_DIR / n).is_file())
    log(f"already present {already:,} -> {len(names) - already:,} to create (idempotent restart)")

    created = skipped = failed = 0
    total_bytes = 0
    t0 = time.time()
    with mp.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(build_one, names, chunksize=16), 1):
            st = res["status"]
            if st == "created":
                created += 1
                total_bytes += res.get("bytes", 0)
            elif st == "skipped":
                skipped += 1
                total_bytes += res.get("bytes", 0)
            else:
                failed += 1
                log(f"  FAIL {res['tile']}: {st}")

            if i % log_every == 0 or i == len(names):
                rate = i / max(time.time() - t0, 1e-9)
                projected = total_bytes / i * len(names)
                log(f"  {i:>6,}/{len(names):,}  created {created:,}  skipped {skipped:,}  "
                    f"failed {failed}  {rate:5.1f} tiles/s  "
                    f"written {total_bytes/1024**3:5.2f} GB  projected {projected/1024**3:5.2f} GB")
                if projected > DISK_BUDGET_BYTES:
                    log(f"STOP: projected {projected/1024**3:.1f} GB exceeds the "
                        f"{DISK_BUDGET_BYTES/1024**3:.0f} GB budget. Halting; nothing deleted.")
                    pool.terminate()
                    sys.exit(3)

    log(f"done in {(time.time()-t0)/60:.1f} min | created {created:,} skipped {skipped:,} "
        f"failed {failed} | {total_bytes/1024**3:.2f} GB")
    log(f"log written to {log_path}")
    if failed:
        sys.exit(f"{failed} tiles failed -- see {log_path}")
    return log_path


def fill_unresolved():
    """Write a zero-filled nDSM for the tiles whose DSM/DTM pair is misregistered.

    Decided 2026-08-15 after the georeferencing audit found 671 of 19,314 tiles (3.47%) where the
    DSM and DTM rasters sharing a filename cover different ground -- 669 of them >= 100 m apart, and
    a tile is only 100 m across, so there is no overlap and no height can be derived.

    Keeping them at zero rather than dropping them holds the evaluation pool at exactly 19,314, so
    the arm's pooled out-of-fold matrix stays directly comparable to the 24 frozen cells, and route
    composition is unchanged (the affected tiles cluster hard: 333 of them in route 82-20 alone).
    Those tiles simply carry no height signal, which is a documented limitation and strictly better
    than the wrong-place elevation the frozen 6ch/10ch runs were given for the same tiles.

    The zero raster is written on the *rgb* tile's grid, which is the reference the labels use.
    """
    import rasterio

    C.banner("nDSM -- zero-fill for tiles with a misregistered DSM/DTM pair")
    names = C.tile_names()
    todo = [n for n in names if not (NDSM_DIR / n).is_file()]
    print(f"  missing nDSM tiles: {len(todo):,} of {len(names):,}")
    if not todo:
        print("  nothing to do")
        return []

    rows = []
    zeros = np.zeros((C.TILE_PX, C.TILE_PX), dtype=np.float32)
    for i, name in enumerate(todo, 1):
        with rasterio.open(C.SPLITTED_DIR / "rgb" / name) as s:
            profile, rgb_t = s.profile.copy(), s.transform
        with rasterio.open(DSM_DIR / name) as s:
            dsm_t = s.transform
        with rasterio.open(DTM_DIR / name) as s:
            dtm_t = s.transform
        profile.update(dtype="float32", count=1, compress="deflate", predictor=3,
                       zlevel=6, tiled=False, nodata=None)
        profile.pop("photometric", None)

        tmp = NDSM_DIR / (name + ".tmp")
        with rasterio.open(tmp, "w", **profile) as out:
            out.write(zeros, 1)
        if (NDSM_DIR / name).is_file():
            tmp.unlink()
        else:
            os.rename(tmp, NDSM_DIR / name)

        rows.append({
            "filename": name, "route": C.parse_filename(name)["route"],
            "reason": "DSM/DTM geotransform mismatch -- no co-located terrain model",
            "dsm_dtm_offset_m": float(np.hypot(dsm_t.c - dtm_t.c, dsm_t.f - dtm_t.f)),
            "dsm_offset_from_rgb_m": float(np.hypot(dsm_t.c - rgb_t.c, dsm_t.f - rgb_t.f)),
            "dtm_offset_from_rgb_m": float(np.hypot(dtm_t.c - rgb_t.c, dtm_t.f - rgb_t.f)),
            "ndsm_value": 0.0,
        })
        if i % 200 == 0:
            print(f"    ...{i:,}/{len(todo):,}", flush=True)

    C.ensure_out_dirs()
    out_csv = C.assert_writes_are_local(C.TABLES / "ndsm_unresolved_tiles.csv")
    with open(out_csv, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    by_route = {}
    for r in rows:
        by_route[r["route"]] = by_route.get(r["route"], 0) + 1
    print(f"\n  zero-filled {len(rows):,} tiles")
    print(f"  by route: " + ", ".join(f"{k} {v}" for k, v in sorted(by_route.items(), key=lambda kv: -kv[1])))
    print(f"  wrote {out_csv}")
    return rows


def _free_bytes(path):
    import shutil
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


# --------------------------------------------------------------------------------------------------
def validate(spot_check=5):
    """The section 3 assertions. Every one is a hard stop."""
    import rasterio

    C.banner("nDSM validation -- section 3 assertions")
    names = C.tile_names()
    rng = np.random.default_rng(0)

    # 1. file count and names
    present = {p.name for p in NDSM_DIR.glob("*.tif")}
    missing = [n for n in names if n not in present]
    extra = sorted(present - set(names))
    stray = sorted(p.name for p in NDSM_DIR.glob("*.tmp"))
    print(f"  1. files            : {len(present):,} present, expected {len(names):,}")
    print(f"     missing          : {len(missing)}   extra: {len(extra)}   stray .tmp: {len(stray)}")
    if missing or extra or stray:
        sys.exit(f"ASSERTION 1 FAILED (missing {len(missing)}, extra {len(extra)}, tmp {len(stray)})")

    # 2-4. geometry, dtype, crs, transform vs DSM source; global min; median
    gmin = np.inf
    gmax = -np.inf
    medians = []
    bad = []
    for i, n in enumerate(names, 1):
        with rasterio.open(NDSM_DIR / n) as s:
            if (s.width, s.height, s.count) != (C.TILE_PX, C.TILE_PX, 1):
                bad.append((n, f"shape {s.width}x{s.height}x{s.count}"))
                continue
            if s.dtypes[0] != "float32":
                bad.append((n, f"dtype {s.dtypes[0]}"))
                continue
            if s.crs is None or s.crs.to_epsg() != C.EPSG:
                bad.append((n, f"crs {s.crs}"))
                continue
            a = s.read(1)
            t = s.transform
        # Compared against the RGB tile, not the DSM. rgb is the reference grid the labels use, and
        # the audit found the DSM itself misplaced on 100 tiles, so DSM is not a sound reference.
        with rasterio.open(C.SPLITTED_DIR / "rgb" / n) as d:
            if d.transform != t:
                bad.append((n, "geotransform != rgb tile (the label grid)"))
                continue
        gmin = min(gmin, float(a.min()))
        gmax = max(gmax, float(a.max()))
        medians.append(float(np.median(a)))
        if bad and len(bad) > 20:
            break
        if i % 2000 == 0:
            print(f"     ...{i:,}/{len(names):,} checked", flush=True)

    print(f"  2. geometry/dtype/crs/transform : {'OK' if not bad else str(len(bad)) + ' BAD'}")
    if bad:
        for n, why in bad[:10]:
            print(f"       {n}: {why}")
        sys.exit("ASSERTION 2 FAILED")

    print(f"  3. global minimum   : {gmin:.6f}  (must be >= 0)   max {gmax:.3f}")
    if gmin < 0:
        sys.exit("ASSERTION 3 FAILED: negative values present after clamping")

    med = float(np.median(medians))
    print(f"  4. median of per-tile medians : {med:.6f} m  (must be ~0.000)")
    if abs(med) > 0.05:
        sys.exit(f"ASSERTION 4 FAILED: median {med:.4f} m drifted from 0 -- possible misregistration")

    # 5. spot-check against a manual numpy difference
    print(f"  5. spot-check {spot_check} tiles against manual numpy DSM-DTM:")
    picks = [names[i] for i in rng.choice(len(names), size=spot_check, replace=False)]
    for n in picks:
        with rasterio.open(DSM_DIR / n) as s:
            dsm = s.read(1).astype(np.float32)
        with rasterio.open(DTM_DIR / n) as s:
            dtm = s.read(1).astype(np.float32)
        dsm = np.where(dsm < NODATA_THRESHOLD, 0.0, dsm)
        dtm = np.where(dtm < NODATA_THRESHOLD, 0.0, dtm)
        expect = np.maximum(dsm - dtm, 0.0).astype(np.float32)
        with rasterio.open(NDSM_DIR / n) as s:
            got = s.read(1)
        err = float(np.abs(got - expect).max())
        print(f"       {n:<48} max abs err {err:.3e}  {'OK' if err == 0 else 'MISMATCH'}")
        if err != 0:
            sys.exit("ASSERTION 5 FAILED")

    print("\n  ALL SECTION 3 ASSERTIONS PASS")
    return True


# --------------------------------------------------------------------------------------------------
def stats():
    """Full-pool statistics on the CLAMPED rasters, and the config constants derived from them."""
    import rasterio

    C.banner("nDSM full-pool statistics (clamped) + corrected config constants")
    names = C.tile_names()
    n = len(names)
    means = np.zeros(n); stds = np.zeros(n)
    mins = np.zeros(n); maxs = np.zeros(n)

    for i, name in enumerate(names):
        with rasterio.open(NDSM_DIR / name) as s:
            a = s.read(1).astype(np.float64)
        means[i] = a.mean(); stds[i] = a.std()
        mins[i] = a.min(); maxs[i] = a.max()
        if (i + 1) % 2000 == 0:
            print(f"  ...{i+1:,}/{n:,}", flush=True)

    between = float(np.var(means))
    within = float(np.mean(stds ** 2))
    total = between + within
    pool_mean = float(np.mean(means))
    pool_sd = float(np.sqrt(total))

    print(f"\n  tiles                : {n:,}")
    print(f"  pooled mean          : {pool_mean:.6f} m")
    print(f"  pooled sd            : {pool_sd:.6f} m   "
          f"(between-tile {np.sqrt(between):.4f}, within-tile {np.sqrt(within):.4f}, "
          f"between share {100*between/total:.1f}%)")
    print(f"  global min / max     : {mins.min():.4f} / {maxs.max():.4f} m")

    # The pipeline divides every channel by 255 before Normalize (measured, P0.1), so the config
    # constants live in post-/255 space.
    cfg_mean = pool_mean / C.INT_TO_FLOAT_DIV
    cfg_std = pool_sd / C.INT_TO_FLOAT_DIV
    print(f"\n  CONFIG CONSTANTS (post-/255 space, for means/stds in the .ini):")
    print(f"    mean = {cfg_mean:.10f}")
    print(f"    std  = {cfg_std:.10f}")
    lo = (mins.min() / C.INT_TO_FLOAT_DIV - cfg_mean) / cfg_std
    hi = (maxs.max() / C.INT_TO_FLOAT_DIV - cfg_mean) / cfg_std
    print(f"    -> normalised range [{lo:.4f}, {hi:.4f}], mean 0.000, sd 1.000")

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(), "n_tiles": n,
           "clamp": "floor at 0, no ceiling", "measured_on": "clamped rasters",
           "pooled_mean_m": pool_mean, "pooled_sd_m": pool_sd,
           "between_tile_sd_m": float(np.sqrt(between)), "within_tile_sd_m": float(np.sqrt(within)),
           "between_share": between / total,
           "global_min_m": float(mins.min()), "global_max_m": float(maxs.max()),
           "int_to_float_div": C.INT_TO_FLOAT_DIV,
           "config_mean": cfg_mean, "config_std": cfg_std,
           "normalised_range": [lo, hi]}
    C.ensure_out_dirs()
    C.write_json(out, STATS_JSON)
    with open(C.assert_writes_are_local(PER_TILE_CSV), "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "ndsm_mean", "ndsm_std", "ndsm_min", "ndsm_max"])
        for i, name in enumerate(names):
            w.writerow([name, means[i], stds[i], mins[i], maxs[i]])
    print(f"\n  wrote {STATS_JSON}")
    print(f"  wrote {PER_TILE_CSV}")
    return out


# --------------------------------------------------------------------------------------------------
def selftest():
    C.banner("build_ndsm_tiles self-test")
    dsm = np.array([[10.0, 12.0], [-9999.0, 20.0]], dtype=np.float32)
    dtm = np.array([[9.0, 12.5], [5.0, -9999.0]], dtype=np.float32)
    dsm_c = np.where(dsm < NODATA_THRESHOLD, 0.0, dsm)
    dtm_c = np.where(dtm < NODATA_THRESHOLD, 0.0, dtm)
    ndsm = np.maximum(dsm_c - dtm_c, 0.0)
    expect = np.array([[1.0, 0.0], [0.0, 20.0]], dtype=np.float32)
    assert np.array_equal(ndsm, expect), (ndsm, expect)
    print("  clamp + nodata rule OK")
    print("    10.0-9.0   -> 1.0   (normal)")
    print("    12.0-12.5  -> 0.0   (negative floor-clamped, not -0.5)")
    print("    nodata DSM -> 0.0   (sentinel neutralised, no fabricated cliff)")
    print("    nodata DTM -> 20.0  (DTM sentinel -> 0, so nDSM = DSM)")
    assert float(ndsm.min()) >= 0.0
    print("  minimum >= 0 OK")
    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fill-unresolved", dest="fill_unresolved", action="store_true",
                    help="zero-fill tiles whose DSM/DTM pair is misregistered")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.selftest:
        selftest(); return
    if not (args.build or args.fill_unresolved or args.validate or args.stats):
        ap.error("nothing to do: pass --build, --fill-unresolved, --validate, --stats or --selftest")
    if args.build:
        build(workers=args.workers)
    if args.fill_unresolved:
        fill_unresolved()
    if args.validate:
        validate()
    if args.stats:
        stats()


if __name__ == "__main__":
    main()
