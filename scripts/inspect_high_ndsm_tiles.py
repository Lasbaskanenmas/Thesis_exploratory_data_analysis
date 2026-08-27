#!/usr/bin/env python
"""
Task 2.3 of the 2026-08-24 work order: the 8 tiles whose nDSM exceeds 100 m.

`2026-08-15_stage1_ndsm_arms.md` section 4 records a clamped nDSM maximum of 298.33 m against a
pooled mean of 2.57 m and sd 5.68 m -- a 52-sigma tail on 8 tiles (0.04 percent). It notes the
extent premise still holds and no ceiling clamp was applied, but nobody has looked at what those
tiles actually are. Three outcomes are possible and all three are reportable:

    DSM spike              a surface-model artefact (noise, cloud, bird, reflective surface)
    residual misregistration   DSM and DTM describing different ground despite passing the audit
    real structure         a genuinely tall object (mast, chimney, silo, crane)

Which one it is decides one sentence in the data-quality section, and it bounds what the elevation
arms could ever have achieved.

Method, per tile: read nDSM, DSM, DTM and rgb; confirm all four share a geotransform (the audit's
own alignment test, re-run per tile); locate the maximum; measure how many pixels sit above 100 m
and above 50 m; and profile the neighbourhood of the peak -- a single-pixel spike, a compact blob,
or a broad plateau discriminate the three hypotheses directly. A tile whose DSM peak sits at the
nDSM peak while DTM is smooth there is a surface feature; a tile whose DTM is wrong under the peak
is misregistration.

Read-only over the data. Writes one CSV and one findings note under exploratory_data_analysis/.

    python inspect_high_ndsm_tiles.py [--threshold 100.0]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

STATS_CSV = C.TABLES / "ndsm_tile_stats.csv"
NDSM_DIR = C.SPLITTED_DIR / "nDSM"
DSM_DIR = C.SPLITTED_DIR / "DSM"
DTM_DIR = C.SPLITTED_DIR / "DTM"
RGB_DIR = C.SPLITTED_DIR / "rgb"
UNRESOLVED_CSV = C.TABLES / "ndsm_unresolved_tiles.csv"
HALF = 25          # neighbourhood half-width in pixels around the peak (50x50 px = 5x5 m)


def classify(rec):
    """Hypothesis from the measured profile. Conservative: says 'inconclusive' rather than guess."""
    if rec["misregistered_pair"]:
        return "residual misregistration (tile is on the 671 list)"
    if not rec["geotransforms_agree"]:
        return "residual misregistration (geotransforms disagree)"
    # A spike is tiny in extent and steep; a structure is compact but not single-pixel; a plateau
    # over many hundreds of pixels at >100 m is not a plausible built object at 0.1 m GSD.
    n100 = rec["px_above_100m"]
    if n100 <= 4:
        return "DSM spike (isolated, <=4 px above threshold)"
    if rec["dtm_std_in_window_m"] > 5.0:
        return "residual misregistration (terrain model unstable under the peak)"
    if n100 <= 2000:
        return "real structure (compact, stable terrain beneath)"
    return "inconclusive (broad high area; inspect visually)"


def main():
    ap = argparse.ArgumentParser(description="inspect the tiles with nDSM above a threshold")
    ap.add_argument("--threshold", type=float, default=100.0)
    args = ap.parse_args()

    import rasterio

    with open(STATS_CSV, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if float(r["ndsm_max"]) > args.threshold]
    rows.sort(key=lambda r: -float(r["ndsm_max"]))
    print(f"{len(rows)} tiles with nDSM max > {args.threshold} m "
          f"(of {C.EXPECTED_TILES:,} in the pool)")
    if not rows:
        sys.exit("nothing above threshold -- nothing to inspect")

    unresolved = set()
    if UNRESOLVED_CSV.is_file():
        with open(UNRESOLVED_CSV, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                unresolved.add(r.get("filename") or next(iter(r.values())))

    out = []
    for r in rows:
        name = r["filename"]
        rec = {"filename": name, "route": C.parse_route(name),
               "ndsm_max_m": float(r["ndsm_max"]), "ndsm_mean_m": float(r["ndsm_mean"]),
               "misregistered_pair": name in unresolved}

        arrays, transforms = {}, {}
        for tag, folder in (("nDSM", NDSM_DIR), ("DSM", DSM_DIR), ("DTM", DTM_DIR), ("rgb", RGB_DIR)):
            p = folder / name
            if not p.is_file():
                rec[f"{tag}_present"] = False
                continue
            rec[f"{tag}_present"] = True
            with rasterio.open(p) as src:
                arrays[tag] = src.read(1).astype("float64")
                transforms[tag] = tuple(round(v, 6) for v in src.transform[:6])
        rec["geotransforms_agree"] = len(set(transforms.values())) == 1

        nd = arrays["nDSM"]
        rec["px_above_100m"] = int((nd > 100.0).sum())
        rec["px_above_50m"] = int((nd > 50.0).sum())
        rec["px_above_20m"] = int((nd > 20.0).sum())
        rec["tile_px"] = int(nd.size)

        iy, ix = np.unravel_index(int(np.argmax(nd)), nd.shape)
        rec["peak_row"], rec["peak_col"] = int(iy), int(ix)
        y0, y1 = max(0, iy - HALF), min(nd.shape[0], iy + HALF + 1)
        x0, x1 = max(0, ix - HALF), min(nd.shape[1], ix + HALF + 1)
        win_nd = nd[y0:y1, x0:x1]
        win_dsm = arrays["DSM"][y0:y1, x0:x1]
        win_dtm = arrays["DTM"][y0:y1, x0:x1]

        rec["dsm_at_peak_m"] = float(arrays["DSM"][iy, ix])
        rec["dtm_at_peak_m"] = float(arrays["DTM"][iy, ix])
        rec["dtm_median_tile_m"] = float(np.median(arrays["DTM"]))
        rec["dtm_std_in_window_m"] = float(win_dtm.std())
        rec["ndsm_mean_in_window_m"] = float(win_nd.mean())
        rec["ndsm_frac_window_above_100m"] = float((win_nd > 100.0).mean())
        rec["dsm_std_in_window_m"] = float(win_dsm.std())
        rec["verdict"] = classify(rec)
        out.append(rec)
        print(f"\n  {name}  route {rec['route']}")
        print(f"    nDSM max {rec['ndsm_max_m']:.2f} m   px>100m {rec['px_above_100m']}   "
              f"px>50m {rec['px_above_50m']}   px>20m {rec['px_above_20m']}")
        print(f"    at peak: DSM {rec['dsm_at_peak_m']:.2f} m, DTM {rec['dtm_at_peak_m']:.2f} m "
              f"(tile median DTM {rec['dtm_median_tile_m']:.2f} m)")
        print(f"    5x5 m window: nDSM mean {rec['ndsm_mean_in_window_m']:.2f} m, "
              f"DTM sd {rec['dtm_std_in_window_m']:.3f} m, DSM sd {rec['dsm_std_in_window_m']:.2f} m")
        print(f"    geotransforms agree: {rec['geotransforms_agree']}   "
              f"on the 671 misregistered list: {rec['misregistered_pair']}")
        print(f"    -> {rec['verdict']}")

    dest = C.assert_writes_are_local(C.TABLES / "high_ndsm_tiles.csv")
    with open(dest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {dest}")

    verdicts = {}
    for rec in out:
        verdicts.setdefault(rec["verdict"], []).append(rec["filename"])
    meta = C.assert_writes_are_local(C.TABLES / "high_ndsm_tiles.json")
    meta.write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "threshold_m": args.threshold, "n_tiles": len(out),
        "pool_tiles": C.EXPECTED_TILES,
        "verdicts": {k: len(v) for k, v in verdicts.items()},
        "tiles": out,
    }, indent=2), encoding="utf-8")
    print(f"wrote {meta}")
    print("\n=== verdict tally ===")
    for k, v in sorted(verdicts.items(), key=lambda kv: -len(kv[1])):
        print(f"  {len(v):>2}  {k}")


if __name__ == "__main__":
    main()
