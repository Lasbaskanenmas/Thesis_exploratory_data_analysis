#!/usr/bin/env python
"""
Do the per-channel tiles that share a filename actually cover the same ground?

Found while building the nDSM channel on 2026-08-15: for a fraction of tiles the DSM and DTM rasters
with identical filenames have different geotransforms, offset by tens to hundreds of metres. A tile
is 1000 x 1000 at 0.1 m = 100 m x 100 m of ground, so an offset of 45 m or more means the two
rasters barely overlap, and 600 m means they are unrelated ground.

This matters beyond nDSM. `ImageBlockReplacement.load_all_datasources_for_image` stacks the channel
folders purely by FILENAME and never compares georeferencing, so wherever this happens the training
sample's channels disagree about where they are, and the label mask can only match one of them.

Header reads only, no pixel data. Read-only; writes one table under exploratory_data_analysis/.

    python eda_channel_georeferencing.py
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

OUT_CSV = C.TABLES / "channel_georeferencing_audit.csv"
OUT_JSON = C.TABLES / "channel_georeferencing_audit.json"
FOLDERS = ["rgb", "cir", "OrtoRGB", "OrtoCIR", "DSM", "DTM"]
TOL_M = 0.05          # half a pixel; anything under this is rounding, not misregistration


def main():
    import rasterio

    C.banner("Per-channel georeferencing audit -- do same-named tiles cover the same ground?")
    names = C.tile_names()
    print(f"  tiles: {len(names):,}   folders: {FOLDERS}   tolerance: {TOL_M} m (half a pixel)\n")

    rows = []
    counts = {f: 0 for f in FOLDERS}
    for i, name in enumerate(names, 1):
        origins = {}
        for f in FOLDERS:
            p = C.SPLITTED_DIR / f / name
            if not p.is_file():
                continue
            with rasterio.open(p) as s:
                origins[f] = (s.transform.c, s.transform.f, s.transform.a, s.transform.e,
                              s.width, s.height)
        if "rgb" not in origins:
            continue
        rx, ry = origins["rgb"][0], origins["rgb"][1]
        row = {"filename": name, "route": C.parse_filename(name)["route"]}
        worst = 0.0
        for f in FOLDERS:
            if f not in origins:
                row[f"{f}_dx"] = row[f"{f}_dy"] = None
                continue
            dx = origins[f][0] - rx
            dy = origins[f][1] - ry
            row[f"{f}_dx"] = dx
            row[f"{f}_dy"] = dy
            off = float(np.hypot(dx, dy))
            if off > TOL_M:
                counts[f] += 1
            worst = max(worst, off)
        row["max_offset_from_rgb_m"] = worst
        row["dsm_dtm_offset_m"] = float(np.hypot(
            origins["DSM"][0] - origins["DTM"][0],
            origins["DSM"][1] - origins["DTM"][1])) if {"DSM", "DTM"} <= origins.keys() else None
        rows.append(row)
        if i % 4000 == 0:
            print(f"    ...{i:,}/{len(names):,}", flush=True)

    print(f"\n  MISALIGNED vs the rgb tile (offset > {TOL_M} m):")
    for f in FOLDERS:
        pct = 100.0 * counts[f] / len(rows)
        print(f"    {f:<10}{counts[f]:>7,} / {len(rows):,}   ({pct:5.2f} %)")

    dd = np.array([r["dsm_dtm_offset_m"] for r in rows if r["dsm_dtm_offset_m"] is not None])
    bad = dd > TOL_M
    print(f"\n  DSM vs DTM offset (the pair nDSM subtracts):")
    print(f"    misaligned : {int(bad.sum()):,} / {len(dd):,}  ({100.0*bad.mean():.2f} %)")
    if bad.any():
        b = dd[bad]
        print(f"    offsets    : min {b.min():.1f} m   median {np.median(b):.1f} m   max {b.max():.1f} m")
        print(f"    >= 100 m (no overlap at all) : {int((b >= 100).sum()):,}")

    # which channel is the odd one out, per affected tile
    aligned_with_rgb = {f: 0 for f in ["DSM", "DTM"]}
    for r in rows:
        if r["dsm_dtm_offset_m"] and r["dsm_dtm_offset_m"] > TOL_M:
            for f in ["DSM", "DTM"]:
                if r[f"{f}_dx"] is not None and np.hypot(r[f"{f}_dx"], r[f"{f}_dy"]) <= TOL_M:
                    aligned_with_rgb[f] += 1
    print(f"\n  Of the misaligned DSM/DTM pairs, which one still matches the rgb tile (= the label)?")
    print(f"    DSM matches rgb : {aligned_with_rgb['DSM']:,}")
    print(f"    DTM matches rgb : {aligned_with_rgb['DTM']:,}")
    print(f"    neither         : "
          f"{int(bad.sum()) - aligned_with_rgb['DSM'] - aligned_with_rgb['DTM']:,}")

    by_route = {}
    for r in rows:
        if r["dsm_dtm_offset_m"] and r["dsm_dtm_offset_m"] > TOL_M:
            by_route[r["route"]] = by_route.get(r["route"], 0) + 1
    if by_route:
        print(f"\n  affected tiles by route:")
        for rt, n in sorted(by_route.items(), key=lambda kv: -kv[1]):
            print(f"    {rt:<8}{n:>6,}")

    C.ensure_out_dirs()
    with open(C.assert_writes_are_local(OUT_CSV), "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    C.write_json({"generated_utc": datetime.now(timezone.utc).isoformat(),
                  "tolerance_m": TOL_M, "n_tiles": len(rows),
                  "misaligned_vs_rgb": counts,
                  "dsm_dtm_misaligned": int(bad.sum()),
                  "dsm_dtm_offsets_m": {"min": float(dd[bad].min()) if bad.any() else None,
                                        "median": float(np.median(dd[bad])) if bad.any() else None,
                                        "max": float(dd[bad].max()) if bad.any() else None},
                  "aligned_with_rgb": aligned_with_rgb,
                  "affected_by_route": by_route}, OUT_JSON)
    print(f"\n  wrote {OUT_CSV}")
    print(f"  wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
