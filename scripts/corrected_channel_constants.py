#!/usr/bin/env python
"""
Full-pool, measured normalisation constants for the `6ch_corrected` arm (work order section 4).

The 72-run matrix used these for the non-RGB bands:
    cir NIR : mean 0.40779021, std 0.15176421   -- documented in sdfi_dataset.py as
                                                   "based on a sample size of 1! OBS! UNKNOWN"
    DSM     : mean 0.5,        std 1.0          -- placeholders, while the tiles hold raw metres
    DTM     : mean 0.5,        std 1.0          -- placeholders

This derives replacements from `channel_per_class_stats.csv`, which carries per-class per-band mean,
std and pixel count over all 14 bands of the full 19,314-tile pool. Pooling across classes by the
law of total variance recovers the exact full-pool moments:

    pooled_mean = sum(n_c * mean_c) / sum(n_c)
    pooled_var  = sum(n_c * (var_c + mean_c^2)) / sum(n_c) - pooled_mean^2

The class pixel counts sum to exactly 19,314,000,000 = 19,314 tiles x 1000 x 1000, so this is the
whole image pool and not a subset. `unknown` is included: it is the ignore index for the LOSS, but
those pixels are still real imagery the network sees, so they belong in a normalisation constant.

RGB is deliberately left at the ImageNet values. The work order specifies the non-RGB constants, and
keeping RGB at the pretrained convention means `6ch_corrected` differs from `6ch` in the auxiliary
bands only, which is the clean contrast.

Constants are emitted in post-/255 space, because the pipeline divides every channel by 255 before
Normalize -- measured, not assumed (P0.1, 2026-08-14).

    python corrected_channel_constants.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

SRC_CSV = C.TABLES / "channel_per_class_stats.csv"
OUT_JSON = C.TABLES / "corrected_channel_constants.json"

# ImageNet values, kept for the RGB bands.
RGB_MEANS = [0.485, 0.456, 0.406]
RGB_STDS = [0.229, 0.224, 0.225]

# What the frozen matrix actually used, for the before/after table.
MATRIX_CONSTANTS = {
    "cir_b0_NIR": (0.40779021, 0.15176421),
    "DSM": (0.5, 1.0),
    "DTM": (0.5, 1.0),
}


def pooled_moments(rows):
    """rows: list of (n, mean, std) -> (pooled_mean, pooled_std) by the law of total variance."""
    n = np.array([r[0] for r in rows], dtype=np.float64)
    m = np.array([r[1] for r in rows], dtype=np.float64)
    s = np.array([r[2] for r in rows], dtype=np.float64)
    tot = n.sum()
    mean = float((n * m).sum() / tot)
    ex2 = float((n * (s ** 2 + m ** 2)).sum() / tot)
    return mean, float(np.sqrt(max(ex2 - mean ** 2, 0.0))), tot


def main():
    import csv

    C.banner("Full-pool measured channel constants (work order section 4)")
    if not SRC_CSV.is_file():
        sys.exit(f"missing {SRC_CSV}")

    by_band = {}
    with open(SRC_CSV, newline="", encoding="utf8") as fh:
        for row in csv.DictReader(fh):
            by_band.setdefault(row["band"], []).append(
                (float(row["pixels"]), float(row["mean"]), float(row["std"])))

    print(f"  bands in source table : {len(by_band)}")
    results = {}
    for band, rows in by_band.items():
        mean_raw, std_raw, tot = pooled_moments(rows)
        results[band] = {"pixels": tot, "mean_raw": mean_raw, "std_raw": std_raw,
                         "mean_div255": mean_raw / C.INT_TO_FLOAT_DIV,
                         "std_div255": std_raw / C.INT_TO_FLOAT_DIV}

    # sanity: the imagery bands should cover the whole pool
    px = results["rgb_R"]["pixels"]
    expect = C.EXPECTED_TILES * C.TILE_PX * C.TILE_PX
    print(f"  rgb_R pixel total     : {px:,.0f}  (expected {expect:,})  "
          f"{'OK' if abs(px - expect) < 1 else 'MISMATCH'}")
    if abs(px - expect) >= 1:
        sys.exit("the per-class table does not cover the full pool -- refusing to emit constants")

    print(f"\n  {'band':<14}{'measured mean (raw)':>21}{'measured std (raw)':>20}"
          f"{'mean /255':>12}{'std /255':>11}")
    for band in ["rgb_R", "rgb_G", "rgb_B", "cir_b0_NIR", "DSM", "DTM"]:
        r = results[band]
        print(f"  {band:<14}{r['mean_raw']:>21.6f}{r['std_raw']:>20.6f}"
              f"{r['mean_div255']:>12.8f}{r['std_div255']:>11.8f}")

    print(f"\n  WHAT CHANGES for 6ch_corrected (RGB stays at ImageNet by design):")
    print(f"  {'band':<14}{'matrix mean':>14}{'matrix std':>12}{'corrected mean':>17}{'corrected std':>15}")
    corrected = {}
    for band, (om, os_) in MATRIX_CONSTANTS.items():
        r = results[band]
        corrected[band] = (r["mean_div255"], r["std_div255"])
        print(f"  {band:<14}{om:>14.8f}{os_:>12.8f}{r['mean_div255']:>17.8f}{r['std_div255']:>15.8f}")

    # the effective std each band presents to the network, before and after
    print(f"\n  EFFECTIVE STD PRESENTED TO THE NETWORK (measured_std/255 / config_std):")
    print(f"  {'band':<14}{'with matrix constants':>23}{'with corrected constants':>26}")
    for band, (om, os_) in MATRIX_CONSTANTS.items():
        true_sd = results[band]["std_div255"]
        print(f"  {band:<14}{true_sd/os_:>23.4f}{true_sd/corrected[band][1]:>26.4f}")
    for band in ["rgb_R"]:
        true_sd = results[band]["std_div255"]
        print(f"  {band:<14}{true_sd/0.229:>23.4f}{'(unchanged)':>26}")

    means_6ch = RGB_MEANS + [corrected["cir_b0_NIR"][0], corrected["DSM"][0], corrected["DTM"][0]]
    stds_6ch = RGB_STDS + [corrected["cir_b0_NIR"][1], corrected["DSM"][1], corrected["DTM"][1]]
    print("\n  6ch_corrected config vectors:")
    print(f"    means = {json.dumps([round(v, 10) for v in means_6ch])}")
    print(f"    stds  = {json.dumps([round(v, 10) for v in stds_6ch])}")

    out = {"generated_utc": datetime.now(timezone.utc).isoformat(),
           "source": str(SRC_CSV), "int_to_float_div": C.INT_TO_FLOAT_DIV,
           "pool_pixels": px, "per_band": results,
           "matrix_constants": {k: {"mean": v[0], "std": v[1]} for k, v in MATRIX_CONSTANTS.items()},
           "corrected_constants": {k: {"mean": v[0], "std": v[1]} for k, v in corrected.items()},
           "rgb_policy": "left at ImageNet 0.485/0.456/0.406, 0.229/0.224/0.225 by design",
           "means_6ch_corrected": means_6ch, "stds_6ch_corrected": stds_6ch}
    C.ensure_out_dirs()
    C.write_json(out, OUT_JSON)
    print(f"\n  wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
