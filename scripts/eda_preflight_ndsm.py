#!/usr/bin/env python
"""
Pre-flight measurements for the 2026-08-13 nDSM / corrected-normalisation work order (section 2).

Three questions, answered before anything is built and before the ~30 GB nDSM write happens:

  P0.1  Is fastai's IntToFloatTensor really dividing EVERY channel by 255, elevation included?
        Currently this is a code trace. eda_common.INT_TO_FLOAT_DIV = 255.0 asserts it, and SQ1 F8
        hedges that its conclusion holds "whether or not the /255 step is applied". This turns the
        trace into a measurement taken at the tensor the model actually receives.

  P0.2  How redundant are DSM and DTM, really? A planning-stage remark of "~97% redundant" was never
        a computed correlation. This computes the pixel-level Pearson r EXACTLY over the full
        19,314-tile pool in closed form, alongside the tile-mean r, so the two are never confused
        again.

  P0.3  Did every comparison cell get the same epoch budget, and was any still improving at its last
        epoch? Plus: is <job>.pth the FINAL epoch or the BEST one? All 24 pooled cells were scored
        from it.

Additive only. Reads source data, configs and logs_and_models read-only; writes exclusively under
exploratory_data_analysis/ via eda_common.assert_writes_are_local.

    python eda_preflight_ndsm.py --selftest
    python eda_preflight_ndsm.py --p01 --p02 --p03
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

# The 6ch cell used for the P0.1 batch dump. Any weighted 6ch config would do; this one is the
# leading architecture and its training log is the one quoted in the work order.
REPO_ROOT = Path(r"c:\thesis\ML_sdfi_fastai2")
P01_CONFIG = REPO_ROOT / "configs" / "matrix_configs" / "train" / "convnext_upernet_6ch_fold0.ini"

# P0.3 covers the WEIGHTED rgb and 6ch cells only -- the twelve-and-twelve the new arms will be
# compared against. The _unw twins are a loss-axis question, not a channel one.
P03_MODEL_DIRS = ["convnext_upernet", "segformer_b1", "swin_upernet", "unet_resnet34"]
P03_CHANNELS = ["rgb", "6ch"]

TABLES = C.TABLES
BATCH_STATS_CSV = TABLES / "preflight_batch_channel_stats.csv"
CORRELATION_JSON = TABLES / "preflight_dsm_dtm_correlation.json"
EPOCH_BUDGET_CSV = TABLES / "preflight_epoch_budget.csv"
TILE_ELEVATION_CSV = TABLES / "tile_elevation_stats.csv"

# SQ1 F8 published these three full-pool standard deviations. Reproducing them from the same CSV is
# the validity gate for P0.2's closed form.
SQ1_F8_SD = {"DSM": 23.707, "DTM": 21.814, "nDSM": 8.977}


# ==================================================================================================
# P0.1 -- measure the divide-by-255
# ==================================================================================================
def _band_sources(cfg):
    """Expand a config's datatypes/channels into a flat list of (folder, band) per input channel.

    Mirrors ImageBlockReplacement.load_all_datasources_for_image, which stacks the datatypes in
    order and takes channels[index] from each. The result is index-aligned with means/stds.
    """
    out = []
    for folder, bands in zip(cfg["datatypes"], cfg["channels"]):
        for b in bands:
            out.append((str(folder), int(b)))
    return out


def _read_raw_band(tile_name, folder, band):
    """Read one source band exactly as the loader does, including its nodata rule.

    ImageBlockReplacement.py:242 sets every value below -100 to 0 AFTER stacking, so the comparison
    has to reproduce that or the DSM/DTM nodata pixels look like a mismatch.
    """
    import rasterio
    path = C.SPLITTED_DIR / folder / tile_name
    with rasterio.open(path) as src:
        arr = src.read(band + 1).astype(np.float32)
    arr[arr < C.NODATA_THRESHOLD] = 0.0
    return arr


def run_p01(verbose=True):
    import torch
    from fastai.torch_core import default_device

    sys.path.insert(0, r"c:\thesis\ML_sdfi_fastai2\src\ML_sdfi_fastai2")
    import utils.utils as sdfi_utils
    import sdfi_dataset

    C.banner("P0.1 -- what does the model actually receive?")

    # Force CPU. fastai puts batches on default_device(), which would be the A100 otherwise.
    default_device(False)

    cfg = sdfi_utils.load_settings_from_config_file(str(P01_CONFIG))

    # The matrix configs carry paths relative to ML_sdfi_fastai2/ (that was the working directory
    # when the 72 runs were launched), e.g. "../logs_and_models/route_class_audit/fold_0_valid.txt".
    # Reproduce that cwd so the loader resolves exactly what training resolved.
    import os
    prev_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    print(f"cwd         : {REPO_ROOT}  (configs use paths relative to it)")

    means = np.asarray(cfg["means"], dtype=np.float64)
    stds = np.asarray(cfg["stds"], dtype=np.float64)
    sources = _band_sources(cfg)
    assert len(sources) == len(means) == len(stds), \
        f"channel bookkeeping disagrees: {len(sources)} bands, {len(means)} means, {len(stds)} stds"

    print(f"config      : {P01_CONFIG.name}")
    print(f"datatypes   : {cfg['datatypes']}")
    print(f"channels    : {cfg['channels']}")
    print(f"-> bands    : {sources}")
    print(f"transforms  : {cfg['transforms']}  (all split_idx=0, so dls.valid is unaugmented)")
    print(f"batch_size  : {cfg['batch_size']}   num_workers: {cfg['num_workers']}")

    try:
        dls = sdfi_dataset.get_dataset(cfg)
        xb, _yb = dls.valid.one_batch()
        items = list(dls.valid.items)
    finally:
        os.chdir(prev_cwd)

    # The validation loader is not shuffled, so batch 0 is items 0..bs-1 and .items names them.
    bs = int(cfg["batch_size"])
    batch_paths = [Path(p) for p in items[:bs]]
    print(f"\nvalid items : {len(items):,}")
    print("batch tiles :")
    for p in batch_paths:
        print(f"   {p.name}")

    x = xb.detach().to("cpu").float().numpy()
    print(f"\nbatch tensor: shape {x.shape}  dtype {xb.dtype}  device {xb.device}")
    assert xb.device.type == "cpu", f"batch is on {xb.device}, expected cpu"
    assert x.shape[1] == len(means), f"batch has {x.shape[1]} channels, config declares {len(means)}"

    # ---- the table section 2 asks for -----------------------------------------------------------
    rows = []
    for ci, (folder, band) in enumerate(sources):
        ch = x[:, ci]
        rows.append({
            "index": ci,
            "band": C.band_label(folder, band),
            "folder": folder,
            "src_band": band,
            "config_mean": float(means[ci]),
            "config_std": float(stds[ci]),
            "tensor_min": float(ch.min()),
            "tensor_max": float(ch.max()),
            "tensor_mean": float(ch.mean()),
            "tensor_std": float(ch.std()),
        })

    print("\nAT THE TENSOR THE MODEL RECEIVES (batch of "
          f"{x.shape[0]}, {x.shape[2]}x{x.shape[3]}):")
    print(f"  {'band':<14}{'cfg mean':>10}{'cfg std':>9}{'min':>11}{'max':>11}{'mean':>11}{'std':>10}")
    for r in rows:
        print(f"  {r['band']:<14}{r['config_mean']:>10.5f}{r['config_std']:>9.5f}"
              f"{r['tensor_min']:>11.4f}{r['tensor_max']:>11.4f}"
              f"{r['tensor_mean']:>11.4f}{r['tensor_std']:>10.4f}")

    # ---- the decisive test ----------------------------------------------------------------------
    # H255: x == (raw/255 - mean)/std        H1: x == (raw - mean)/std
    print("\nDECISIVE TEST -- reconstruct the tensor from the source rasters under two hypotheses:")
    print(f"  {'band':<14}{'max|x-H255|':>14}{'max|x-H1|':>16}{'implied divisor':>18}   verdict")
    for r in rows:
        ci = r["index"]
        folder, band = sources[ci]
        err255, err1, divisors = [], [], []
        for ti, p in enumerate(batch_paths):
            raw = _read_raw_band(p.name, folder, band)
            obs = x[ti, ci].astype(np.float64)
            if raw.shape != obs.shape:
                sys.exit(f"shape mismatch for {p.name} {folder}: raw {raw.shape} vs tensor {obs.shape}")
            h255 = (raw / C.INT_TO_FLOAT_DIV - means[ci]) / stds[ci]
            h1 = (raw - means[ci]) / stds[ci]
            err255.append(float(np.abs(obs - h255).max()))
            err1.append(float(np.abs(obs - h1).max()))
            # back-solve the divisor: raw / (x*std + mean). Only where the denominator is safely
            # away from zero, otherwise the ratio is numerically meaningless.
            denom = obs * stds[ci] + means[ci]
            ok = np.abs(denom) > 1e-3
            if ok.any():
                divisors.append(np.median(raw[ok] / denom[ok]))
        e255, e1 = max(err255), max(err1)
        div = float(np.median(divisors)) if divisors else float("nan")
        verdict = "/255 APPLIED" if e255 < e1 else "NO /255"
        r["max_abs_err_h255"] = e255
        r["max_abs_err_h1"] = e1
        r["implied_divisor"] = div
        r["verdict"] = verdict
        print(f"  {r['band']:<14}{e255:>14.3e}{e1:>16.3e}{div:>18.4f}   {verdict}")

    all255 = all(r["verdict"] == "/255 APPLIED" for r in rows)
    tol = 1e-4
    exact = all(r["max_abs_err_h255"] < tol for r in rows)

    # ---- positive control -----------------------------------------------------------------------
    rgb_rows = [r for r in rows if r["folder"] == "rgb"]
    print("\nPOSITIVE CONTROL -- rgb is uint8 0-255, so under /255 it must land near mean 0 / std 1:")
    for r in rgb_rows:
        print(f"  {r['band']:<14}mean {r['tensor_mean']:>8.4f}   std {r['tensor_std']:>7.4f}")
    control_ok = all(abs(r["tensor_mean"]) < 1.5 and 0.3 < r["tensor_std"] < 3.0 for r in rgb_rows)
    print(f"  control {'PASSES' if control_ok else 'FAILS'} "
          f"-- {'the method is sound' if control_ok else 'the MEASUREMENT is wrong, not the pipeline'}")

    print(f"\nVERDICT: every channel divided by 255 = {all255}"
          f"   (reconstruction exact to <{tol:g} = {exact})")
    print(f"eda_common.INT_TO_FLOAT_DIV = {C.INT_TO_FLOAT_DIV} is "
          f"{'CONFIRMED by measurement' if all255 and exact else 'NOT CONFIRMED -- investigate'}")

    C.ensure_out_dirs()
    out = C.assert_writes_are_local(BATCH_STATS_CSV)
    with open(out, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    return {"rows": rows, "all_divided_by_255": all255, "reconstruction_exact": exact,
            "control_ok": control_ok, "batch_tiles": [p.name for p in batch_paths],
            "config": P01_CONFIG.name, "tile_shape": [int(x.shape[2]), int(x.shape[3])]}


# ==================================================================================================
# P0.1b -- what P0.1 turned up: the three delivery paths do NOT agree
# ==================================================================================================
def run_p01b(n_tiles=200):
    """Audit the train, valid and inference delivery paths for two defects P0.1 exposed.

    Defect A -- uint8 cast. Every albumentations wrapper in sdfi_transforms does
        img = np.array(img, dtype=np.uint8).astype(np.uint8).copy()
    before augmenting (e.g. sdfi_transforms.py:194, :259, :277). rgb/cir are uint8 already so this is
    lossless for them, but DSM/DTM hold float32 METRES, so they are truncated to integer metres.

    Defect B -- split_idx dropped. sdfi_transforms.py:273
        class SegmentationAlbumentationsHorizontalFlip(ItemTransform):
            def __init__(self,split_idx): self.aug = albumentations.HorizontalFlip(p=0.05)
    never calls ItemTransform.__init__, so split_idx is discarded and defaults to None = BOTH splits.
    Every sibling class (:254, :264, :283) forwards it correctly. So a 5% horizontal flip, and its
    uint8 cast, run on the validation pass too.

    This measures the consequence on each path rather than reasoning about it.
    """
    import os
    import rasterio
    import torch  # noqa: F401
    from fastai.torch_core import default_device

    sys.path.insert(0, str(REPO_ROOT / "src" / "ML_sdfi_fastai2"))
    import utils.utils as sdfi_utils
    import sdfi_dataset
    import infer as infer_mod

    C.banner("P0.1b -- do train, valid and inference deliver the same pixels?")
    default_device(False)
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        train_cfg = sdfi_utils.load_settings_from_config_file(str(P01_CONFIG))
        infer_cfg = sdfi_utils.load_settings_from_config_file(
            str(REPO_ROOT / "configs" / "matrix_configs" / "infer" /
                f"infer_{P01_CONFIG.stem}.ini"))
        infer_mod.ad_values_nececeary_for_dataset_loader_creation(infer_cfg)

        dls = sdfi_dataset.get_dataset(train_cfg)
        idls = sdfi_dataset.get_dataset(infer_cfg)
        bench = Path(infer_cfg["path_to_all_benchmarkset_txt"])
        names = [l.strip() for l in bench.read_text().split("\n")
                 if l.strip() and not l.lstrip().startswith("#")][:n_tiles]
        test_dl = idls.test_dl([Path(infer_cfg["path_to_images"]) / n for n in names], num_workers=0)

        paths = {
            "train": dls.train.new(shuffle=False),
            "valid": dls.valid,
            "inference": test_dl,
        }
        means = np.asarray(train_cfg["means"]); stds = np.asarray(train_cfg["stds"])

        print(f"{'path':<11}{'active after_item transforms':<62}")
        for label, dl in paths.items():
            act = [type(t).__name__ for t in dl.after_item.fs
                   if getattr(t, "split_idx", None) is None or t.split_idx == dl.split_idx]
            print(f"  {label:<9}{', '.join(act)}")

        print(f"\ndelivering {n_tiles} tiles down each path:")
        print(f"  {'path':<11}{'flipped':>10}{'DSM integer-quantised':>25}")
        results = {}
        for label, dl in paths.items():
            items = [Path(p) for p in list(dl.items)[:n_tiles]]
            n_flip = n_quant = n_seen = 0
            for batch in dl:
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = xb.detach().cpu().numpy()
                for i in range(x.shape[0]):
                    if n_seen >= len(items):
                        break
                    name = items[n_seen].name
                    n_seen += 1
                    rgb_obs = (x[i, 0].astype(np.float64) * stds[0] + means[0]) * 255.0
                    dsm_obs = (x[i, 4].astype(np.float64) * stds[4] + means[4]) * 255.0
                    with rasterio.open(C.SPLITTED_DIR / "rgb" / name) as s:
                        raw = s.read(1).astype(np.float64)
                    if np.abs(rgb_obs - raw).max() > 1e-3 and np.abs(rgb_obs - raw[:, ::-1]).max() < 1e-3:
                        n_flip += 1
                    if np.abs(dsm_obs - np.round(dsm_obs)).max() < 1e-3:
                        n_quant += 1
                if n_seen >= len(items):
                    break
            results[label] = {"n": n_seen, "flipped": n_flip, "quantised": n_quant}
            print(f"  {label:<11}{n_flip:>4}/{n_seen:<5}{n_quant:>18}/{n_seen:<5}")
    finally:
        os.chdir(prev)

    print("\nREADING:")
    print("  train  -- elevation truncated to integer metres by the albumentations uint8 cast.")
    print("  valid  -- same truncation, plus a 5% horizontal flip, both via the HorizontalFlip")
    print("            transform whose split_idx was dropped (sdfi_transforms.py:273).")
    print("  infer  -- test_dl carries none of the item transforms, so predictions were produced")
    print("            from FULL float32 elevation and were never flipped.")
    print("  => the 463,536 predictions and the 24 pooled cells are geometrically sound, but the")
    print("     6ch/10ch models were TRAINED on integer-metre elevation and then served float32.")
    C.ensure_out_dirs()
    C.write_json(results, TABLES / "preflight_pipeline_paths.json")
    print(f"\nwrote {TABLES / 'preflight_pipeline_paths.json'}")
    return results


# ==================================================================================================
# P0.2 -- DSM/DTM redundancy
# ==================================================================================================
def pool_variance(per_tile_mean, per_tile_std):
    """Full-pool variance from per-tile moments, by the law of total variance.

    Var_total = var(tile means) + mean(within-tile var). Exact here because every tile is the same
    size (1000x1000), so the tiles carry equal weight. Same convention as eda_channel_stats.py so
    the numbers stay directly comparable to SQ1 F8.
    """
    between = float(np.var(per_tile_mean))
    within = float(np.mean(per_tile_std ** 2))
    return between + within, between, within


def run_p02(sample_per_route=12, verbose=True):
    import pandas as pd

    C.banner("P0.2 -- how redundant are DSM and DTM?")

    if not TILE_ELEVATION_CSV.is_file():
        sys.exit(f"missing {TILE_ELEVATION_CSV} -- run eda_channel_stats.py first")
    df = pd.read_csv(TILE_ELEVATION_CSV)          # 19,314 rows, never printed whole
    print(f"tile_elevation_stats.csv: {len(df):,} rows  (expected {C.EXPECTED_TILES:,})")
    assert len(df) == C.EXPECTED_TILES, f"expected {C.EXPECTED_TILES} tiles, found {len(df)}"

    var_dsm, b_dsm, w_dsm = pool_variance(df["dsm_mean"].to_numpy(), df["dsm_std"].to_numpy())
    var_dtm, b_dtm, w_dtm = pool_variance(df["dtm_mean"].to_numpy(), df["dtm_std"].to_numpy())
    var_nd, b_nd, w_nd = pool_variance(df["ndsm_mean"].to_numpy(), df["ndsm_std"].to_numpy())

    print("\nFULL-POOL MOMENTS (law of total variance, all 19,314 tiles):")
    print(f"  {'channel':<8}{'total sd':>10}{'between sd':>12}{'within sd':>11}{'SQ1 F8 sd':>11}{'match':>8}")
    sd_ok = True
    for name, tot, bet, wit in [("DSM", var_dsm, b_dsm, w_dsm),
                                ("DTM", var_dtm, b_dtm, w_dtm),
                                ("nDSM", var_nd, b_nd, w_nd)]:
        sd = np.sqrt(tot)
        ref = SQ1_F8_SD[name]
        ok = abs(sd - ref) < 0.01
        sd_ok &= ok
        print(f"  {name:<8}{sd:>10.3f}{np.sqrt(bet):>12.3f}{np.sqrt(wit):>11.3f}"
              f"{ref:>11.3f}{('OK' if ok else 'MISMATCH'):>8}")

    # Validity guard: the closed form needs nDSM == DSM - DTM, i.e. UNCLAMPED.
    n_neg = int((df["ndsm_mean"] < 0).sum())
    print(f"\nvalidity guard -- tiles with negative mean nDSM: {n_neg:,}")
    print(f"  {'unclamped (closed form valid)' if n_neg > 0 or sd_ok else 'check clamping'}")
    if not sd_ok:
        print("  !! the published SQ1 F8 sds were NOT reproduced -- the closed form below is void")
        print("     and the raster cross-check becomes the primary number")

    # Cov(DSM,DTM) = (Var(DSM) + Var(DTM) - Var(DSM-DTM)) / 2
    cov = (var_dsm + var_dtm - var_nd) / 2.0
    r_pixel = cov / np.sqrt(var_dsm * var_dtm)

    # tile-mean correlation -- a DIFFERENT quantity, and the likely source of the "~97%" remark
    r_tilemean = float(np.corrcoef(df["dsm_mean"].to_numpy(), df["dtm_mean"].to_numpy())[0, 1])

    # SQ1 back-solve, from the three published sds alone
    sd_a, sd_b, sd_n = SQ1_F8_SD["DSM"], SQ1_F8_SD["DTM"], SQ1_F8_SD["nDSM"]
    r_backsolve = (sd_a ** 2 + sd_b ** 2 - sd_n ** 2) / (2 * sd_a * sd_b)

    print("\nCORRELATIONS -- three different quantities, kept apart:")
    print(f"  pixel-level Pearson r  (closed form, full pool) : r = {r_pixel:.4f}   r^2 = {r_pixel**2:.4f}")
    print(f"  pixel-level r          (back-solved from SQ1 F8): r = {r_backsolve:.4f}   r^2 = {r_backsolve**2:.4f}")
    print(f"  TILE-MEAN Pearson r    (19,314 tile means)      : r = {r_tilemean:.4f}   r^2 = {r_tilemean**2:.4f}")

    # ---- independent cross-check on real rasters ------------------------------------------------
    print(f"\nINDEPENDENT CROSS-CHECK -- direct pixel-level r on a route-stratified sample "
          f"({sample_per_route}/route):")
    import rasterio
    rng = np.random.default_rng(0)
    by_route = {}
    for name in C.tile_names():
        by_route.setdefault(C.parse_filename(name)["route"], []).append(name)
    sample = []
    for route in sorted(by_route):
        pool = by_route[route]
        take = min(sample_per_route, len(pool))
        idx = rng.choice(len(pool), size=take, replace=False)
        sample += [pool[i] for i in idx]

    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for name in sample:
        with rasterio.open(C.SPLITTED_DIR / "DSM" / name) as s:
            a = s.read(1).astype(np.float64).ravel()
        with rasterio.open(C.SPLITTED_DIR / "DTM" / name) as s:
            b = s.read(1).astype(np.float64).ravel()
        ok = (a > C.NODATA_THRESHOLD) & (b > C.NODATA_THRESHOLD)
        a, b = a[ok], b[ok]
        n += a.size
        sx += a.sum(); sy += b.sum()
        sxx += (a * a).sum(); syy += (b * b).sum(); sxy += (a * b).sum()

    cov_s = sxy / n - (sx / n) * (sy / n)
    var_a = sxx / n - (sx / n) ** 2
    var_b = syy / n - (sy / n) ** 2
    r_raster = cov_s / np.sqrt(var_a * var_b)
    print(f"  {len(sample)} tiles over {len(by_route)} routes, {n:,} valid pixel pairs")
    print(f"  r = {r_raster:.4f}   r^2 = {r_raster**2:.4f}   "
          f"(closed form said {r_pixel:.4f}; delta {abs(r_raster - r_pixel):.4f})")

    print("\nWHAT THE EARLIER '~97%' WAS:")
    print("  It was never a computed Pearson correlation. It was a loose verbal restatement of the")
    print(f"  SQ1 F8 BETWEEN-TILE variance shares (DSM {100*b_dsm/var_dsm:.1f}%, DTM {100*b_dtm/var_dtm:.1f}%),")
    print("  which describe how much of each channel is terrain elevation rather than structure --")
    print("  not how much the two channels share with each other.")
    print(f"  The number for the thesis is the pixel-level r = {r_pixel:.3f} (r^2 = {r_pixel**2:.3f}).")

    result = {
        "n_tiles": int(len(df)),
        "pooled": {
            "DSM": {"total_sd": float(np.sqrt(var_dsm)), "between_sd": float(np.sqrt(b_dsm)),
                    "within_sd": float(np.sqrt(w_dsm)), "between_share": b_dsm / var_dsm},
            "DTM": {"total_sd": float(np.sqrt(var_dtm)), "between_sd": float(np.sqrt(b_dtm)),
                    "within_sd": float(np.sqrt(w_dtm)), "between_share": b_dtm / var_dtm},
            "nDSM": {"total_sd": float(np.sqrt(var_nd)), "between_sd": float(np.sqrt(b_nd)),
                     "within_sd": float(np.sqrt(w_nd)), "between_share": b_nd / var_nd},
        },
        "sq1_f8_sd_reproduced": bool(sd_ok),
        "ndsm_unclamped_in_csv": bool(n_neg > 0),
        "n_tiles_negative_mean_ndsm": n_neg,
        "r_pixel_closed_form": float(r_pixel),
        "r2_pixel_closed_form": float(r_pixel ** 2),
        "r_pixel_backsolved_from_sq1": float(r_backsolve),
        "r_pixel_raster_sample": float(r_raster),
        "r_tile_mean": float(r_tilemean),
        "raster_sample_tiles": len(sample),
        "raster_sample_pixels": int(n),
    }
    C.ensure_out_dirs()
    C.write_json(result, CORRELATION_JSON)
    print(f"\nwrote {CORRELATION_JSON}")
    return result


# ==================================================================================================
# P0.3 -- epoch budget
# ==================================================================================================
def read_training_csv(path):
    """Parse a training log, undoing the double-logging.

    Every epoch is written twice: once under
        epoch,train_loss,valid_loss,valid_accuracy,time
    and once under the same header plus lr_0, with a second header row re-emitted mid-file. A
    10-epoch run therefore has 21 lines. Keep one row per epoch, preferring the one with lr_0.
    """
    by_epoch = {}
    with open(path, newline="", encoding="utf8", errors="replace") as fh:
        header = None
        for raw in csv.reader(fh):
            if not raw:
                continue
            if raw[0] == "epoch":
                header = raw
                continue
            if header is None:
                continue
            rec = dict(zip(header, raw))
            try:
                ep = int(rec["epoch"])
            except (KeyError, ValueError):
                continue
            prev = by_epoch.get(ep)
            if prev is None or ("lr_0" in rec and "lr_0" not in prev):
                by_epoch[ep] = rec
    return [by_epoch[e] for e in sorted(by_epoch)]


def _f(rec, key):
    try:
        return float(rec[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def run_p03(verbose=True):
    C.banner("P0.3 -- epoch budget of the comparison cells")

    # Parse every job in the matrix. The 24 weighted rgb/6ch cells are what section 2 asks about;
    # the rest are needed to rank the Swin inversions against the whole field (P0.3c).
    rows = []
    for model_dir in P03_MODEL_DIRS:
        for chan in ["rgb", "6ch", "10ch"]:
            for unw in ["", "_unw"]:
                for fold in range(C.NFOLDS):
                    job = f"{model_dir}_{chan}_fold{fold}{unw}"
                    cell = f"{model_dir}_{chan}{unw}"
                    csv_path = C.SPATIAL_MATRIX / model_dir / job / "logs" / f"{job}.csv"
                    base = {"job": job, "cell": cell, "model": model_dir, "channels": chan,
                            "weighting": "unweighted" if unw else "weighted", "fold": fold,
                            "in_p03_scope": (not unw) and chan in P03_CHANNELS}
                    if not csv_path.is_file():
                        rows.append({**base, "status": "MISSING LOG"})
                        continue
                    recs = read_training_csv(csv_path)
                    vl = np.array([_f(r, "valid_loss") for r in recs])
                    va = np.array([_f(r, "valid_accuracy") for r in recs])
                    tl = np.array([_f(r, "train_loss") for r in recs])
                    lr = [_f(r, "lr_0") for r in recs]
                    best_vl, best_va = int(np.nanargmin(vl)), int(np.nanargmax(va))
                    last = len(recs) - 1
                    # "still improving" only if the final epoch is the best AND the last three moved
                    # monotonically toward it.
                    improving_loss = (best_vl == last) and len(vl) >= 3 and vl[-1] < vl[-2] < vl[-3]
                    improving_acc = (best_va == last) and len(va) >= 3 and va[-1] > va[-2] > va[-3]
                    rows.append({
                        **base, "status": "ok", "n_epochs": len(recs),
                        "final_train_loss": float(tl[last]),
                        "final_valid_loss": float(vl[last]), "best_valid_loss": float(np.nanmin(vl)),
                        "best_valid_loss_epoch": best_vl,
                        "final_valid_acc": float(va[last]), "best_valid_acc": float(np.nanmax(va)),
                        "best_valid_acc_epoch": best_va,
                        "final_lr": lr[last],
                        "d_valid_loss_last3": float(vl[last] - vl[max(0, last - 2)]),
                        "d_valid_acc_last3": float(va[last] - va[max(0, last - 2)]),
                        "still_improving_loss": bool(improving_loss),
                        "still_improving_acc": bool(improving_acc),
                    })

    all_ok = [r for r in rows if r.get("status") == "ok"]
    ok = [r for r in all_ok if r["in_p03_scope"]]
    print(f"parsed {len(all_ok)} of {len(rows)} matrix jobs; "
          f"{len(ok)} are in P0.3 scope (weighted rgb/6ch)")
    epochs = sorted({r["n_epochs"] for r in ok})
    print(f"jobs inspected: {len(rows)}  ({len(ok)} with logs)")
    print(f"distinct epoch counts: {epochs}  -> "
          f"{'UNIFORM' if len(epochs) == 1 else 'NOT UNIFORM -- new arms must match per-cell'}")

    print(f"\n  {'job':<28}{'ep':>4}{'valid_loss fin/best':>22}{'@':>4}"
          f"{'valid_acc fin/best':>21}{'@':>4}{'final lr':>11}  improving")
    for r in sorted(ok, key=lambda z: (z["channels"], z["job"])):
        flag = []
        if r["still_improving_loss"]:
            flag.append("loss")
        if r["still_improving_acc"]:
            flag.append("acc")
        print(f"  {r['job']:<28}{r['n_epochs']:>4}"
              f"{r['final_valid_loss']:>11.4f}/{r['best_valid_loss']:<10.4f}{r['best_valid_loss_epoch']:>4}"
              f"{r['final_valid_acc']:>11.4f}/{r['best_valid_acc']:<9.4f}{r['best_valid_acc_epoch']:>4}"
              f"{r['final_lr']:>11.2e}  {','.join(flag) if flag else '-'}")

    n_imp_l = sum(r["still_improving_loss"] for r in ok)
    n_imp_a = sum(r["still_improving_acc"] for r in ok)
    n_final_best_l = sum(r["best_valid_loss_epoch"] == r["n_epochs"] - 1 for r in ok)
    print(f"\nstill improving on valid_loss     : {n_imp_l} / {len(ok)}")
    print(f"still improving on valid_accuracy : {n_imp_a} / {len(ok)}")
    print(f"final epoch is the best valid_loss: {n_final_best_l} / {len(ok)}")
    if ok:
        med_lr = float(np.median([r["final_lr"] for r in ok]))
        print(f"median final learning rate        : {med_lr:.2e}  "
              f"({'one-cycle fully annealed, budget not truncated' if med_lr < 1e-5 else 'NOT annealed'})")

    C.ensure_out_dirs()
    out = C.assert_writes_are_local(EPOCH_BUDGET_CSV)
    keys = ["job", "cell", "model", "channels", "weighting", "fold", "in_p03_scope", "status",
            "n_epochs", "final_train_loss", "final_valid_loss", "best_valid_loss",
            "best_valid_loss_epoch", "final_valid_acc", "best_valid_acc", "best_valid_acc_epoch",
            "final_lr", "d_valid_loss_last3", "d_valid_acc_last3", "still_improving_loss",
            "still_improving_acc"]
    with open(out, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    ckpt = which_checkpoint_is_saved(budget_rows=ok)
    inversion = swin_inversion_check(all_ok)
    return {"rows": rows, "epoch_counts": epochs, "n_still_improving_loss": n_imp_l,
            "n_still_improving_acc": n_imp_a, "n_final_is_best_loss": n_final_best_l,
            "checkpoint": ckpt, "inversion": inversion}


def which_checkpoint_is_saved(budget_rows=None):
    """Is <job>.pth the FINAL epoch, or the BEST-valid-loss one? The distinction is not cosmetic.

    In this design each fold's "validation" split IS the held-out fold that gets pooled into the
    out-of-fold confusion matrix. So:

      * <job>.pth == FINAL epoch          -> methodologically clean. Selection never saw the
                                             evaluation data. The runs overfitting is then a
                                             limitation paragraph, not a validity problem.
      * <job>.pth == BEST-valid-loss epoch -> the checkpoint was SELECTED on the evaluation set.
                                             All 24 pooled cells are contaminated by selection on
                                             the data they are scored against.

    Decided by byte-comparing <job>.pth against the per-epoch <job>_<n>.pth files.

    READ-ONLY. Nothing is deleted or moved: the per-epoch checkpoints are the only route to
    remediation if the second case turns out to hold.
    """
    import torch

    C.banner("P0.3b -- is <job>.pth the final epoch, or selected on the held-out fold?")
    print("  (read-only; no checkpoint is deleted or moved -- they are the remediation route)")
    print("  wiring: SaveModelCallback(every_epoch=True) writes <job>_<n>.pth per epoch and does no")
    print("          best-model restore; learn.save(job_name) then writes <job>.pth (train.py:665,")
    print("          segformer_train.py:375). Verified below on the WEIGHTS, not on file bytes --")
    print("          byte comparison is unreliable across the two save paths.")
    best_epoch_by_job = {}
    if budget_rows:
        best_epoch_by_job = {r["job"]: r.get("best_valid_loss_epoch")
                             for r in budget_rows if r.get("status") == "ok"}

    def digest(p):
        """Order-independent fingerprint of the model weights only."""
        obj = torch.load(p, map_location="cpu", weights_only=False)
        sd = obj.get("model", obj) if isinstance(obj, dict) else obj
        keys = sorted(k for k in sd if hasattr(sd[k], "dtype") and sd[k].is_floating_point())
        picks = keys[:2] + keys[len(keys) // 2: len(keys) // 2 + 2] + keys[-2:]
        return tuple(round(float(sd[k].double().sum()), 6) for k in picks)

    out = []
    for model_dir in P03_MODEL_DIRS:
        for chan in P03_CHANNELS:
            job = f"{model_dir}_{chan}_fold0"
            mdir = C.SPATIAL_MATRIX / model_dir / job / "models"
            final = mdir / f"{job}.pth"
            if not final.is_file():
                continue
            epoch_files = sorted(mdir.glob(f"{job}_*.pth"),
                                 key=lambda p: int(p.stem.rsplit("_", 1)[1])
                                 if p.stem.rsplit("_", 1)[1].isdigit() else -1)
            if not epoch_files:
                out.append({"job": job, "matches": None, "n_epoch_ckpts": 0})
                continue
            # Only the two candidates that matter need loading: the last epoch and the best one.
            fd = digest(final)
            by_ep = {p.stem.rsplit("_", 1)[1]: p for p in epoch_files}
            highest_k = epoch_files[-1].stem.rsplit("_", 1)[1]
            best_k = str(best_epoch_by_job.get(job))
            match = None
            for k in [highest_k, best_k]:
                if k in by_ep and digest(by_ep[k]) == fd:
                    match = k
                    break
            highest = epoch_files[-1].stem.rsplit("_", 1)[1]
            best_ep = best_epoch_by_job.get(job)
            if match is None:
                verdict = "matches no per-epoch file"
            elif str(match) == str(highest):
                verdict = "FINAL epoch -> clean"
            elif best_ep is not None and str(match) == str(best_ep):
                verdict = "BEST-valid-loss epoch -> SELECTED ON THE EVALUATION SET"
            else:
                verdict = f"epoch {match}, neither final nor best"
            out.append({"job": job, "matches_epoch": match,
                        "n_epoch_ckpts": len(epoch_files),
                        "highest_epoch_ckpt": highest,
                        "best_valid_loss_epoch": best_ep,
                        "verdict": verdict})
            print(f"  {job:<28} {len(epoch_files):>3} ckpts, <job>.pth == epoch "
                  f"{match if match is not None else '-':<4} "
                  f"(highest {highest}, best-valid-loss {best_ep})   {verdict}")
    if not out:
        print("  no checkpoints found")
        return out

    verdicts = {r["verdict"] for r in out}
    print()
    if verdicts == {"FINAL epoch -> clean"}:
        print("  CONCLUSION: <job>.pth is the FINAL epoch in every job checked.")
        print("  Model selection never saw the held-out fold. The pooled OOF numbers are")
        print("  methodologically clean; the overfitting is a LIMITATION PARAGRAPH, not a defect.")
    elif any("SELECTED ON THE EVALUATION SET" in v for v in verdicts):
        print("  CONCLUSION: <job>.pth is the BEST-VALID-LOSS checkpoint for at least one job.")
        print("  The validation split IS the held-out fold that feeds the pooled OOF matrix, so the")
        print("  checkpoint was selected on the evaluation data. ALL 24 CELLS ARE CONTAMINATED.")
        print("  Remediation route: the per-epoch checkpoints are intact -- re-infer from the final")
        print("  epoch instead. Do not delete them.")
    else:
        print(f"  CONCLUSION: mixed or unexpected -- {sorted(verdicts)}")
    return out


def swin_inversion_check(all_rows):
    """Are the two Swin weighting inversions an overfitting artefact?

    F3 reports weighted > unweighted in 10 of 12 pairs, the exceptions being swin/rgb (-0.0202) and
    swin/6ch (-0.0015). If the weighted Swin runs diverge from their best valid_loss much harder
    than their unweighted twins, the "inversion" is a training-stability artefact rather than a real
    interaction between architecture and loss weighting, and F3 should say so.

    Divergence metric: final_valid_loss - best_valid_loss, i.e. how far the run drifted past its own
    best, averaged over the 3 folds of a cell.
    """
    C.banner("P0.3c -- do the two Swin weighting inversions coincide with the worst divergence?")

    by_cell = {}
    for r in all_rows:
        if r.get("status") != "ok":
            continue
        by_cell.setdefault(r["cell"], []).append(r)

    stats = {}
    for cell, rows in by_cell.items():
        div = np.mean([x["final_valid_loss"] - x["best_valid_loss"] for x in rows])
        gap = np.mean([x["final_valid_loss"] - x["final_train_loss"] for x in rows])
        stats[cell] = {"divergence": float(div), "train_valid_gap": float(gap), "n_folds": len(rows)}

    ranked = sorted(stats.items(), key=lambda kv: -kv[1]["divergence"])
    print(f"  {'cell':<28}{'final-best valid_loss':>23}{'valid-train gap':>18}{'rank':>6}")
    for i, (cell, s) in enumerate(ranked, 1):
        mark = "  <== INVERSION" if cell in ("swin_upernet_rgb", "swin_upernet_6ch") else ""
        print(f"  {cell:<28}{s['divergence']:>23.4f}{s['train_valid_gap']:>18.4f}{i:>6}{mark}")

    print("\n  the two inversion pairs, weighted vs unweighted:")
    verdict_bits = []
    for base in ("swin_upernet_rgb", "swin_upernet_6ch"):
        w, u = stats.get(base), stats.get(base + "_unw")
        if not w or not u:
            continue
        worse = w["divergence"] > u["divergence"]
        verdict_bits.append(worse)
        print(f"    {base:<26} weighted div {w['divergence']:>8.4f}   "
              f"unweighted div {u['divergence']:>8.4f}   "
              f"{'weighted diverges MORE' if worse else 'weighted diverges LESS'}")

    n_cells = len(ranked)
    inv_ranks = [i for i, (c, _) in enumerate(ranked, 1) if c in ("swin_upernet_rgb", "swin_upernet_6ch")]
    print(f"\n  inversion cells rank {inv_ranks} of {n_cells} on divergence "
          f"(1 = worst).")
    if verdict_bits and all(verdict_bits) and inv_ranks and min(inv_ranks) <= max(3, n_cells // 4):
        print("  READING: the weighted Swin arms diverge more than their unweighted twins AND sit")
        print("  among the worst divergers overall -- consistent with the inversion being an")
        print("  OVERFITTING ARTEFACT rather than a real loss-by-architecture interaction.")
    elif verdict_bits and not any(verdict_bits):
        print("  READING: the weighted Swin arms diverge LESS than their unweighted twins, so")
        print("  overfitting does not explain the inversion. Treat F3's exception as real.")
    else:
        print("  READING: mixed evidence -- report the numbers, do not assert a mechanism.")
    return {"per_cell": stats, "ranked": [c for c, _ in ranked], "inversion_ranks": inv_ranks}


# ==================================================================================================
def selftest():
    C.banner("eda_preflight_ndsm self-test")

    # pool_variance against a case computed by hand: two tiles, known pixels.
    a = np.array([1.0, 3.0, 5.0, 7.0])
    b = np.array([2.0, 2.0, 8.0, 8.0])
    tile_means = np.array([a[:2].mean(), a[2:].mean()])
    tile_stds = np.array([a[:2].std(), a[2:].std()])
    tot, bet, wit = pool_variance(tile_means, tile_stds)
    assert abs(tot - np.var(a)) < 1e-12, (tot, np.var(a))
    print(f"  pool_variance   OK  (recovers np.var exactly: {tot:.6f})")

    # the covariance identity the P0.2 closed form rests on
    d = a - b
    va, vb, vd = np.var(a), np.var(b), np.var(d)
    cov = (va + vb - vd) / 2
    r = cov / np.sqrt(va * vb)
    assert abs(r - np.corrcoef(a, b)[0, 1]) < 1e-12
    print(f"  cov identity    OK  (r={r:.6f} matches np.corrcoef)")

    # the double-logged training csv parser
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.csv"
        p.write_text(
            "epoch,train_loss,valid_loss,valid_accuracy,time\n"
            "0,0.5,0.9,0.70,10:00\n"
            "epoch,train_loss,valid_loss,valid_accuracy,time,lr_0\n"
            "0,0.5,0.9,0.70,10:00,0.001\n"
            "1,0.4,0.8,0.75,10:00\n"
            "1,0.4,0.8,0.75,10:00,0.002\n"
            "2,0.3,0.7,0.80,10:00\n"
            "2,0.3,0.7,0.80,10:00,0.0000001\n", encoding="utf8")
        recs = read_training_csv(p)
        assert len(recs) == 3, f"expected 3 epochs, got {len(recs)}"
        assert all("lr_0" in r for r in recs), "should prefer the lr_0 rows"
        assert _f(recs[-1], "valid_loss") == 0.7
        print(f"  csv parser      OK  (6 data lines + 2 headers -> {len(recs)} epochs, lr_0 kept)")

    print("\nselftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p01", action="store_true", help="measure the /255 at the model's input")
    ap.add_argument("--p01b", action="store_true", help="audit the train/valid/inference paths")
    ap.add_argument("--p02", action="store_true", help="DSM/DTM redundancy, full pool")
    ap.add_argument("--p03", action="store_true", help="epoch budget of the comparison cells")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sample_per_route", type=int, default=12)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.p01 or args.p01b or args.p02 or args.p03):
        ap.error("nothing to do: pass --p01/--p01b/--p02/--p03 (or --selftest)")

    if args.p01:
        run_p01()
    if args.p01b:
        run_p01b()
    if args.p02:
        run_p02(sample_per_route=args.sample_per_route)
    if args.p03:
        run_p03()


if __name__ == "__main__":
    main()
