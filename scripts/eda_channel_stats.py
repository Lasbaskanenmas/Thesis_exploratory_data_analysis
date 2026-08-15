#!/usr/bin/env python
"""
E2 -- Channel audit: what the inputs actually carry, and what the pipeline does to them.

WHY THIS EXISTS
    Finding F4 of the 2026-07-28 handoff reports RGB >= 10ch >= 6ch, i.e. the auxiliary channels
    do not help. It is marked UNVERIFIED because it gates a PROCUREMENT recommendation to KDS, and
    a normalisation or stacking defect would produce exactly that pattern. The labels have been
    audited twice; the six image channels have never been audited at all.

    Two defects are already visible in the source and are the reason this module is a full-pool
    scan rather than a spot check:

      sdfi_dataset.py:146     nir_mean = 0.40779018  #based on a sample size of 1! OBS! UNKNOWN
      matrix configs          DSM and DTM are normalised with mean 0.5 / std 1.0, which are
                              placeholders, while the tiles hold RAW METRES ABOVE SEA LEVEL

    If the second holds, the elevation channels reach the network with almost no variance, and
    what little survives encodes absolute terrain height -- which is largely a proxy for WHERE IN
    DENMARK the tile is, i.e. a spatial confound -- while the physically meaningful quantity
    nDSM = DSM - DTM (object height above ground) is never computed at all. That would make F4 a
    statement about the configuration rather than about the imagery, which is a completely
    different recommendation.

WHAT IT MEASURES
    per band (all 14 available bands, not just the 10 the model uses)
        count, mean, std, min, max, percentiles, full histogram, nodata incidence
    normalisation audit
        measured statistics against the config vectors, and the effective post-normalisation mean
        and std each channel actually presents to the network
    nDSM = DSM - DTM
        distribution overall and per class -- does the discarded signal separate the classes the
        models fail on
    redundancy
        per-band per-class means, and rgb against OrtoRGB agreement and registration offset, which
        also settles which product is nadir and which is oblique
    sampling
        a per-class pixel sample written to cache/ for module E1's separability probe

    Bands are scanned for every folder, including cir and OrtoCIR bands 1 and 2 which the pipeline
    discards, so the assumption that band 0 is the NIR band is itself testable.

READ-ONLY over the source data. Writes only under exploratory_data_analysis/.

    python eda_channel_stats.py [--procs N] [--limit N] [--sample-target N]
                                [--registration-stride N] [--selftest]
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

# ----------------------------------------------------------------------------------------------
# Histogram specification per band. Reflectance bands are uint8; elevation is metres.
# ----------------------------------------------------------------------------------------------
HIST_SPEC = {"reflectance": (0.0, 256.0, 256),
             "elevation": (-50.0, 300.0, 350),
             "ndsm": (-20.0, 80.0, 200)}


def spec_for(folder: str):
    return HIST_SPEC["elevation"] if folder in C.FLOAT_FOLDERS else HIST_SPEC["reflectance"]


NBAND = len(C.ALL_BANDS)
SAMPLE_BANDS = C.USED_BANDS_10CH                 # the 10 the model consumes, in config order

# Filled per process by _init so the worker does not re-read the audit for every tile.
_CAPS: dict[int, int] = {}


def _init(caps):
    global _CAPS
    _CAPS = caps


def worker(fname: str):
    """Scan every band of one tile. Returns accumulators, never writes anything."""
    import rasterio
    try:
        label = C.read_label(C.LABEL_DIR / fname)

        band_data = {}
        for folder in C.CHANNEL_FOLDERS:
            with rasterio.open(C.SPLITTED_DIR / folder / fname) as src:
                arr = src.read().astype(np.float32)
            for b in range(C.FOLDER_NBANDS[folder]):
                band_data[(folder, b)] = arr[b]

        n = NBAND
        count = np.zeros(n, dtype=np.int64)
        bsum = np.zeros(n, dtype=np.float64)
        bsumsq = np.zeros(n, dtype=np.float64)
        bmin = np.full(n, np.inf)
        bmax = np.full(n, -np.inf)
        hists = []
        nodata = np.zeros(n, dtype=np.int64)

        # per (class, band) first and second moments, for spectral signatures
        cls_count = np.zeros((C.NCLASS, n), dtype=np.int64)
        cls_sum = np.zeros((C.NCLASS, n), dtype=np.float64)
        cls_sumsq = np.zeros((C.NCLASS, n), dtype=np.float64)
        flat_label = label.ravel()

        for k, (folder, b) in enumerate(C.ALL_BANDS):
            v = band_data[(folder, b)].ravel()
            if folder in C.FLOAT_FOLDERS:
                valid = v > C.NODATA_THRESHOLD
                nodata[k] = int((~valid).sum())
                vv = v[valid]
                lv = flat_label[valid]
            else:
                vv = v
                lv = flat_label
            count[k] = vv.size
            if vv.size:
                bsum[k] = float(vv.sum(dtype=np.float64))
                bsumsq[k] = float((vv.astype(np.float64) ** 2).sum())
                bmin[k] = float(vv.min())
                bmax[k] = float(vv.max())
                cls_count[:, k] = np.bincount(lv, minlength=C.NCLASS)
                cls_sum[:, k] = np.bincount(lv, weights=vv.astype(np.float64),
                                            minlength=C.NCLASS)
                cls_sumsq[:, k] = np.bincount(lv, weights=vv.astype(np.float64) ** 2,
                                              minlength=C.NCLASS)
            lo, hi, nb = spec_for(folder)
            hists.append(np.histogram(vv, bins=nb, range=(lo, hi))[0].astype(np.int64)
                         if vv.size else np.zeros(nb, dtype=np.int64))

        # ---- nDSM: object height above ground, the quantity the pipeline never forms ----------
        dsm, dtm = band_data[("DSM", 0)], band_data[("DTM", 0)]
        both = (dsm > C.NODATA_THRESHOLD) & (dtm > C.NODATA_THRESHOLD)
        nd = (dsm - dtm)[both]
        ndl = label[both]
        lo, hi, nb = HIST_SPEC["ndsm"]
        nd_hist = np.histogram(nd, bins=nb, range=(lo, hi))[0].astype(np.int64) if nd.size \
            else np.zeros(nb, dtype=np.int64)
        nd_cls_count = np.bincount(ndl, minlength=C.NCLASS) if nd.size else np.zeros(C.NCLASS, np.int64)
        nd_cls_sum = (np.bincount(ndl, weights=nd.astype(np.float64), minlength=C.NCLASS)
                      if nd.size else np.zeros(C.NCLASS))
        nd_cls_sumsq = (np.bincount(ndl, weights=nd.astype(np.float64) ** 2, minlength=C.NCLASS)
                        if nd.size else np.zeros(C.NCLASS))

        # ---- rgb vs OrtoRGB agreement (are these two acquisitions or two processings?) --------
        a = np.stack([band_data[("rgb", i)] for i in range(3)]).mean(0).ravel()
        b_ = np.stack([band_data[("OrtoRGB", i)] for i in range(3)]).mean(0).ravel()
        if a.std() > 1e-6 and b_.std() > 1e-6:
            corr = float(np.corrcoef(a, b_)[0, 1])
        else:
            corr = np.nan
        agree = (corr, float(a.mean()), float(b_.mean()), float(a.std()), float(b_.std()))

        # ---- per-class pixel sample for module E1 ---------------------------------------------
        rng = np.random.default_rng(abs(hash(fname)) % (2 ** 32))
        stack = np.stack([band_data[fb].ravel() for fb in SAMPLE_BANDS], axis=1)
        sample = {}
        for cls in C.PREDICTED:
            cap = _CAPS.get(cls, 0)
            if cap <= 0:
                continue
            where = np.flatnonzero(flat_label == cls)
            if where.size == 0:
                continue
            take = where if where.size <= cap else rng.choice(where, cap, replace=False)
            sample[cls] = stack[take].astype(np.float32)

        # Per-tile elevation moments. These drive the within-tile versus between-tile variance
        # decomposition: pool-wide DSM spread is dominated by Denmark's terrain, which is a proxy
        # for WHERE the tile is, whereas object height lives in the within-tile component.
        dv = dsm[dsm > C.NODATA_THRESHOLD]
        tv = dtm[dtm > C.NODATA_THRESHOLD]
        nan = float("nan")
        elev = (float(dv.mean()) if dv.size else nan, float(dv.std()) if dv.size else nan,
                float(tv.mean()) if tv.size else nan, float(tv.std()) if tv.size else nan,
                float(nd.mean()) if nd.size else nan, float(nd.std()) if nd.size else nan)

        return ("OK", fname, count, bsum, bsumsq, bmin, bmax, np.concatenate(hists), nodata,
                cls_count, cls_sum, cls_sumsq,
                nd_hist, nd_cls_count, nd_cls_sum, nd_cls_sumsq, agree, sample, elev)
    except Exception as exc:                       # noqa: BLE001 -- report, never repair
        return ("FAIL", fname, f"{type(exc).__name__}: {exc}")


def hist_offsets():
    offs, total = [], 0
    for folder, _ in C.ALL_BANDS:
        nb = spec_for(folder)[2]
        offs.append((total, total + nb))
        total += nb
    return offs, total


def main() -> None:
    ap = argparse.ArgumentParser(description="E2 -- full-pool channel audit")
    ap.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 8))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-target", type=int, default=100_000,
                    help="target sampled pixels per class for E1's probe")
    ap.add_argument("--registration-stride", type=int, default=20,
                    help="run the rgb/OrtoRGB shift estimate on every Nth tile")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    C.ensure_out_dirs()
    names = C.tile_names()
    if args.limit:
        names = names[: args.limit]

    # Per-tile sampling cap chosen so each class contributes roughly the same total, drawn evenly
    # across the tiles it occurs in rather than concentrated in whichever tiles finish first.
    audit = C.load_class_pixel_audit()
    caps = {}
    for cls in C.PREDICTED:
        tiles_present = max(int(audit[C.CODES[cls]]["tiles_present"]), 1)
        caps[cls] = int(np.ceil(args.sample_target / tiles_present))

    C.banner(f"E2 -- channel audit over {len(names):,} tiles x {len(C.ALL_BANDS)} bands")
    print(f"channels : {C.SPLITTED_DIR}  (read-only)")
    print(f"processes: {args.procs}")
    print(f"volume   : roughly {len(names) * 20 / 1000:.0f} GB of reads; this runs for hours\n")
    print("per-tile sampling cap per class (E1 probe):")
    for cls in C.PREDICTED:
        print(f"  {C.CODES[cls]:<12} present in {audit[C.CODES[cls]]['tiles_present']:>6,} tiles"
              f" -> {caps[cls]:>6,} px/tile")

    offs, hist_total = hist_offsets()
    n = NBAND
    count = np.zeros(n, dtype=np.int64)
    bsum = np.zeros(n, dtype=np.float64)
    bsumsq = np.zeros(n, dtype=np.float64)
    bmin = np.full(n, np.inf)
    bmax = np.full(n, -np.inf)
    hist = np.zeros(hist_total, dtype=np.int64)
    nodata = np.zeros(n, dtype=np.int64)
    cls_count = np.zeros((C.NCLASS, n), dtype=np.int64)
    cls_sum = np.zeros((C.NCLASS, n), dtype=np.float64)
    cls_sumsq = np.zeros((C.NCLASS, n), dtype=np.float64)
    nd_hist = np.zeros(HIST_SPEC["ndsm"][2], dtype=np.int64)
    nd_cls_count = np.zeros(C.NCLASS, dtype=np.int64)
    nd_cls_sum = np.zeros(C.NCLASS, dtype=np.float64)
    nd_cls_sumsq = np.zeros(C.NCLASS, dtype=np.float64)
    samples = {cls: [] for cls in C.PREDICTED}
    sample_folds = {cls: [] for cls in C.PREDICTED}
    agree_rows, elev_rows, failures = [], [], []

    # Fold provenance for every sampled pixel. Module E1 trains its linear probe on some folds and
    # tests on the others, so without this the probe would train and test on pixels from the same
    # tiles and report an optimistically biased floor -- exactly the leak this thesis is about.
    inv = C.load_tile_inventory()
    tile_to_fold = dict(zip(inv["filename"], inv["fold"]))

    print()
    with Pool(processes=args.procs, initializer=_init, initargs=(caps,)) as pool:
        for i, res in enumerate(pool.imap_unordered(worker, names, chunksize=4), 1):
            if res[0] != "OK":
                failures.append((res[1], res[2]))
                continue
            (_, fname, c, s, sq, mn, mx, h, nd_, cc, cs, csq,
             ndh, ndcc, ndcs, ndcsq, agree, sample, elev) = res
            elev_rows.append((fname,) + elev)
            count += c; bsum += s; bsumsq += sq
            bmin = np.minimum(bmin, mn); bmax = np.maximum(bmax, mx)
            hist += h; nodata += nd_
            cls_count += cc; cls_sum += cs; cls_sumsq += csq
            nd_hist += ndh; nd_cls_count += ndcc; nd_cls_sum += ndcs; nd_cls_sumsq += ndcsq
            agree_rows.append((fname,) + agree)
            fold = tile_to_fold.get(fname, -1)
            for cls, arr in sample.items():
                samples[cls].append(arr)
                sample_folds[cls].append(np.full(len(arr), fold, dtype=np.int8))
            if i % 500 == 0:
                print(f"  {i:,} / {len(names):,}")

    if failures:
        print(f"\n{len(failures)} tiles failed to read (reported, not repaired):")
        for fname, err in failures[:20]:
            print(f"  {fname}: {err}")

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = bsum / count
        var = bsumsq / count - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))

    # ------------------------------------------------------------------------------------------
    C.banner("MEASURED BAND STATISTICS (full pool)")
    print(f"{'band':<20}{'pixels':>16}{'min':>10}{'max':>10}{'mean':>10}{'std':>10}{'nodata':>14}")
    for k, lab in enumerate(C.BAND_LABELS):
        print(f"{lab:<20}{count[k]:>16,}{bmin[k]:>10.2f}{bmax[k]:>10.2f}"
              f"{mean[k]:>10.3f}{std[k]:>10.3f}{nodata[k]:>14,}")

    # ------------------------------------------------------------------------------------------
    C.banner("NORMALISATION AUDIT  (this is what gates finding F4)")
    print("The configs normalise with (x/255 - mean) / std. Below, 'measured' is what the data")
    print("actually holds and 'config' is what the 10-channel configs assumed. 'effective std' is")
    print("the standard deviation each channel presents to the network after normalisation --")
    print("channels whose effective std is far below the others contribute correspondingly little.\n")
    idx = {fb: i for i, fb in enumerate(C.ALL_BANDS)}
    print(f"{'channel (config order)':<24}{'measured mean':>15}{'measured std':>14}"
          f"{'cfg mean':>10}{'cfg std':>9}{'eff mean':>10}{'eff std':>10}")
    eff_rows = []
    for j, fb in enumerate(C.USED_BANDS_10CH):
        k = idx[fb]
        cm, cs_ = C.CONFIG_MEANS_10CH[j], C.CONFIG_STDS_10CH[j]
        eff_mean = (mean[k] / C.INT_TO_FLOAT_DIV - cm) / cs_
        eff_std = (std[k] / C.INT_TO_FLOAT_DIV) / cs_
        lab = C.band_label(*fb)
        print(f"{lab:<24}{mean[k]:>15.4f}{std[k]:>14.4f}{cm:>10.4f}{cs_:>9.4f}"
              f"{eff_mean:>10.4f}{eff_std:>10.5f}")
        eff_rows.append({"channel": lab, "measured_mean": mean[k], "measured_std": std[k],
                         "config_mean": cm, "config_std": cs_,
                         "effective_mean_div255": eff_mean, "effective_std_div255": eff_std,
                         "effective_mean_no_div": (mean[k] - cm) / cs_,
                         "effective_std_no_div": std[k] / cs_})

    finite = [r["effective_std_div255"] for r in eff_rows if np.isfinite(r["effective_std_div255"])]
    if finite:
        hi, lo = max(finite), min(finite)
        print(f"\nspread of effective std across the 10 channels: {lo:.5f} to {hi:.5f} "
              f"({hi / lo:,.0f}x)" if lo > 0 else "")
        print("A large spread means the configuration, not the imagery, decides which channels the")
        print("network can use. Reported both ways below, since fastai's IntToFloatTensor divides")
        print("every channel by 255 including the float elevation bands:")
        print(f"{'channel':<24}{'eff std (with /255)':>22}{'eff std (no /255)':>20}")
        for r in eff_rows:
            print(f"{r['channel']:<24}{r['effective_std_div255']:>22.5f}"
                  f"{r['effective_std_no_div']:>20.5f}")
        print("\nThe ranking is what matters and it is identical either way, so the conclusion does")
        print("not depend on resolving where IntToFloatTensor is applied.")

    # ------------------------------------------------------------------------------------------
    C.banner("nDSM = DSM - DTM  (object height above ground: the signal never formed)")
    with np.errstate(invalid="ignore", divide="ignore"):
        nd_mean = nd_cls_sum / nd_cls_count
        nd_std = np.sqrt(np.maximum(nd_cls_sumsq / nd_cls_count - nd_mean ** 2, 0.0))
    print(f"{'class':<12}{'pixels':>16}{'mean nDSM (m)':>16}{'std':>10}")
    for i, name in enumerate(C.CODES):
        if nd_cls_count[i] == 0:
            continue
        print(f"{name:<12}{nd_cls_count[i]:>16,}{nd_mean[i]:>16.3f}{nd_std[i]:>10.3f}")
    print("\nIf the elevated classes (green_roof, drivhus, solceller) separate here while the")
    print("models score them near zero, then a real and physically meaningful signal exists in")
    print("data KDS already pays for, and the pipeline discards it.")

    # ------------------------------------------------------------------------------------------
    C.banner("ELEVATION VARIANCE: WITHIN TILE vs BETWEEN TILES")
    print("Law of total variance: total = between-tile (variance of per-tile means) + within-tile")
    print("(mean of per-tile variances). Object height lives in the WITHIN component. The BETWEEN")
    print("component is terrain elevation, which is largely a proxy for where in Denmark the tile")
    print("sits -- and route is the blocking unit, so that component is a spatial confound.\n")
    decomp = {}
    if elev_rows:
        arr = np.array([r[1:] for r in elev_rows], dtype=np.float64)
        for name, mcol, scol in (("DSM", 0, 1), ("DTM", 2, 3), ("nDSM", 4, 5)):
            m, s = arr[:, mcol], arr[:, scol]
            ok_ = np.isfinite(m) & np.isfinite(s)
            between = float(np.var(m[ok_]))
            within = float(np.mean(s[ok_] ** 2))
            total = between + within
            decomp[name] = {"between_var": between, "within_var": within, "total_var": total,
                            "between_sd": float(np.sqrt(between)),
                            "within_sd": float(np.sqrt(within)),
                            "between_share": between / total if total else None}
            print(f"{name:<6} total sd {np.sqrt(total):>7.3f} m  =  between-tile sd "
                  f"{np.sqrt(between):>7.3f} m  +  within-tile sd {np.sqrt(within):>6.3f} m"
                  f"   (between = {100 * between / total:5.1f}% of variance)")
        b = decomp["DSM"]
        print(f"\nSo of everything the DSM channel varies by, {100 * b['between_share']:.1f}% is")
        print("between-tile terrain rather than structure inside the tile. After the config's")
        print("std of 1.0 the within-tile signal is compressed to roughly")
        print(f"{b['within_sd'] / C.INT_TO_FLOAT_DIV:.5f} in normalised units, against about 0.8")
        print("for the reflectance bands. nDSM removes the terrain component by construction,")
        print("which is exactly why it is the quantity worth feeding the network.")
        ecsv = C.assert_writes_are_local(C.TABLES / "tile_elevation_stats.csv")
        with open(ecsv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "dsm_mean", "dsm_std", "dtm_mean", "dtm_std",
                        "ndsm_mean", "ndsm_std"])
            w.writerows(elev_rows)
        print(f"\nwrote {ecsv}")

    # ------------------------------------------------------------------------------------------
    C.banner("rgb vs OrtoRGB  (redundancy, and which product is which)")
    corrs = np.array([r[1] for r in agree_rows], dtype=np.float64)
    ok = np.isfinite(corrs)
    if ok.any():
        print(f"per-tile Pearson r between rgb and OrtoRGB intensity, over {ok.sum():,} tiles")
        print(f"  mean {corrs[ok].mean():.4f}   median {np.median(corrs[ok]):.4f}"
              f"   p05 {np.percentile(corrs[ok], 5):.4f}   p95 {np.percentile(corrs[ok], 95):.4f}")
        a_mean = np.mean([r[2] for r in agree_rows]); b_mean = np.mean([r[3] for r in agree_rows])
        a_std = np.mean([r[4] for r in agree_rows]); b_std = np.mean([r[5] for r in agree_rows])
        print(f"  rgb     mean intensity {a_mean:7.2f}   mean within-tile sd {a_std:6.2f}")
        print(f"  OrtoRGB mean intensity {b_mean:7.2f}   mean within-tile sd {b_std:6.2f}")
        print("\nHigh correlation with different dynamic range indicates two renderings of the same")
        print("ground; moderate correlation indicates two independent acquisitions. Plan 2.2 and")
        print("the preprocessing README disagree on which folder is nadir and which is oblique, so")
        print("this measurement plus confirmation from Rasmus settles it.")

    # ------------------------------------------------------------------------------------------
    C.banner("WRITING ARTIFACTS")
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_tiles_scanned": len(names) - len(failures),
        "n_failures": len(failures),
        "failures": failures[:50],
        "bands": [{"band": C.BAND_LABELS[k], "folder": C.ALL_BANDS[k][0],
                   "band_index": C.ALL_BANDS[k][1], "pixels": int(count[k]),
                   "min": float(bmin[k]), "max": float(bmax[k]),
                   "mean": float(mean[k]), "std": float(std[k]),
                   "nodata_pixels": int(nodata[k])} for k in range(n)],
        "normalisation_audit": eff_rows,
        "int_to_float_div": C.INT_TO_FLOAT_DIV,
        "ndsm_per_class": [{"class": C.CODES[i], "pixels": int(nd_cls_count[i]),
                            "mean_m": None if nd_cls_count[i] == 0 else float(nd_mean[i]),
                            "std_m": None if nd_cls_count[i] == 0 else float(nd_std[i])}
                           for i in range(C.NCLASS)],
        "elevation_variance_decomposition": decomp,
        "rgb_vs_ortorgb": {
            "n_tiles": int(ok.sum()),
            "corr_mean": float(corrs[ok].mean()) if ok.any() else None,
            "corr_median": float(np.median(corrs[ok])) if ok.any() else None,
        },
    }
    print(f"wrote {C.write_json(payload, C.TABLES / 'channel_stats.json')}")

    per_class_csv = C.assert_writes_are_local(C.TABLES / "channel_per_class_stats.csv")
    with np.errstate(invalid="ignore", divide="ignore"):
        cmean = cls_sum / cls_count
        cstd = np.sqrt(np.maximum(cls_sumsq / cls_count - cmean ** 2, 0.0))
    with open(per_class_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "band", "pixels", "mean", "std"])
        for i, name in enumerate(C.CODES):
            for k, lab in enumerate(C.BAND_LABELS):
                if cls_count[i, k]:
                    w.writerow([name, lab, int(cls_count[i, k]), cmean[i, k], cstd[i, k]])
    print(f"wrote {per_class_csv}")

    hist_path = C.assert_writes_are_local(C.TABLES / "channel_histograms.npz")
    np.savez_compressed(hist_path, hist=hist, nd_hist=nd_hist,
                        offsets=np.array(offs), labels=np.array(C.BAND_LABELS),
                        count=count, mean=mean, std=std, bmin=bmin, bmax=bmax, nodata=nodata)
    print(f"wrote {hist_path}")

    agree_csv = C.assert_writes_are_local(C.TABLES / "rgb_vs_ortorgb_per_tile.csv")
    with open(agree_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "pearson_r", "rgb_mean", "ortorgb_mean", "rgb_std", "ortorgb_std"])
        w.writerows(agree_rows)
    print(f"wrote {agree_csv}")

    # The pixel sample is large and regenerable -> cache/, which is gitignored.
    rng = np.random.default_rng(12345)
    out = {}
    for cls, chunks in samples.items():
        if not chunks:
            continue
        arr = np.concatenate(chunks, axis=0)
        fld = np.concatenate(sample_folds[cls], axis=0)
        if len(arr) > args.sample_target:
            sel = rng.choice(len(arr), args.sample_target, replace=False)
            arr, fld = arr[sel], fld[sel]
        out[C.CODES[cls]] = arr
        out[f"fold_{C.CODES[cls]}"] = fld
    sample_path = C.assert_writes_are_local(C.CACHE / "class_pixel_sample.npz")
    np.savez_compressed(sample_path, bands=np.array([C.band_label(*fb) for fb in SAMPLE_BANDS]),
                        **out)
    print(f"wrote {sample_path}  (gitignored; {sample_path.stat().st_size / 1e6:.1f} MB)")
    for cls in C.PREDICTED:
        name = C.CODES[cls]
        if name not in out:
            continue
        fld = out[f"fold_{name}"]
        spread = ", ".join(f"f{f}:{int((fld == f).sum()):,}" for f in range(C.NFOLDS))
        print(f"   {name:<12}{len(out[name]):>9,} px   ({spread})")


def selftest() -> None:
    """Check the normalisation arithmetic and the histogram layout without touching tiles."""
    C.banner("E2 selftest (synthetic, no files touched)")

    offs, total = hist_offsets()
    assert len(offs) == NBAND == 14, (len(offs), NBAND)
    assert total == 12 * 256 + 2 * 350, total
    assert offs[0] == (0, 256) and offs[-1][1] == total
    print(f"histogram layout : OK ({NBAND} bands, {total} bins total)")

    assert len(C.CONFIG_MEANS_10CH) == len(C.CONFIG_STDS_10CH) == len(C.USED_BANDS_10CH) == 10
    print("config vectors   : OK (10 channels, means and stds aligned)")

    # The real DSM tile measured during planning: mean 21.984 m, within-tile sd 0.728 m, against
    # config mean 0.5 / std 1.0. Contrast with an 8-bit reflectance band at mean 91.8, sd 40.4.
    dsm_eff_std = (0.728 / 255.0) / 1.0
    rgb_eff_std = (40.354 / 255.0) / 0.229
    ratio = rgb_eff_std / dsm_eff_std
    assert ratio > 100, ratio
    print(f"normalisation    : DSM effective sd {dsm_eff_std:.5f} vs rgb {rgb_eff_std:.5f} "
          f"-> {ratio:,.0f}x")
    print("  the arithmetic reproduces the defect this module exists to quantify at full scale")

    # Same conclusion without the /255 step, so the finding does not hinge on that detail.
    ratio_nodiv = (40.354 / 0.229) / (0.728 / 1.0)
    assert ratio_nodiv > 100, ratio_nodiv
    print(f"  without /255   : {ratio_nodiv:,.0f}x -- identical ranking, so the conclusion holds")
    print("                   either way")

    lab = [C.band_label(f, b) for f, b in C.ALL_BANDS]
    assert lab[3] == "cir_b0_NIR" and lab[4] == "cir_b1_unused", lab[:6]
    print("band labelling   : OK (unused cir/OrtoCIR bands are scanned and flagged)")

    print("\nSELFTEST PASSED")


if __name__ == "__main__":
    main()
