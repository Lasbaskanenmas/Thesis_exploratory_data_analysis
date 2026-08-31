#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
chapter2_batch_render.py -- one-run batch of every raster-derived figure input
Chapter 2 still needs. Written 2026-08-28 for the Befaestelsesdata thesis.

WHAT IT PRODUCES (under --out, default C:\\thesis\\logs_and_models\\chapter2_figures)

  2.1.2.elevation\
      DSM_<greenroof>.png, DTM_<greenroof>.png, nDSM_<greenroof>.png
        Green roof tile O2021_84_40_1_0039_00062718_2000_3000 per plan B2:
        DSM and DTM on a JOINT 2nd-98th percentile stretch (so heights are
        comparable between the two panels; --per-panel-stretch to override),
        nDSM = DSM - DTM on a FIXED 0-15 m ramp, colour bar on every panel.
        HARD GATE: DSM and DTM geotransforms must match rgb, else no render.

  2.3.misregistration\
      rgb_<solar>.png, DTM_<solar>.png, DSM_<solar>.png
        Solar tile O2021_82_19_1_0023_00005474_8000_1000. Expectation checked:
        DTM aligned with rgb, DSM displaced (the exhibit). Per-panel stretch
        here, because the DSM shows different ground with its own range.

  2.2.label-overlay\
      rgb_<t>.png, labels_<t>.png, overlay_<t>.png for every tile listed in
      OVERLAY_TILES, plus legend_classes.png. GATE per tile: label geotransform
      must match rgb, which excludes the 273 misplaced-label tiles. A blocked
      tile is reported and the run continues with the rest.
        The class-value mapping is NOT assumed: per tile, the label raster's
        pixel counts are matched against counts embedded from
        tile_inventory.csv. Exact match confirms the mapping; otherwise a
        permutation is inferred from the counts and flagged for verification;
        otherwise the legend falls back to raw values.

  bounds_entangled_routes.csv
      filename, route, minx, miny, maxx, maxy from the rgb rasters (image-side
      geometry, immune to the 273-tile label-geotransform defect) for routes
      82-20, 82-21, 85-45, 85-48. Feeds the 2.4 footprint map.

  render_report.json  -- every guard result, stretch range, and skip/render
      decision, machine-readable.

DISCIPLINE: read-only outside --out. Nothing under example_dataset is written,
moved or renamed. Existing outputs are found and skipped unless --force.

USAGE
  python chapter2_batch_render.py --check          # report only, render nothing
  python chapter2_batch_render.py                  # render whatever is missing
  python chapter2_batch_render.py --force          # re-render everything
  python chapter2_batch_render.py --root D:\\other\\example_dataset --out ...

CHANGES SINCE THE FIRST RUN, 2026-08-29
  1. BOTH DEFAULT PATHS CORRECTED. DEFAULT_ROOT pointed at C:\\thesis\\
     example_dataset, but the tree is under multi_channel_dataset_creation.
     DEFAULT_OUT pointed into logs_and_models, which contradicts this script's
     own read-only discipline and the Plan 3.1 section 8 guard rail keeping
     logs_and_models agent-read-only. Both now match where the first run was
     actually pointed by hand.
  2. ASFALT IS NOW MAGENTA rather than dark grey. A dark grey tint over grey
     asphalt is very close to invisible, so in the first run's overlays the
     roads read as unlabelled when they were not. Magenta was chosen over red
     because red sits too close to fliser once tinted, and asphalt beside
     paving tiles is the commonest adjacency in this material.
     >>> THIS CHANGES EVERY OVERLAY, so the run needs --force or the old
     >>> and new overlays will disagree about what colour asfalt is.
  3. OVERLAY_TILES REPLACES the two hardcoded candidates, and now carries four
     scenes including the ground-mounted solar park.

RECOMMENDED COMMAND FOR THE SECOND RUN
  python chapter2_batch_render.py --force --skip-bounds

  --force because of the colour change above. --skip-bounds because
  bounds_entangled_routes.csv is already correct, was verified against
  tile_inventory.csv, and rebuilding it means re-reading about 1,470 raster
  headers for no gain.

ENVIRONMENT NOTE FROM THE FIRST RUN
  multi-channel-env hard-aborts inside fig.savefig (0xC0000409), matplotlib
  3.9.4 against numpy 2.0.2. ML_sdfi works and also carries osgeo.gdal, so the
  gdal backend is still used. Run under ML_sdfi.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------
# CONFIG -- edit here if the tree differs from C:\thesis\example_dataset
# ----------------------------------------------------------------------------

DEFAULT_ROOT = r"C:\thesis\multi_channel_dataset_creation\example_dataset"
DEFAULT_OUT = r"C:\thesis\exploratory_data_analysis\chapter2_figures"

CHANNELS = ["rgb", "cir", "DSM", "DTM", "OrtoRGB", "OrtoCIR"]
SPLITTED_REL = os.path.join("data", "splitted")          # <root>/data/splitted/<channel>/
LABELS_REL = os.path.join("labels", "splitted_labels")   # <root>/labels/splitted_labels/
ALL_TXT_REL = os.path.join("data", "all.txt")            # dataset copy, 19,314 active lines

GREENROOF_TILE = "O2021_84_40_1_0039_00062718_2000_3000"
SOLAR_TILE = "O2021_82_19_1_0023_00005474_8000_1000"
URBAN_TILE = "O2021_82_21_1_0004_00002040_4000_2000"
HIGH_IGNORE_TILE = "O2023_82_24_1_0012_00000540_1000_5000"

# Label overlays, rendered in this order. Each entry is (tile, why it is here).
# Edit this list to change which scenes get an overlay. Every tile named here
# must also have an entry in EXPECTED_COUNTS below, otherwise its class names
# cannot be verified and the legend falls back to raw pixel values.
OVERLAY_TILES = [
    (GREENROOF_TILE,
     "green roof scene, the only overlay carrying green_roof, and it also"
     " carries rooftop solceller"),
    (HIGH_IGNORE_TILE,
     "sparse annotation at 76.7 per cent unlabelled, and the scene where"
     " ground resembling ubefestet is left unlabelled"),
    (SOLAR_TILE,
     "ground-mounted solar park, settles whether solceller covers the panels"
     " alone or the whole installation including the strips between rows"),
    (URBAN_TILE,
     "industrial yard, already in the chapter as fig:scene-urban"),
]

ENTANGLED_ROUTES = ["82-20", "82-21", "85-45", "85-48"]

STRETCH_PERCENTILES = (2.0, 98.0)   # plan B2
NDSM_RANGE = (0.0, 15.0)            # plan B2, fixed so building height reads off the ramp
GT_TOL_M = 0.5                      # origin tolerance; the defect displaces >= 100 m
NODATA_FLOOR = -1000.0              # mask elevation values at/below this (the -9999 sentinel)
OVERLAY_ALPHA = 0.45

# Hypothesised raster-value order 0..10. NOT trusted blindly: verified per tile
# against the embedded pixel counts below before any name is printed.
CLASS_ORDER = ["unknown", "asfalt", "fliser", "grus", "ubefestet", "green_roof",
               "drivhus", "betonflade", "brosten", "unknown2", "solceller"]

CLASS_COLOURS = {          # unknown stays white so unlabelled area reads as absence
    "unknown": "#ffffff",
    "asfalt": "#e7298a",   # magenta, see the note in the header
    "fliser": "#e08214",
    "grus": "#d8b365",
    "ubefestet": "#7fbc41",
    "green_roof": "#276419",
    "drivhus": "#35978f",
    "betonflade": "#bababa",
    "brosten": "#9970ab",
    "unknown2": "#ff00ff",  # vestigial, must never appear; magenta screams if it does
    "solceller": "#2166ac",
}

# Per-class pixel counts from tile_inventory.csv (label-raster content), used to
# verify the value->class mapping and that the right raster was opened.
EXPECTED_COUNTS = {
    GREENROOF_TILE: {"unknown": 456806, "asfalt": 256787, "ubefestet": 144536,
                     "green_roof": 100110, "brosten": 23844, "solceller": 17917},
    SOLAR_TILE: {"unknown": 30400, "grus": 19336, "ubefestet": 56035,
                 "solceller": 894229},
    URBAN_TILE: {"unknown": 162683, "asfalt": 483308, "fliser": 276134,
                 "grus": 49411, "ubefestet": 19455, "betonflade": 5691,
                 "solceller": 3318},
    HIGH_IGNORE_TILE: {"unknown": 766731, "asfalt": 82602, "fliser": 58038,
                       "grus": 936, "ubefestet": 27021, "drivhus": 616,
                       "brosten": 64056},
    "O2023_82_24_1_0011_00000393_13000_0": {
        "unknown": 757272, "asfalt": 103051, "fliser": 63508, "grus": 936,
        "ubefestet": 21854, "drivhus": 616, "brosten": 52763},
    "O2021_84_40_1_0039_00062721_1000_0": {
        "unknown": 754697, "asfalt": 123956, "fliser": 22067, "ubefestet": 69053,
        "green_roof": 21103, "brosten": 6645, "solceller": 2479},
}

# ----------------------------------------------------------------------------
# Raster backend: osgeo.gdal preferred (multi-channel-env has GDAL 3.10.2),
# rasterio as fallback. Both return the GDAL-style 6-tuple geotransform.
# ----------------------------------------------------------------------------

BACKEND = None
try:
    from osgeo import gdal  # noqa: N813
    gdal.UseExceptions()
    BACKEND = "gdal"
except ImportError:
    try:
        import rasterio
        BACKEND = "rasterio"
    except ImportError:
        BACKEND = None

import numpy as np  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def read_header(path):
    """(width, height, geotransform) without reading pixels."""
    if BACKEND == "gdal":
        ds = gdal.Open(str(path))
        if ds is None:
            raise IOError("gdal could not open %s" % path)
        gt = ds.GetGeoTransform()
        w, h = ds.RasterXSize, ds.RasterYSize
        ds = None
        return w, h, tuple(gt)
    with rasterio.open(str(path)) as src:
        t = src.transform
        return src.width, src.height, (t.c, t.a, t.b, t.f, t.d, t.e)


def read_raster(path):
    """(array, geotransform, nodata). Array is (bands,H,W) or (H,W) for 1 band."""
    if BACKEND == "gdal":
        ds = gdal.Open(str(path))
        if ds is None:
            raise IOError("gdal could not open %s" % path)
        arr = ds.ReadAsArray()
        gt = tuple(ds.GetGeoTransform())
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        ds = None
        return arr, gt, nodata
    with rasterio.open(str(path)) as src:
        arr = src.read()
        if arr.shape[0] == 1:
            arr = arr[0]
        t = src.transform
        return arr, (t.c, t.a, t.b, t.f, t.d, t.e), src.nodatavals[0]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def tile_path(paths, channel, tile):
    return paths["splitted"] / channel / (tile + ".tif")


def label_path(paths, tile):
    return paths["labels"] / (tile + ".tif")


def gt_origin(gt):
    return (gt[0], gt[3])


def origins_match(gt_a, gt_b, tol=GT_TOL_M):
    return abs(gt_a[0] - gt_b[0]) <= tol and abs(gt_a[3] - gt_b[3]) <= tol


def route_of(tile_name):
    parts = tile_name.split("_")
    return "%s-%s" % (parts[1], parts[2]) if len(parts) >= 3 else "?"


def check_geotransforms(paths, tile, report):
    """Record header geotransforms of all six channels plus the label for one
    tile, compared against rgb. Returns dict channel -> bool(match) or None."""
    result = {}
    detail = {}
    try:
        _, _, gt_rgb = read_header(tile_path(paths, "rgb", tile))
    except Exception as exc:
        report["geotransform_checks"][tile] = {"error": "rgb unreadable: %s" % exc}
        return None
    for ch in CHANNELS:
        p = tile_path(paths, ch, tile)
        try:
            _, _, gt = read_header(p)
            result[ch] = origins_match(gt, gt_rgb)
            detail[ch] = {"origin": gt_origin(gt), "match_rgb": result[ch]}
        except Exception as exc:
            result[ch] = None
            detail[ch] = {"error": str(exc)}
    lp = label_path(paths, tile)
    try:
        _, _, gt = read_header(lp)
        result["label"] = origins_match(gt, gt_rgb)
        detail["label"] = {"origin": gt_origin(gt), "match_rgb": result["label"]}
    except Exception as exc:
        result["label"] = None
        detail["label"] = {"error": str(exc)}
    detail["rgb_origin"] = gt_origin(gt_rgb)
    report["geotransform_checks"][tile] = detail
    return result


def verify_label_mapping(label_arr, tile, report):
    """Compare pixel counts against EXPECTED_COUNTS. Returns (mapping, status):
    mapping is value->class-name for values present; status is one of
    'confirmed', 'inferred', 'unverified'."""
    counts = np.bincount(label_arr.ravel().astype(np.int64), minlength=256)
    present = {int(v): int(counts[v]) for v in range(256) if counts[v] > 0}
    high = [v for v in present if v > 10]
    expected = EXPECTED_COUNTS.get(tile)
    entry = {"observed_value_counts": present, "values_above_10": high}
    if expected is None:
        entry["status"] = "unverified"
        report["label_mapping"][tile] = entry
        return {v: "value_%d" % v for v in present}, "unverified"
    exp_full = {c: expected.get(c, 0) for c in CLASS_ORDER}
    hyp_ok = all(int(counts[i]) == exp_full[c] for i, c in enumerate(CLASS_ORDER))
    if hyp_ok and not high:
        entry["status"] = "confirmed"
        report["label_mapping"][tile] = entry
        return {i: c for i, c in enumerate(CLASS_ORDER) if counts[i] > 0}, "confirmed"
    # try to infer a permutation from exact count matches
    inferred, used = {}, set()
    ambiguous = False
    for v, n in present.items():
        matches = [c for c, e in exp_full.items() if e == n and c not in used]
        if len(matches) == 1:
            inferred[v] = matches[0]
            used.add(matches[0])
        else:
            ambiguous = True
    if not ambiguous and set(exp_full[c] for c in used) == set(present.values()) \
            and len(inferred) == len(present):
        entry["status"] = "inferred"
        entry["inferred_mapping"] = {str(k): v for k, v in inferred.items()}
        report["label_mapping"][tile] = entry
        return inferred, "inferred"
    entry["status"] = "unverified"
    entry["note"] = "counts do not reproduce tile_inventory.csv for this tile"
    report["label_mapping"][tile] = entry
    return {v: "value_%d" % v for v in present}, "unverified"


# ----------------------------------------------------------------------------
# Panel rendering. Every panel shares one canvas geometry: image axes on the
# left, a fixed right-hand strip that carries the colour bar, the legend, or
# nothing, so panels sit level in a LaTeX subfigure row.
# ----------------------------------------------------------------------------

FIGSIZE = (6.2, 5.0)
IMG_RECT = [0.015, 0.03, 0.775, 0.94]
CBAR_RECT = [0.845, 0.08, 0.035, 0.84]
LEGEND_RECT = [0.795, 0.03, 0.20, 0.94]
DPI = 200


def _new_canvas():
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(IMG_RECT)
    ax.set_axis_off()
    return fig, ax


def save_continuous_panel(arr, out_png, vmin, vmax, cbar_label, cmap="viridis"):
    fig, ax = _new_canvas()
    cm = plt.get_cmap(cmap)
    try:
        cm = cm.copy()          # matplotlib >= 3.4
    except AttributeError:
        pass                    # older matplotlib mutates the shared instance
    cm.set_bad("#eeeeee")
    im = ax.imshow(arr, cmap=cm, vmin=vmin, vmax=vmax, interpolation="nearest")
    cax = fig.add_axes(CBAR_RECT)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def save_rgb_panel(rgb_hwc, out_png):
    fig, ax = _new_canvas()
    ax.imshow(rgb_hwc, interpolation="nearest")
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def label_to_rgb(label_arr, mapping):
    h, w = label_arr.shape
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    for v, cname in mapping.items():
        colour = CLASS_COLOURS.get(cname, "#000000")
        r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        m = label_arr == v
        out[m] = (r, g, b)
    return out


def legend_handles(mapping, include_unknown=True):
    handles = []
    ordered = sorted(mapping.items(), key=lambda kv: kv[0])
    for v, cname in ordered:
        if cname == "unknown":
            if include_unknown:
                handles.append(Patch(facecolor="#ffffff", edgecolor="#999999",
                                     label="unlabelled"))
            continue
        handles.append(Patch(facecolor=CLASS_COLOURS.get(cname, "#000000"),
                             edgecolor="none", label=cname.replace("_", " ")))
    return handles


def save_label_panel(label_arr, mapping, out_png):
    fig, ax = _new_canvas()
    ax.imshow(label_to_rgb(label_arr, mapping), interpolation="nearest")
    lax = fig.add_axes(LEGEND_RECT)
    lax.set_axis_off()
    lax.legend(handles=legend_handles(mapping), loc="center left", fontsize=9,
               frameon=False, handlelength=1.2, borderaxespad=0)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def save_overlay_panel(rgb_hwc, label_arr, mapping, out_png):
    tinted = rgb_hwc.astype(np.float32)
    lab_rgb = label_to_rgb(label_arr, mapping).astype(np.float32)
    labelled = label_arr != 0
    a = OVERLAY_ALPHA
    tinted[labelled] = (1 - a) * tinted[labelled] + a * lab_rgb[labelled]
    fig, ax = _new_canvas()
    ax.imshow(tinted.astype(np.uint8), interpolation="nearest")
    lax = fig.add_axes(LEGEND_RECT)
    lax.set_axis_off()
    lax.legend(handles=legend_handles(mapping, include_unknown=False),
               loc="center left", fontsize=9, frameon=False, handlelength=1.2)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


def save_standalone_legend(out_png):
    fig = plt.figure(figsize=(2.6, 3.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    mapping = {i: c for i, c in enumerate(CLASS_ORDER) if c != "unknown2"}
    ax.legend(handles=legend_handles(mapping), loc="center", fontsize=10,
              frameon=False)
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Elevation loading
# ----------------------------------------------------------------------------

def load_elevation(path):
    arr, gt, nodata = read_raster(path)
    arr = arr.astype(np.float64)
    invalid = ~np.isfinite(arr) | (arr <= NODATA_FLOOR)
    if nodata is not None:
        invalid |= (arr == nodata)
    masked = np.ma.array(arr, mask=invalid)
    return masked, gt, int(invalid.sum())


def stretch_range(masked_arrays):
    vals = np.concatenate([np.asarray(m.compressed(), dtype=np.float64)
                           for m in masked_arrays])
    if vals.size == 0:
        raise ValueError("no valid elevation pixels to stretch over")
    lo, hi = np.percentile(vals, STRETCH_PERCENTILES)
    return float(lo), float(hi)


# ----------------------------------------------------------------------------
# Jobs
# ----------------------------------------------------------------------------

def outputs_exist(paths_list):
    return all(p.exists() for p in paths_list)


def job_elevation_greenroof(paths, out_dir, force, per_panel, report):
    name = "elevation_greenroof"
    tile = GREENROOF_TILE
    dst = out_dir / "2.1.2.elevation"
    outs = [dst / ("%s_%s.png" % (k, tile)) for k in ("DSM", "DTM", "nDSM")]
    job = {"tile": tile, "outputs": [str(p) for p in outs]}
    report["jobs"][name] = job
    if outputs_exist(outs) and not force:
        job["status"] = "found_existing_skipped"
        print("[%s] outputs already present, skipped (use --force to redo)" % name)
        return
    gtc = check_geotransforms(paths, tile, report)
    if gtc is None:
        job["status"] = "failed_rgb_unreadable"
        print("[%s] FAILED: rgb tile unreadable" % name)
        return
    job["geotransform_ok"] = gtc
    if not (gtc.get("DSM") and gtc.get("DTM")):
        job["status"] = "BLOCKED_misaligned_elevation"
        print("[%s] BLOCKED: DSM/DTM do not share the rgb geotransform for the"
              " green roof tile. The plan-B2 payload tile must be re-chosen."
              " Nothing rendered." % name)
        return
    dsm, _, ndsm_bad = load_elevation(tile_path(paths, "DSM", tile))
    dtm, _, _ = load_elevation(tile_path(paths, "DTM", tile))
    # content cross-check via the label raster
    try:
        lab, _, _ = read_raster(label_path(paths, tile))
        _, status = verify_label_mapping(lab, tile, report)
        job["label_content_check"] = status
    except Exception as exc:
        job["label_content_check"] = "unreadable: %s" % exc
    dst.mkdir(parents=True, exist_ok=True)
    if per_panel:
        r_dsm = stretch_range([dsm])
        r_dtm = stretch_range([dtm])
    else:
        r_dsm = r_dtm = stretch_range([dsm, dtm])
    ndsm = dsm - dtm
    nv = ndsm.compressed() if isinstance(ndsm, np.ma.MaskedArray) \
        else np.asarray(ndsm).ravel()
    job["ndsm_clipped_below_0_pct"] = round(100.0 * float((nv < NDSM_RANGE[0]).mean()), 3)
    job["ndsm_clipped_above_15_pct"] = round(100.0 * float((nv > NDSM_RANGE[1]).mean()), 3)
    job["stretch_dsm_m"] = [round(v, 2) for v in r_dsm]
    job["stretch_dtm_m"] = [round(v, 2) for v in r_dtm]
    job["stretch_joint"] = not per_panel
    save_continuous_panel(dsm, outs[0], r_dsm[0], r_dsm[1], "m a.s.l.")
    save_continuous_panel(dtm, outs[1], r_dtm[0], r_dtm[1], "m a.s.l.")
    save_continuous_panel(np.ma.clip(ndsm, *NDSM_RANGE), outs[2],
                          NDSM_RANGE[0], NDSM_RANGE[1], "m above terrain")
    job["status"] = "rendered"
    print("[%s] rendered 3 panels; DSM/DTM stretch %s to %s m (joint=%s);"
          " nDSM fixed 0-15 m, %.2f%% above 15 m clipped"
          % (name, job["stretch_dsm_m"][0], job["stretch_dsm_m"][1],
             not per_panel, job["ndsm_clipped_above_15_pct"]))


def job_misreg_solar(paths, out_dir, force, report):
    name = "misreg_solar"
    tile = SOLAR_TILE
    dst = out_dir / "2.3.misregistration"
    outs = [dst / ("%s_%s.png" % (k, tile)) for k in ("rgb", "DTM", "DSM")]
    job = {"tile": tile, "outputs": [str(p) for p in outs]}
    report["jobs"][name] = job
    if outputs_exist(outs) and not force:
        job["status"] = "found_existing_skipped"
        print("[%s] outputs already present, skipped" % name)
        return
    gtc = check_geotransforms(paths, tile, report)
    if gtc is None:
        job["status"] = "failed_rgb_unreadable"
        print("[%s] FAILED: rgb tile unreadable" % name)
        return
    job["geotransform_ok"] = gtc
    if gtc.get("DTM") is not True:
        print("[%s] WARNING: DTM does not match rgb here, expected aligned" % name)
    if gtc.get("DSM") is True:
        print("[%s] WARNING: DSM MATCHES rgb, but this tile was chosen as the"
              " displaced-DSM exhibit. Check the tile before using the figure." % name)
    job["exhibit_as_expected"] = (gtc.get("DTM") is True and gtc.get("DSM") is False)
    try:
        lab, _, _ = read_raster(label_path(paths, tile))
        _, status = verify_label_mapping(lab, tile, report)
        job["label_content_check"] = status
    except Exception as exc:
        job["label_content_check"] = "unreadable: %s" % exc
    dst.mkdir(parents=True, exist_ok=True)
    rgb, _, _ = read_raster(tile_path(paths, "rgb", tile))
    save_rgb_panel(np.transpose(rgb, (1, 2, 0)), outs[0])
    dtm, _, _ = load_elevation(tile_path(paths, "DTM", tile))
    dsm, _, _ = load_elevation(tile_path(paths, "DSM", tile))
    r_dtm = stretch_range([dtm])
    r_dsm = stretch_range([dsm])
    job["stretch_dtm_m"] = [round(v, 2) for v in r_dtm]
    job["stretch_dsm_m"] = [round(v, 2) for v in r_dsm]
    save_continuous_panel(dtm, outs[1], r_dtm[0], r_dtm[1], "m a.s.l.")
    save_continuous_panel(dsm, outs[2], r_dsm[0], r_dsm[1], "m a.s.l.")
    job["status"] = "rendered"
    print("[%s] rendered 3 panels; exhibit_as_expected=%s"
          % (name, job["exhibit_as_expected"]))


def job_overlay(paths, out_dir, force, report, tile, purpose=""):
    name = "overlay_%s" % tile
    dst = out_dir / "2.2.label-overlay"
    outs = [dst / ("%s_%s.png" % (k, tile)) for k in ("rgb", "labels", "overlay")]
    job = {"tile": tile, "purpose": purpose, "outputs": [str(p) for p in outs]}
    report["jobs"][name] = job
    if outputs_exist(outs) and not force:
        job["status"] = "found_existing_skipped"
        print("[%s] outputs already present, skipped" % name)
        return True
    gtc = check_geotransforms(paths, tile, report)
    if gtc is None:
        job["status"] = "failed_rgb_unreadable"
        print("[%s] FAILED: rgb tile unreadable" % name)
        return False
    job["geotransform_ok"] = gtc
    if gtc.get("label") is not True:
        job["status"] = "BLOCKED_label_misplaced"
        print("[%s] BLOCKED: label geotransform disagrees with rgb, so this tile"
              " is among the 273 and cannot serve as an overlay figure."
              " Continuing with the remaining tiles." % name)
        return False
    lab, _, _ = read_raster(label_path(paths, tile))
    mapping, status = verify_label_mapping(lab, tile, report)
    job["label_mapping_status"] = status
    if status != "confirmed":
        print("[%s] NOTE: class-value mapping %s for this tile, legend %s"
              % (name, status,
                 "uses inferred names, verify against codes.txt" if status == "inferred"
                 else "falls back to raw values"))
    rgb, _, _ = read_raster(tile_path(paths, "rgb", tile))
    rgb_hwc = np.transpose(rgb, (1, 2, 0))
    dst.mkdir(parents=True, exist_ok=True)
    save_rgb_panel(rgb_hwc, outs[0])
    save_label_panel(lab, mapping, outs[1])
    save_overlay_panel(rgb_hwc, lab, mapping, outs[2])
    job["ignore_share_pct"] = round(100.0 * float((lab == 0).mean()), 2)
    job["status"] = "rendered"
    print("[%s] rendered rgb/labels/overlay; ignore share %.1f%%; mapping %s"
          % (name, job["ignore_share_pct"], status))
    return True


def job_bounds(paths, out_dir, force, report):
    name = "bounds_entangled_routes"
    out_csv = out_dir / "bounds_entangled_routes.csv"
    job = {"outputs": [str(out_csv)], "routes": ENTANGLED_ROUTES}
    report["jobs"][name] = job
    if out_csv.exists() and not force:
        job["status"] = "found_existing_skipped"
        print("[%s] output already present, skipped" % name)
        return
    tiles = []
    all_txt = paths["all_txt"]
    if all_txt.exists():
        with open(all_txt) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                base = os.path.basename(line.replace("\\", "/"))
                if base.lower().endswith(".tif"):
                    base = base[:-4]
                if route_of(base) in ENTANGLED_ROUTES:
                    tiles.append(base)
        job["tile_source"] = str(all_txt)
    else:
        print("[%s] all.txt not found at %s, globbing rgb folder instead"
              % (name, all_txt))
        for r in ENTANGLED_ROUTES:
            pat = str(paths["splitted"] / "rgb" / ("O????_%s_*.tif" % r.replace("-", "_")))
            tiles += [Path(p).stem for p in glob.glob(pat)]
        job["tile_source"] = "glob of rgb folder"
    job["n_tiles_listed"] = len(tiles)
    rows, errors = [], 0
    for i, t in enumerate(tiles):
        try:
            w, h, gt = read_header(tile_path(paths, "rgb", t))
            minx = gt[0]
            maxy = gt[3]
            maxx = gt[0] + w * gt[1]
            miny = gt[3] + h * gt[5]
            rows.append((t + ".tif", route_of(t), minx, min(miny, maxy),
                         maxx, max(miny, maxy)))
        except Exception:
            errors += 1
        if (i + 1) % 250 == 0:
            print("[%s] %d / %d headers read" % (name, i + 1, len(tiles)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("filename,route,minx,miny,maxx,maxy\n")
        for r in rows:
            f.write("%s,%s,%.2f,%.2f,%.2f,%.2f\n" % r)
    job["n_rows_written"] = len(rows)
    job["n_errors"] = errors
    job["status"] = "rendered"
    print("[%s] wrote %d rows (%d errors) to %s" % (name, len(rows), errors, out_csv))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def resolve_paths(args):
    root = Path(args.root)
    paths = {
        "root": root,
        "splitted": Path(args.splitted_dir) if args.splitted_dir else root / SPLITTED_REL,
        "labels": Path(args.labels_dir) if args.labels_dir else root / LABELS_REL,
        "all_txt": Path(args.all_txt) if args.all_txt else root / ALL_TXT_REL,
    }
    return paths


def preflight(paths, out_dir):
    print("=" * 72)
    print("PRE-FLIGHT")
    print("  raster backend : %s" % (BACKEND or "NONE -- install GDAL python"
                                     " bindings or rasterio"))
    print("  matplotlib     : %s" % ("present" if HAVE_MPL else
                                     "MISSING -- pip install matplotlib"))
    ok = BACKEND is not None and HAVE_MPL
    for key in ("splitted", "labels", "all_txt"):
        exists = paths[key].exists()
        print("  %-14s : %s  [%s]" % (key, paths[key], "found" if exists else "MISSING"))
        if key != "all_txt":
            ok = ok and exists
    for ch in CHANNELS:
        d = paths["splitted"] / ch
        if not d.exists():
            print("  channel folder MISSING: %s" % d)
            ok = False
    print("  target tiles:")
    seen = []
    for t in [GREENROOF_TILE, SOLAR_TILE] + [x[0] for x in OVERLAY_TILES]:
        if t in seen:
            continue
        seen.append(t)
        pr = tile_path(paths, "rgb", t)
        pl = label_path(paths, t)
        known = "yes" if t in EXPECTED_COUNTS else "NO, legend unverifiable"
        print("    %-45s rgb:%-5s label:%-5s counts:%s"
              % (t, "yes" if pr.exists() else "NO",
                 "yes" if pl.exists() else "NO", known))
    print("  output folder  : %s" % out_dir)
    expected = [
        out_dir / "2.1.2.elevation" / ("DSM_%s.png" % GREENROOF_TILE),
        out_dir / "2.1.2.elevation" / ("DTM_%s.png" % GREENROOF_TILE),
        out_dir / "2.1.2.elevation" / ("nDSM_%s.png" % GREENROOF_TILE),
        out_dir / "2.3.misregistration" / ("rgb_%s.png" % SOLAR_TILE),
        out_dir / "2.3.misregistration" / ("DTM_%s.png" % SOLAR_TILE),
        out_dir / "2.3.misregistration" / ("DSM_%s.png" % SOLAR_TILE),
        out_dir / "bounds_entangled_routes.csv",
    ] + [out_dir / "2.2.label-overlay" / ("overlay_%s.png" % t)
         for t, _ in OVERLAY_TILES]
    n_have = sum(1 for p in expected if p.exists())
    print("  existing outputs: %d of %d expected files already present"
          % (n_have, len(expected)))
    for p in expected:
        if p.exists():
            print("    found: %s" % p)
    print("=" * 72)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--splitted-dir", default=None,
                    help="override <root>/data/splitted")
    ap.add_argument("--labels-dir", default=None,
                    help="override <root>/labels/splitted_labels")
    ap.add_argument("--all-txt", default=None, help="override <root>/data/all.txt")
    ap.add_argument("--check", action="store_true",
                    help="report environment, inputs and existing outputs, render nothing")
    ap.add_argument("--force", action="store_true", help="re-render existing outputs")
    ap.add_argument("--per-panel-stretch", action="store_true",
                    help="stretch DSM and DTM independently in the 2.1.2 figure"
                         " (default is one joint range so the panels compare)")
    ap.add_argument("--skip-bounds", action="store_true")
    args = ap.parse_args()

    paths = resolve_paths(args)
    out_dir = Path(args.out)
    ok = preflight(paths, out_dir)
    if args.check:
        print("--check: nothing rendered.")
        return 0
    if not ok:
        print("Pre-flight failed. Fix the items marked MISSING above"
              " (or pass --splitted-dir / --labels-dir overrides).")
        return 1

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "backend": BACKEND, "root": str(paths["root"]),
              "out": str(out_dir), "force": args.force,
              "jobs": {}, "geotransform_checks": {}, "label_mapping": {}}

    out_dir.mkdir(parents=True, exist_ok=True)
    job_elevation_greenroof(paths, out_dir, args.force, args.per_panel_stretch, report)
    job_misreg_solar(paths, out_dir, args.force, report)
    for tile, purpose in OVERLAY_TILES:
        job_overlay(paths, out_dir, args.force, report, tile, purpose)
    legend_png = out_dir / "2.2.label-overlay" / "legend_classes.png"
    if args.force or not legend_png.exists():
        legend_png.parent.mkdir(parents=True, exist_ok=True)
        save_standalone_legend(legend_png)
    if not args.skip_bounds:
        job_bounds(paths, out_dir, args.force, report)

    with open(out_dir / "render_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 72)
    print("SUMMARY FOR THE SESSION (paste this block back)")
    for name, job in report["jobs"].items():
        print("  %-28s %s" % (name, job.get("status", "?")))
        if "exhibit_as_expected" in job:
            print("  %-28s exhibit_as_expected=%s" % ("", job["exhibit_as_expected"]))
        if "geotransform_ok" in job:
            bad = [k for k, v in job["geotransform_ok"].items() if v is not True]
            print("  %-28s non-matching rasters: %s" % ("", bad if bad else "none"))
        for k in ("stretch_dsm_m", "stretch_dtm_m",
                  "ndsm_clipped_above_15_pct", "label_mapping_status",
                  "ignore_share_pct", "n_rows_written"):
            if k in job:
                print("  %-28s %s=%s" % ("", k, job[k]))
    print("  full detail: %s" % (out_dir / "render_report.json"))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
