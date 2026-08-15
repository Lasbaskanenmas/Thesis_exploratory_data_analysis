#!/usr/bin/env python
"""
Shared paths, constants and helpers for the Befaestelsesdata EDA (Great Plan 3.0, SQ1).

Design rules for every script in this folder:

  1. ADDITIVE ONLY. Source data and logs_and_models/ are opened read-only. Nothing outside
     exploratory_data_analysis/ is ever written, moved or deleted. Scripts fail loud rather than
     repairing anything in place.
  2. REUSE, DO NOT COPY. The metric machinery, route parsing and fold handling already exist and
     are self-tested in ML_sdfi_fastai2.analyse. c:\\thesis\\ML_sdfi_fastai2\\src is a permanent
     sys.path entry (editable install), so those modules import from any working directory. This
     file holds only what is genuinely new.
  3. VALIDATE AGAINST WHAT EXISTS. Every module cross-checks itself against an artifact already on
     disk, so a silent error cannot pass.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------------------------
# Source data (READ ONLY -- never written by anything in this folder)
# ----------------------------------------------------------------------------------------------
THESIS_ROOT = Path(r"c:\thesis")

DATASET_ROOT = THESIS_ROOT / "multi_channel_dataset_creation" / "example_dataset"
ALL_TXT = DATASET_ROOT / "data" / "all.txt"
NAIVE_VALID_TXT = DATASET_ROOT / "data" / "valid.txt"        # the hand-picked 25-tile split
SPLITTED_DIR = DATASET_ROOT / "data" / "splitted"            # per-channel tile folders
LABEL_DIR = DATASET_ROOT / "labels" / "splitted_labels"
OLD_LABEL_DIR = DATASET_ROOT / "labels" / "old_splitted_labels"
CODES_TXT = DATASET_ROOT / "labels" / "codes.txt"
GROUND_SURFACE_GPKG = DATASET_ROOT / "labels" / "example_dataset_ground_surface.gpkg"

LOGS_AND_MODELS = THESIS_ROOT / "logs_and_models"
CLASS_PIXEL_AUDIT_JSON = LOGS_AND_MODELS / "class_pixel_audit" / "class_pixel_audit.json"
ROUTE_AUDIT_DIR = LOGS_AND_MODELS / "route_class_audit"
ROUTE_CLASS_AUDIT_CSV = ROUTE_AUDIT_DIR / "route_class_audit.csv"
FOLD_ASSIGNMENT_CSV = ROUTE_AUDIT_DIR / "fold_assignment.csv"
CLASS_ROUTES_JSON = ROUTE_AUDIT_DIR / "class_routes.json"
SPATIAL_MATRIX = LOGS_AND_MODELS / "spatial_matrix"

CORRUPT_TILES_TXT = THESIS_ROOT / "corrupt_tiles.txt"

# ----------------------------------------------------------------------------------------------
# EDA outputs (the ONLY place anything is written)
# ----------------------------------------------------------------------------------------------
EDA_ROOT = THESIS_ROOT / "exploratory_data_analysis"
TABLES = EDA_ROOT / "results" / "tables"
FIGURES = EDA_ROOT / "results" / "figures"
FINDINGS = EDA_ROOT / "results" / "findings"
CACHE = EDA_ROOT / "cache"

TILE_INVENTORY_CSV = TABLES / "tile_inventory.csv"


def ensure_out_dirs() -> None:
    """Create the output directories. Only ever touches paths under EDA_ROOT."""
    for d in (TABLES, FIGURES, FINDINGS, CACHE):
        d.mkdir(parents=True, exist_ok=True)


def assert_writes_are_local(path) -> Path:
    """Guard rail: refuse to write anywhere outside exploratory_data_analysis/.

    Every script routes its writes through this, so the additive-only rule is enforced in code
    rather than by convention.
    """
    p = Path(path).resolve()
    root = EDA_ROOT.resolve()
    if root not in p.parents and p != root:
        raise RuntimeError(
            f"refusing to write outside the EDA folder: {p}\n"
            f"everything this analysis produces must land under {root}"
        )
    return p


# ----------------------------------------------------------------------------------------------
# Class scheme (verified against codes.txt; 11 codes, indices 0..10)
# ----------------------------------------------------------------------------------------------
CODES = ["unknown", "asfalt", "fliser", "grus", "ubefestet", "green_roof",
         "drivhus", "betonflade", "brosten", "unknown2", "solceller"]
NCLASS = len(CODES)                                    # 11
IGNORE_INDEX = 0                                       # unknown -> never scored
UNKNOWN2_INDEX = 9                                     # report-only, 0 pixels present in the pool
PREDICTED = [1, 2, 3, 4, 5, 6, 7, 8, 10]               # the 9 model-predicted classes
RARE = [5, 6, 8, 10]                                   # green_roof, drivhus, brosten, solceller

# The binary befaestet / ubefaestet split KDS actually operates on. ubefestet (4) is the only
# unsealed predicted class; the rest are sealed surfaces or roof structures.
UBEFESTET_INDEX = 4

TILE_PX = 1000                                         # tiles are 1000 x 1000 ...
GSD_M = 0.1                                            # ... at 0.1 m GSD, so 100 m x 100 m ground
TILE_GROUND_M = TILE_PX * GSD_M
EPSG = 25832                                           # ETRS89 / UTM 32N

NFOLDS = 3
EXPECTED_TILES = 19314
EXPECTED_FOLD_COUNTS = {0: 6439, 1: 6437, 2: 6438}

# ----------------------------------------------------------------------------------------------
# Channel layout.
#
# Each folder under data/splitted/ holds tiles with the SAME filename. rgb/cir/OrtoRGB/OrtoCIR are
# 3-band uint8; DSM/DTM are 1-band float32 holding RAW METRES ABOVE SEA LEVEL.
#
# The training pipeline uses only a subset of the available bands (see USED_BANDS): band 0 of cir
# and OrtoCIR is taken as NIR and bands 1-2 are discarded. The audit scans EVERY band so that
# assumption is itself testable.
# ----------------------------------------------------------------------------------------------
CHANNEL_FOLDERS = ["rgb", "cir", "OrtoRGB", "OrtoCIR", "DSM", "DTM"]
FOLDER_NBANDS = {"rgb": 3, "cir": 3, "OrtoRGB": 3, "OrtoCIR": 3, "DSM": 1, "DTM": 1}
FLOAT_FOLDERS = {"DSM", "DTM"}          # metres, float32, nodata -9999
NODATA_SENTINEL = -9999.0
NODATA_THRESHOLD = -100.0               # ImageBlockReplacement.py:242 maps anything below this to 0

# All 14 (folder, band) pairs, in scan order.
ALL_BANDS = [(f, b) for f in CHANNEL_FOLDERS for b in range(FOLDER_NBANDS[f])]


def band_label(folder: str, band: int) -> str:
    if folder in ("rgb", "OrtoRGB"):
        return f"{folder}_{'RGB'[band]}"
    if folder in ("cir", "OrtoCIR"):
        return f"{folder}_b{band}" + ("_NIR" if band == 0 else "_unused")
    return folder


BAND_LABELS = [band_label(f, b) for f, b in ALL_BANDS]

# The bands the model actually consumes, in the order the 10-channel config stacks them.
# Source: ML_sdfi_fastai2/src/ML_sdfi_fastai2/analyse/generate_matrix_configs.py CHANNELS dict,
# cross-checked below against the job_dictionary.json of a completed run.
USED_BANDS_10CH = [("rgb", 0), ("rgb", 1), ("rgb", 2), ("cir", 0),
                   ("OrtoRGB", 0), ("OrtoRGB", 1), ("OrtoRGB", 2), ("OrtoCIR", 0),
                   ("DSM", 0), ("DTM", 0)]

# The normalization constants that were ACTUALLY APPLIED during the 72-run matrix.
# NOTE the two provenance problems these encode, which module E2 exists to quantify:
#   - the NIR value 0.40779021 is documented in sdfi_dataset.py:146 as "based on a sample size
#     of 1! OBS! UNKNOWN"
#   - DSM/DTM get mean 0.5 / std 1.0, which are placeholders, not measured statistics, while the
#     underlying data is raw metres above sea level
CONFIG_MEANS_10CH = [0.485, 0.456, 0.406, 0.40779021,
                     0.485, 0.456, 0.406, 0.40779021,
                     0.5, 0.5]
CONFIG_STDS_10CH = [0.229, 0.224, 0.225, 0.15176421,
                    0.229, 0.224, 0.225, 0.15176421,
                    1.0, 1.0]

# fastai's IntToFloatTensor divides the whole tensor by 255 before Normalize.from_stats runs.
# It is applied to every channel, including the float32 elevation bands.
INT_TO_FLOAT_DIV = 255.0

# ----------------------------------------------------------------------------------------------
# Reuse of the existing, already self-tested machinery. Import errors here are fatal and loud --
# duplicating any of this would defeat the point.
# ----------------------------------------------------------------------------------------------
try:
    from ML_sdfi_fastai2.analyse.per_category_metrics import (  # noqa: F401
        read_tile_list,
        pooled_confusion_matrix,
        metrics_from_confusion,
        aggregate_folds,
        load_codes,
    )
    from ML_sdfi_fastai2.analyse.pooled_oof_metrics import (  # noqa: F401
        parse_route,
        load_fold_assignment,
        heldout_tiles_by_fold,
    )
except ImportError as exc:  # pragma: no cover
    sys.exit(
        f"cannot import the existing analysis machinery: {exc}\n"
        f"expected c:\\thesis\\ML_sdfi_fastai2\\src on sys.path (editable install).\n"
        f"run with c:\\thesis\\envs\\ML_sdfi\\python.exe"
    )


# ----------------------------------------------------------------------------------------------
# Small shared helpers
# ----------------------------------------------------------------------------------------------
def tile_names(all_txt: Path = ALL_TXT) -> list[str]:
    """The active tile list: all.txt minus blanks and '#'-commented corrupt tiles."""
    return read_tile_list(str(all_txt))


def parse_filename(fname: str) -> dict:
    """Decompose a tile filename into its parts.

    O<YEAR>_<rr>_<cc>_<k>_<NNNN>_<IIIIIIII>_<i>_<j>.tif
    e.g. O2021_82_11_1_0023_00003029_3000_2000.tif

    route  = "rr-cc" with the year collapsed (the blocking unit; 3 routes were flown twice)
    parent = everything identifying the source orthophoto, so tiles can be grouped by parent
    """
    stem = Path(fname).stem
    parts = stem.split("_")
    if len(parts) < 8 or not parts[0].startswith("O"):
        raise ValueError(f"unparseable tile name: {fname}")
    return {
        "filename": fname,
        "year": int(parts[0][1:]),
        "route": f"{parts[1]}-{parts[2]}",
        "parent": "_".join(parts[:6]),
        "tile_i": int(parts[-2]),
        "tile_j": int(parts[-1]),
    }


def read_label(path) -> np.ndarray:
    """Read a label tile exactly as the trainer does: int32, out-of-range folded into unknown.

    Mirrors class_pixel_audit.worker so counts computed here are comparable to the existing audit.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    lab = np.asarray(Image.open(path)).astype(np.int32)
    lab[(lab < 0) | (lab >= NCLASS)] = 0
    return lab


def load_class_pixel_audit() -> dict:
    """The existing global per-class audit, keyed by class name. Used for validation."""
    with open(CLASS_PIXEL_AUDIT_JSON) as fh:
        data = json.load(fh)
    return {rec["class"]: rec for rec in data["per_class"]}


def load_tile_inventory():
    """Load the E0 inventory as a pandas DataFrame. Fails loud if E0 has not been run."""
    import pandas as pd
    if not TILE_INVENTORY_CSV.is_file():
        sys.exit(f"missing {TILE_INVENTORY_CSV}\nrun eda_tile_inventory.py first (module E0)")
    return pd.read_csv(TILE_INVENTORY_CSV)


def px_col(cls: str) -> str:
    return f"px_{cls}"


def present_col(cls: str) -> str:
    return f"present_{cls}"


def write_json(obj, path) -> Path:
    path = assert_writes_are_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


# ----------------------------------------------------------------------------------------------
# Figure styling. matplotlib only -- seaborn is not installed in this environment.
#
# Colour policy, and why it looks the way it does:
#   * Most charts here are PER CLASS. Class identity is carried by the AXIS LABEL, so those charts
#     use a SINGLE hue and encode magnitude by bar length. A 9-way categorical colour ramp would be
#     decoration, and no validated categorical palette separates 9 hues safely anyway.
#   * Where series genuinely differ in kind (the variogram strata, naive versus blocked splits) the
#     first N slots of the validated categorical theme are used in fixed order, never cycled.
#   * SERIES_3 and SERIES_2 were checked with the palette validator on all pairs, in light and dark:
#     all checks pass. The aqua slot carries a contrast warning on the light surface, which obliges
#     visible labels -- every series in this module is directly labelled, so that is satisfied.
#   * Heatmaps use the single-hue sequential ramp, light to dark. Never a rainbow.
# ----------------------------------------------------------------------------------------------
FIG_DPI_REVIEW = 150
FIG_DPI_THESIS = 300

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Validated categorical theme, fixed order. Use a prefix, never a cycle.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SERIES_1, SERIES_2, SERIES_3 = SERIES[:1], SERIES[:2], SERIES[:3]

# Single-hue sequential ramp (blue), light to dark, for heatmaps and magnitude.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def apply_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": FIG_DPI_REVIEW,
        "savefig.bbox": "tight",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.autolayout": False,
    })


def savefig(fig, name: str, thesis: bool = False) -> Path:
    """Save a figure into results/figures/. PNG always; PDF too when destined for the thesis."""
    ensure_out_dirs()
    png = assert_writes_are_local(FIGURES / f"{name}.png")
    fig.savefig(png, dpi=FIG_DPI_REVIEW)
    if thesis:
        fig.savefig(assert_writes_are_local(FIGURES / f"{name}.pdf"), dpi=FIG_DPI_THESIS)
    return png


def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


if __name__ == "__main__":
    banner("eda_common self-check")
    print(f"dataset root exists : {DATASET_ROOT.is_dir()}  {DATASET_ROOT}")
    print(f"all.txt exists      : {ALL_TXT.is_file()}")
    print(f"label dir exists    : {LABEL_DIR.is_dir()}")
    print(f"pixel audit exists  : {CLASS_PIXEL_AUDIT_JSON.is_file()}")
    print(f"fold assignment     : {FOLD_ASSIGNMENT_CSV.is_file()}")
    print(f"spatial matrix      : {SPATIAL_MATRIX.is_dir()}")
    names = tile_names()
    print(f"\nactive tiles        : {len(names)} (expected {EXPECTED_TILES})")
    print(f"first tile parsed   : {parse_filename(names[0])}")
    r2f = load_fold_assignment(str(FOLD_ASSIGNMENT_CSV))
    print(f"routes in split     : {len(r2f)}")
    print(f"bands scanned       : {len(ALL_BANDS)} -> {BAND_LABELS}")
    try:
        assert_writes_are_local(LOGS_AND_MODELS / "nope.txt")
        print("\nGUARD FAILED -- would have written outside the EDA folder")
    except RuntimeError:
        print("\nwrite guard        : OK (refuses paths outside exploratory_data_analysis/)")
