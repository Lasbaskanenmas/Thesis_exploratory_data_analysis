#!/usr/bin/env python
"""
Figures for the EDA. Every figure is regenerated from the persisted artifacts in results/tables/,
never from a live scan, so the plots and the numbers can never drift apart.

Modules that have not been run are skipped with a note rather than failing, so this can be run at
any point during the analysis.

Design notes (the reasoning, so later edits do not undo it):
  * Per-class charts put class identity on the AXIS and use a single hue. Bar length encodes the
    magnitude. Nine categorical hues would be decoration, and no validated categorical palette
    separates nine hues safely for colour-vision-deficient readers.
  * Charts with series of genuinely different KIND use the first N slots of the validated
    categorical theme, in fixed order, and every series is directly labelled.
  * Heatmaps use one sequential hue, light to dark. Never a rainbow, never a hue at a midpoint.

    python eda_figures.py [--thesis]
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


def _load_json(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def _note(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            color=C.INK_MUTED, fontsize=9)
    ax.set_axis_off()


# ----------------------------------------------------------------------------------------------
def fig_semivariogram(plt, thesis: bool):
    """THE figure: how far apart two tiles must be before they are independent."""
    p = C.TABLES / "spatial_dependence_curves.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    centers, gamma, moran, count = z["centers"], z["gamma"], z["moran"], z["count"]
    sill = float(z["sill"][0])
    meta = _load_json(C.TABLES / "spatial_dependence.json") or {}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    # Beyond ~20 km pairs span disjoint parts of the country and the estimate becomes unstable on
    # few, unrelated pairs. The structure worth reading lives between one tile and a few km.
    ok = np.isfinite(gamma) & (count > 0) & (centers <= 20_000)

    # The sill is a reference level, not a series -> neutral, dashed, labelled in place.
    ax.axhline(1.0, color=C.BASELINE, lw=1.5, ls="--", zorder=1)
    ax.text(centers[ok][-1], 1.02, "sill (no correlation)", ha="right", va="bottom",
            color=C.INK_MUTED, fontsize=8)

    ax.plot(centers[ok], gamma[ok] / sill, color=C.SERIES[0], marker="o", ms=4, zorder=3)
    ax.text(centers[ok][0], gamma[ok][0] / sill - 0.06, "all tile pairs",
            color=C.SERIES[0], fontsize=8.5, va="top", fontweight="bold")

    for key, colour, label in (("range_50pct_sill_m", C.INK_MUTED, "50% of sill"),
                               ("range_95pct_sill_m", C.STATUS["critical"], "95% of sill")):
        r = meta.get(key)
        if r:
            ax.axvline(r, color=colour, lw=1.2, ls=":", zorder=2)
            ax.text(r, 0.04, f" {label}\n {r:,.0f} m", color=colour, fontsize=8,
                    va="bottom", ha="left")

    ax.axvline(C.TILE_GROUND_M, color=C.INK_MUTED, lw=1, ls="-", alpha=0.5)
    ax.text(C.TILE_GROUND_M, 1.12, " one tile\n 100 m", color=C.INK_MUTED, fontsize=8, va="top")

    ax.set_xscale("log")
    ax.set_ylabel("semivariance / sill")
    ax.set_ylim(0, 1.25)
    ax.set_title("Tiles stay correlated for kilometres, so a random split cannot be honest")

    ax2.axhline(0, color=C.BASELINE, lw=1.5)
    ax2.plot(centers[ok], moran[ok], color=C.SERIES[1], marker="o", ms=4)
    ax2.text(centers[ok][0], moran[ok][0], "  Moran's I", color=C.SERIES[1],
             fontsize=8.5, va="center", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xlabel("distance between tile centres (m, log scale)")
    ax2.set_ylabel("Moran's I")
    fig.tight_layout()
    return C.savefig(fig, "01_semivariogram", thesis)


def fig_blocking_strata(plt, thesis: bool):
    """Route versus parent blocking: the Karasiak argument on this dataset."""
    p = C.TABLES / "spatial_dependence_curves.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    centers = z["centers"]
    sill = float(z["sill"][0])
    strata = [("same_parent", "same parent orthophoto"),
              ("same_route_diff_parent", "same route, different parent"),
              ("diff_route", "different route")]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(1.0, color=C.BASELINE, lw=1.5, ls="--")
    ax.text(centers[-1], 1.02, "sill (independent)", ha="right", va="bottom",
            color=C.INK_MUTED, fontsize=8)

    for (key, label), colour in zip(strata, C.SERIES_3):
        g, c = z[f"gamma_{key}"], z[f"count_{key}"]
        ok = np.isfinite(g) & (c > 0) & (centers <= 120_000)
        if not ok.any():
            continue
        ax.plot(centers[ok], g[ok] / sill, color=colour, marker="o", ms=4)
        ax.text(centers[ok][-1], g[ok][-1] / sill, f"  {label}", color=colour,
                fontsize=8.5, va="center", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlabel("distance between tile centres (m, log scale)")
    ax.set_ylabel("semivariance / sill")
    ax.set_ylim(0, 1.3)
    ax.set_xlim(right=centers[centers <= 120_000][-1] * 6)
    ax.set_title("Only whole-route separation reaches independence")
    fig.tight_layout()
    return C.savefig(fig, "02_blocking_strata", thesis)


def fig_split_leakage(plt, thesis: bool):
    """Sub-1 in one figure: how close the nearest training tile is under each split."""
    p = C.TABLES / "spatial_dependence_curves.npz"
    if not p.is_file():
        return None
    z = np.load(p, allow_pickle=False)
    if "nearest_naive" not in z:
        return None
    naive = z["nearest_naive"]
    blocked = np.concatenate([z[f"nearest_fold_{f}"] for f in range(C.NFOLDS)
                              if f"nearest_fold_{f}" in z])

    # The two sets differ in size by a factor of ~770, so raw counts are not comparable. Each is
    # drawn as a share of ITS OWN tiles.
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bins = np.geomspace(1, max(blocked.max(), naive.max()) * 1.1, 40)
    for data, colour, label in ((naive, C.SERIES[0], f"naive 25-tile split (n={len(naive)})"),
                                (blocked, C.SERIES[1],
                                 f"route-blocked folds (n={len(blocked):,})")):
        w = np.full(len(data), 100.0 / len(data))
        ax.hist(data, bins=bins, weights=w, color=colour, alpha=0.62, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("distance from a held-out tile to its nearest TRAINING tile (m, log scale)")
    ax.set_ylabel("share of that split's held-out tiles (%)")
    ax.set_title("Under the naive split the answer is almost always next door")
    ax.axvline(C.TILE_GROUND_M, color=C.INK_MUTED, lw=1, ls=":")
    ax.text(C.TILE_GROUND_M, ax.get_ylim()[1] * 0.98, " one tile width ", rotation=90,
            va="top", ha="right", color=C.INK_MUTED, fontsize=8)
    ax.legend(loc="upper right", fontsize=8.5)

    near = 150.0
    n_naive = 100.0 * (naive <= near).mean()
    n_block = 100.0 * (blocked <= near).mean()
    ax.text(0.015, 0.96,
            f"within {near:.0f} m of a training tile\n"
            f"  naive        {n_naive:5.1f}%\n"
            f"  route-blocked{n_block:5.1f}%\n"
            f"median  naive {np.median(naive):,.0f} m   "
            f"blocked {np.median(blocked):,.0f} m",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            family="monospace", color=C.INK_SECONDARY)
    fig.tight_layout()
    return C.savefig(fig, "03_split_leakage", thesis)


def fig_bounds(plt, thesis: bool):
    """The ladder that makes 0.24-0.36 Macro-IoU readable."""
    p = C.TABLES / "performance_bounds.json"
    d = _load_json(p)
    if not d:
        return None
    b = d["bounds"]
    cells = d.get("measured_cells", [])
    rows = [("constant majority\n(predict ubefestet everywhere)",
             b["B1_constant_majority"]["macro_iou"], b["B1_constant_majority"]["overall_accuracy"]),
            ("prior-matched random",
             b["B2_prior_matched_random"]["macro_iou"],
             b["B2_prior_matched_random"]["overall_accuracy"])]
    if cells:
        w, bst = cells[-1], cells[0]
        rows += [("worst trained cell", w["macro_iou"], w["overall_accuracy"]),
                 ("best trained cell", bst["macro_iou"], bst["overall_accuracy"])]
    rows.append(("per-tile majority ORACLE\n(uses the labels)",
                 b["B3_per_tile_majority_oracle"]["macro_iou"],
                 b["B3_per_tile_majority_oracle"]["overall_accuracy"]))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, idx, title, xmax in ((axes[0], 1, "Macro-IoU separates them", 0.55),
                                 (axes[1], 2, "Overall accuracy does not", 1.0)):
        y = np.arange(len(rows))
        vals = [r[idx] for r in rows]
        colours = [C.INK_MUTED if "ORACLE" in r[0] else C.SERIES[0] for r in rows]
        ax.barh(y, vals, color=colours, height=0.6)
        ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
        ax.set_xlim(0, xmax)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        for yi, v in zip(y, vals):
            ax.text(v + xmax * 0.015, yi, f"{v:.3f}", va="center", fontsize=8.5,
                    color=C.INK_SECONDARY, fontweight="bold")
    fig.suptitle("A predictor that does nothing already scores 0.87 accuracy",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return C.savefig(fig, "04_performance_bounds", thesis)


def fig_class_distribution(plt, thesis: bool):
    """Imbalance at pixel and tile level. Identity on the axis, one hue."""
    audit = C.load_class_pixel_audit()
    names = [C.CODES[c] for c in C.PREDICTED]
    px = np.array([audit[n]["pixel_count"] for n in names], dtype=float)
    tiles = np.array([audit[n]["tiles_present"] for n in names], dtype=float)
    order = np.argsort(-px)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, vals, title, xlabel in (
            (axes[0], px[order], "Pixels per class", "label pixels (log scale)"),
            (axes[1], tiles[order], "Tiles containing the class",
             f"tiles of {C.EXPECTED_TILES:,} (log scale)")):
        y = np.arange(len(order))
        ax.barh(y, vals, color=C.SERIES[0], height=0.62)
        ax.set_yticks(y, [names[i] for i in order], fontsize=8.5)
        ax.set_xscale("log")
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        for yi, v in zip(y, vals):
            ax.text(v * 1.15, yi, f"{v:,.0f}", va="center", fontsize=8, color=C.INK_SECONDARY)
    ratio = px.max() / px.min()
    fig.suptitle(f"Severe imbalance: {ratio:,.0f}:1 between the commonest and rarest class "
                 f"by pixels", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return C.savefig(fig, "05_class_distribution", thesis)


def fig_channel_normalisation(plt, thesis: bool):
    """What each channel's variance looks like after the configured normalisation."""
    d = _load_json(C.TABLES / "channel_stats.json")
    if not d or "normalisation_audit" not in d:
        return None
    rows = d["normalisation_audit"]
    names = [r["channel"] for r in rows]
    eff = [r["effective_std_div255"] for r in rows]
    is_elev = [n in ("DSM", "DTM") for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax = axes[0]
    y = np.arange(len(names))
    colours = [C.STATUS["critical"] if e else C.SERIES[0] for e in is_elev]
    ax.barh(y, eff, color=colours, height=0.62)
    ax.set_yticks(y, names, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("standard deviation presented to the network")
    ax.set_title("Effective variance after normalisation")
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, eff):
        ax.text(v + max(eff) * 0.015, yi, f"{v:.3f}", va="center", fontsize=8,
                color=C.INK_SECONDARY)
    ax.text(0.98, 0.04, "elevation bands in red", transform=ax.transAxes, ha="right",
            fontsize=8, color=C.STATUS["critical"], fontweight="bold")

    ax = axes[1]
    dec = d.get("elevation_variance_decomposition", {})
    if dec:
        keys = [k for k in ("DSM", "DTM", "nDSM") if k in dec]
        x = np.arange(len(keys))
        between = [dec[k]["between_var"] for k in keys]
        within = [dec[k]["within_var"] for k in keys]
        tot = [b + w for b, w in zip(between, within)]
        ax.bar(x, [100 * b / t for b, t in zip(between, tot)], color=C.SERIES[1],
               width=0.55, label="between tiles (terrain)")
        ax.bar(x, [100 * w / t for w, t in zip(within, tot)], width=0.55,
               bottom=[100 * b / t for b, t in zip(between, tot)],
               color=C.SERIES[2], label="within a tile (structure)")
        ax.set_xticks(x, keys)
        ax.set_ylabel("share of variance (%)")
        ax.set_title("Where the elevation signal lives")
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
        for xi, k in zip(x, keys):
            share = 100 * dec[k]["between_var"] / (dec[k]["between_var"] + dec[k]["within_var"])
            ax.text(xi, share / 2, f"{share:.0f}%", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
            ax.text(xi, share + (100 - share) / 2, f"{100 - share:.0f}%", ha="center",
                    va="center", color="white", fontsize=9, fontweight="bold")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=8)
    else:
        _note(ax, "elevation decomposition unavailable")
    fig.tight_layout()
    return C.savefig(fig, "06_channel_normalisation", thesis)


def fig_ndsm(plt, thesis: bool):
    """Object height by class: the signal the pipeline discards."""
    d = _load_json(C.TABLES / "channel_stats.json")
    if not d or "ndsm_per_class" not in d:
        return None
    recs = [r for r in d["ndsm_per_class"]
            if r["mean_m"] is not None and r["class"] in [C.CODES[c] for c in C.PREDICTED]]
    if not recs:
        return None
    recs.sort(key=lambda r: r["mean_m"])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(recs))
    ax.barh(y, [r["mean_m"] for r in recs],
            xerr=[r["std_m"] for r in recs], color=C.SERIES[0], height=0.6,
            error_kw={"ecolor": C.INK_MUTED, "elinewidth": 1.2, "capsize": 3})
    ax.set_yticks(y, [r["class"] for r in recs], fontsize=8.5)
    ax.set_xlabel("nDSM = DSM - DTM, height above ground (m); bars show one standard deviation")
    ax.set_title("Height above ground separates the classes, and is never fed to the network")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return C.savefig(fig, "07_ndsm_by_class", thesis)


def fig_footprint(plt, thesis: bool):
    """Where the data actually is. Fold identity is 3 categories -> 3 validated slots."""
    df = C.load_tile_inventory()
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    for f, colour in zip(range(C.NFOLDS), C.SERIES_3):
        m = df["fold"] == f
        ax.scatter(df.loc[m, "centroid_e"] / 1000, df.loc[m, "centroid_n"] / 1000,
                   s=3, c=colour, alpha=0.55, linewidths=0)
    for f, colour in zip(range(C.NFOLDS), C.SERIES_3):
        m = df["fold"] == f
        ax.text(df.loc[m, "centroid_e"].mean() / 1000, df.loc[m, "centroid_n"].max() / 1000 + 4,
                f"fold {f}", color=colour, fontsize=10, fontweight="bold", ha="center")
    ax.set_xlabel("easting (km, EPSG:25832)")
    ax.set_ylabel("northing (km)")
    ax.set_aspect("equal")
    ax.set_title(f"The labelled footprint: {len(df):,} tiles on 16 flight routes, not national")
    fig.tight_layout()
    return C.savefig(fig, "08_footprint_map", thesis)


def fig_route_composition(plt, thesis: bool):
    """Covariate shift between routes, shown as one interpretable measure."""
    p = C.TABLES / "route_composition.csv"
    if not p.is_file():
        return None
    rows = list(csv.DictReader(open(p)))
    rows.sort(key=lambda r: float(r["sealed_fraction"]))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(rows))
    ax.barh(y, [100 * float(r["sealed_fraction"]) for r in rows], color=C.SERIES[0], height=0.62)
    ax.set_yticks(y, [f"{r['route']}  (fold {r['fold']}, {int(r['tiles']):,} tiles)" for r in rows],
                  fontsize=8)
    ax.set_xlabel("sealed surface as % of scored pixels")
    ax.set_title("Routes are not interchangeable: composition ranges from 0% to 68% sealed")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return C.savefig(fig, "09_route_composition", thesis)


def fig_boundary(plt, thesis: bool):
    d = _load_json(C.TABLES / "label_quality.json")
    if not d or "boundary" not in d:
        return None
    b = d["boundary"]
    labels, px, err = b["labels"], np.array(b["pixels"]), np.array(b["errors"])
    keep = px > 0
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    x = np.arange(keep.sum())
    axes[0].bar(x, 100 * err[keep] / px[keep], color=C.SERIES[0], width=0.62)
    axes[0].set_xticks(x, [labels[i] for i in np.flatnonzero(keep)], rotation=45, ha="right",
                       fontsize=8)
    axes[0].set_ylabel("error rate (%)")
    axes[0].set_title("Error rate by distance to a label boundary")
    axes[0].grid(axis="x", visible=False)
    axes[1].bar(x, 100 * err[keep] / err.sum(), color=C.SERIES[0], width=0.62)
    axes[1].set_xticks(x, [labels[i] for i in np.flatnonzero(keep)], rotation=45, ha="right",
                       fontsize=8)
    axes[1].set_ylabel("share of all error (%)")
    axes[1].set_title("Where the error mass sits")
    axes[1].grid(axis="x", visible=False)
    fig.suptitle("Boundary imprecision versus genuine class confusion", fontsize=11,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    return C.savefig(fig, "10_boundary_error", thesis)


def fig_separability(plt, thesis: bool):
    d = _load_json(C.TABLES / "separability.json")
    if not d or "jm_matrix" not in d:
        return None
    names = d["jm_matrix"]["classes"]
    m = np.array(d["jm_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(m, cmap=C.seq_cmap(), vmin=0, vmax=2)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.grid(visible=False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if m[i, j] > 1.2 else C.INK)
    fig.colorbar(im, ax=ax, label="Jeffries-Matusita distance (2 = fully separable)",
                 fraction=0.046)
    ax.set_title("Spectral separability before any model is involved")
    fig.tight_layout()
    return C.savefig(fig, "11_separability", thesis)


def fig_route_spread(plt, thesis: bool):
    p = C.TABLES / "route_cell_metrics.csv"
    if not p.is_file():
        return None
    rows = list(csv.DictReader(open(p)))
    by_cell = {}
    for r in rows:
        v = r["macro_iou_present_classes"]
        if v not in ("", "None", None):
            by_cell.setdefault(r["cell"], []).append(float(v))
    if not by_cell:
        return None
    cells = sorted(by_cell, key=lambda c: -np.median(by_cell[c]))
    fig, ax = plt.subplots(figsize=(7.6, max(4.0, 0.28 * len(cells))))
    for i, c in enumerate(cells):
        v = np.array(by_cell[c])
        ax.plot([v.min(), v.max()], [i, i], color=C.BASELINE, lw=2, solid_capstyle="round")
        ax.scatter(v, np.full(len(v), i), s=14, color=C.SERIES[0], alpha=0.55, linewidths=0)
        ax.scatter([np.median(v)], [i], s=42, color=C.SERIES[1], zorder=3, linewidths=0)
    ax.set_yticks(range(len(cells)), cells, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("route-level Macro-IoU (over the classes present in each route)")
    ax.set_title("Every cell's score is an average over routes that behave very differently")
    ax.grid(axis="y", visible=False)
    ax.text(0.99, 0.02, "orange = median across routes", transform=ax.transAxes, ha="right",
            fontsize=8, color=C.SERIES[1], fontweight="bold")
    fig.tight_layout()
    return C.savefig(fig, "12_route_spread", thesis)


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA figures")
    ap.add_argument("--thesis", action="store_true", help="also write 300 dpi PDFs")
    args = ap.parse_args()

    C.ensure_out_dirs()
    C.apply_style()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C.banner("EDA FIGURES")
    made, skipped = [], []
    for fn in (fig_semivariogram, fig_blocking_strata, fig_split_leakage, fig_bounds,
               fig_class_distribution, fig_channel_normalisation, fig_ndsm, fig_footprint,
               fig_route_composition, fig_boundary, fig_separability, fig_route_spread):
        try:
            out = fn(plt, args.thesis)
        except Exception as exc:                   # noqa: BLE001
            print(f"  {fn.__name__:<28} ERROR {type(exc).__name__}: {exc}")
            skipped.append(fn.__name__)
            continue
        if out is None:
            print(f"  {fn.__name__:<28} skipped (its module has not been run yet)")
            skipped.append(fn.__name__)
        else:
            print(f"  {fn.__name__:<28} -> {out.name}")
            made.append(out)
        plt.close("all")

    print(f"\n{len(made)} figures written to {C.FIGURES}")
    if skipped:
        print(f"{len(skipped)} skipped: {', '.join(skipped)}")
    total = sum(p.stat().st_size for p in C.FIGURES.glob("*")) / 1e6
    print(f"figure directory total: {total:.1f} MB")


if __name__ == "__main__":
    main()
