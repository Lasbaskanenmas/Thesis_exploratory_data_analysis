#!/usr/bin/env python
"""
Part B figure -- paired route differences (Great Plan 3.1 section 5.2, item 6).

Section 5.2 asks for "one statistics table artifact per family + one figure (paired route
differences)". The tables have existed since 25/8; this is the figure, and it is the last piece of
that item.

WHAT IT SHOWS AND WHY THAT SHAPE. One panel per declared Holm family. Within a panel, one row per
declared pair, and within a row one dot per route -- the paired difference d_r = m_A(r) - m_B(r) on
the declared per-route metric. A vertical line at zero, the median difference marked, and the 95 %
paired bootstrap interval drawn through it.

The point of plotting all sixteen dots rather than a summary is that the paired test's evidence IS
the sign pattern across routes. A pair that wins on 15 of 16 routes and a pair that wins on 9 of 16
can share a median; only the scatter distinguishes them, and D2's fragility rule turns on exactly
that. Routes are never comparable to each other (D3), so the x axis is a difference and never a
score.

Reads the locked declaration for the families, and `wilcoxon_by_family.csv` for the statistics, so
the figure cannot drift from the tables it accompanies.

    python partb_figure.py --declaration ..\\2026-08-25_pre_declarations.md
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402
import partb_statistics as PB  # noqa: E402

OUT_DIR = C.EDA_ROOT / "results" / "figures"
SHORT = [("convnext_upernet", "cnx"), ("unet_resnet34", "rn34"),
         ("swin_upernet", "swin"), ("segformer_b1", "segf")]


def s(name):
    for a, b in SHORT:
        name = name.replace(a, b)
    return name


def main():
    ap = argparse.ArgumentParser(description="Part B paired-route-difference figure")
    ap.add_argument("--declaration", required=True)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dec = PB.load_declaration(args.declaration)          # refuses unless LOCKED
    D = PB.normalised(dec)
    scores, tiles = PB.load_route_scores(D["metric"])
    routes = sorted(tiles)

    stats = {}
    with open(C.TABLES / "part_b" / "wilcoxon_by_family.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["route_set"] == "primary":
                stats[(r["cell_a"], r["cell_b"])] = r

    fams = list(D["families"].items())
    heights = [len(p) for _n, p in fams]
    fig, axes = plt.subplots(len(fams), 1, figsize=(11, 1.0 + 0.62 * sum(heights)),
                             gridspec_kw={"height_ratios": heights}, squeeze=False)
    axes = axes[:, 0]

    for ax, (fam, pairs) in zip(axes, fams):
        labels = []
        for i, (a, b) in enumerate(pairs):
            y = len(pairs) - 1 - i
            d = np.array([scores[a][r] - scores[b][r] for r in routes])
            st = stats.get((a, b), {})
            rej = st.get("holm_reject") == "True"
            frag = st.get("fragile") == "True"

            ax.scatter(d, np.full_like(d, y, dtype=float), s=26, zorder=3,
                       facecolor="#4878a8" if rej else "#b0b0b0",
                       edgecolor="#20304a", linewidth=0.5,
                       label=None)
            lo = float(st.get("effect_ci_low", "nan"))
            hi = float(st.get("effect_ci_high", "nan"))
            med = float(st.get("effect_median_difference", "nan"))
            ax.plot([lo, hi], [y - 0.22, y - 0.22], color="#20304a", lw=2.2, zorder=4)
            ax.plot([med], [y - 0.22], marker="|", ms=13, color="#20304a", zorder=5)

            wins = int(st.get("sign_n_positive", 0))
            nprime = st.get("wilcoxon_effective_n", "?")
            tag = f"{s(a)}  -  {s(b)}"
            note = f"{wins}/{nprime}"
            if rej:
                note += "  Holm"
            if frag:
                note += "  FRAGILE"
            labels.append((y, tag, note))

        ax.axvline(0, color="#c04040", lw=1.0, zorder=1)
        ax.set_yticks([y for y, _t, _n in labels])
        ax.set_yticklabels([t for _y, t, _n in labels], fontsize=8.5, family="monospace")
        for y, _t, note in labels:
            ax.text(1.005, y, note, transform=ax.get_yaxis_transform(), va="center",
                    fontsize=7.5, family="monospace", color="#20304a")
        ax.set_ylim(-0.7, len(pairs) - 0.3)
        ax.set_title(fam, fontsize=9, loc="left", pad=4)
        ax.grid(axis="x", color="#e8e8e8", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)

    axes[-1].set_xlabel(f"paired route difference in {D['metric']}   "
                        f"(one dot per route, n = {len(routes)}; bar = median with 95 % paired "
                        f"whole-route bootstrap CI)", fontsize=8.5)
    fig.suptitle("Part B: paired route differences per declared pair", fontsize=11, x=0.01,
                 ha="left")
    fig.text(0.01, 0.005,
             "Filled dots: pair rejects under Holm within its family. Right margin: sign-test wins "
             "/ effective N. Differences only — route scores are not comparable between routes "
             "(declaration D3).", fontsize=7.2, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 0.90, 0.97))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = C.assert_writes_are_local(OUT_DIR / f"13_partb_paired_route_differences.{ext}")
        fig.savefig(p, dpi=args.dpi)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
