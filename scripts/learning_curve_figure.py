#!/usr/bin/env python
"""
The learning-curve figure — Great Plan 3.1 §2's "exhibit".

§2 calls the per-class panel "one of the thesis's central figures" and the data has existed since
25/8 in `learning_curve/learning_curve_per_class.csv`, but no figure was ever made. This is it.

TWO PANELS, because §2's answer to Mads's questions 1 and 2 is deliberately two-part and both halves
have to be visible at once:

  (a) the AGGREGATE does not plateau. Macro-IoU rises at a near-constant ~0.0006-0.0007 per
      percentage point of training pool across all three intervals, so at 100 % it is still
      data-limited.

  (b) decomposed PER CLASS, that rise is not spread evenly. brosten and green_roof sit at exactly
      0.0000 at every volume including the 100 % anchor, while grus steps sharply on the final
      interval and betonflade rises monotonically off zero. The two classes volume cannot touch are
      exactly the two whose ceiling the other evidence already explains.

The two flat-zero classes are drawn heavy and dark; everything else is muted, because the figure
exists to make that contrast legible rather than to let nine lines compete.

A caution belongs in the caption, not on the axes: curve accuracies must NOT be read against the
pooled trivial floor of 0.8747. Fold 0's held-out composition gives a fold-specific
constant-majority floor of ~0.754. Every point is scored on a byte-identical held-out set, so
within-curve comparisons are clean; cross-cell comparisons are not.

    python learning_curve_figure.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eda_common as C  # noqa: E402

SRC = C.TABLES / "learning_curve" / "learning_curve_per_class.csv"
OUT = C.EDA_ROOT / "results" / "figures"
FLAT_ZERO = {"brosten", "green_roof"}          # the classes volume never moves
FOLD0_TRIVIAL = 0.754                          # fold-0 constant-majority floor, NOT the pooled 0.8747


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(SRC, newline="", encoding="utf-8")))
    agg = defaultdict(dict)
    per = defaultdict(dict)
    for r in rows:
        x = float(r["train_pct_of_fold0_pool"])
        if r["scope"] == "aggregate":
            agg[r["metric"]][x] = float(r["value"])
        else:
            per[r["class"]][x] = float(r["value"])
    xs = sorted(agg["macro_iou"])
    tiles = {float(r["train_pct_of_fold0_pool"]): int(r["train_tiles"]) for r in rows}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.2, 9.0),
                                   gridspec_kw={"height_ratios": [1, 1.45]})

    # ---- (a) aggregate ----
    ax1.plot(xs, [agg["overall_accuracy"][x] for x in xs], marker="o", color="#888888",
             lw=1.8, label="overall accuracy")
    ax1.plot(xs, [agg["macro_iou"][x] for x in xs], marker="o", color="#20304a",
             lw=2.4, label="Macro-IoU")
    ax1.axhline(FOLD0_TRIVIAL, color="#c04040", ls="--", lw=1.2)
    ax1.text(25.6, FOLD0_TRIVIAL + 0.008,
             "fold-0 constant-majority floor ≈ 0.754  (not the pooled 0.8747)",
             fontsize=7.4, color="#c04040")
    for x in xs:
        # First point sits on the left spine; centring it would clip the leading digit.
        ax1.annotate(f"{agg['macro_iou'][x]:.4f}", (x, agg["macro_iou"][x]),
                     textcoords="offset points", xytext=(4 if x == xs[0] else 0, -14),
                     ha="left" if x == xs[0] else "center", fontsize=7.5, color="#20304a")
    ax1.set_ylim(0.20, 1.0)
    ax1.set_ylabel("score")
    ax1.set_title("(a) the aggregate does not plateau — still data-limited at 100 %",
                  fontsize=9.5, loc="left")
    ax1.legend(fontsize=8, frameon=False, loc="center right")
    ax1.grid(color="#ececec")
    ax1.set_axisbelow(True)

    # ---- (b) per class ----
    order = sorted(per, key=lambda c: -per[c][xs[-1]])
    # Nudge labels apart where end-points nearly coincide, and give the two flat-zero classes a
    # single shared annotation instead of two that would print on top of each other at y = 0.
    placed = []
    for cl in order:
        ys = [per[cl][x] for x in xs]
        flat = cl in FLAT_ZERO
        ax2.plot(xs, ys, marker="o" if flat else ".", ms=7 if flat else 6,
                 lw=3.0 if flat else 1.4,
                 color="#c04040" if flat else "#9aa7b8",
                 zorder=5 if flat else 2)
        if flat:
            continue
        y = ys[-1]
        while any(abs(y - p) < 0.035 for p in placed):
            y += 0.035
        placed.append(y)
        ax2.annotate(f"  {cl}", (xs[-1], y), fontsize=8, color="#54637a", va="center")
    ax2.annotate("  brosten & green_roof\n  0.0000 at every volume",
                 (xs[-1], 0.0), fontsize=8, color="#c04040", fontweight="bold", va="center")
    ax2.set_ylim(-0.06, 1.0)
    ax2.set_xlabel("training pool, % of the fold-0 pool   "
                   f"({tiles[xs[0]]:,} → {tiles[xs[-1]]:,} tiles)")
    ax2.set_ylabel("per-class IoU")
    ax2.set_title("(b) the rise does not reach the two flagship weak classes", fontsize=9.5,
                  loc="left")
    ax2.grid(color="#ececec")
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.0f}" if x != 71.8 else "71.8" for x in xs])
        ax.set_xlim(22, 150)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Learning curve — resnet34+UNet, rgb, weighted CE, held-out fold 0",
                 fontsize=11, x=0.01, ha="left")
    fig.text(0.01, 0.012,
             "Nested whole-route subsets; every point scored on a byte-identical held-out set, so\n"
             "within-curve comparisons are clean and cross-cell ones are not. Single fold, one "
             "seed,\ndescriptive — no inferential claim is attached.",
             fontsize=7.2, color="#555555", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.062, 1, 0.965))

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = C.assert_writes_are_local(OUT / f"14_learning_curve_per_class.{ext}")
        fig.savefig(p, dpi=150)
        print(f"wrote {p}")

    print("\nper-class IoU at each point:")
    print(f"  {'class':<12}" + "".join(f"{x:>9.1f}%" for x in xs))
    for cl in order:
        print(f"  {cl:<12}" + "".join(f"{per[cl][x]:>10.4f}" for x in xs))


if __name__ == "__main__":
    main()
