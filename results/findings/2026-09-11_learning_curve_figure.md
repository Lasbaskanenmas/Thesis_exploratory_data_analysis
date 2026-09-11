# 2026-09-11 — Figure 14: the learning curve, per class

**Status:** built. `results/figures/14_learning_curve_per_class.{png,pdf}`, from
`exploratory_data_analysis/scripts/learning_curve_figure.py`.

## Why this note exists

Great Plan 3.1 §2 calls the per-class learning-curve panel "the exhibit" and "one of the thesis's
central figures". The underlying table
(`results/tables/learning_curve/learning_curve_per_class.csv`) has existed since 25/8, but no
figure was ever rendered from it — the figure sequence on disk ran 01–09, 11, 12, 13 with a gap at
10 and nothing for the learning curve. This closes that gap. Nothing was recomputed; the figure
reads the existing CSV.

## What it shows

Two panels, because §2's answer to Mads's questions 1 and 2 is deliberately two-part.

**(a) The aggregate does not plateau.** Macro-IoU rises 0.2806 → 0.2979 → 0.3110 → 0.3292 across
25 % → 50.1 % → 71.8 % → 100 % of the fold-0 training pool (3,222 → 12,875 tiles). The slope is
near-constant at ~0.0006–0.0007 per percentage point across all three intervals — there is no
knee. At 100 % of the data available, the model is still data-limited.

**(b) That rise is not spread evenly across classes.**

| class | 25 % | 50.1 % | 71.8 % | 100 % |
|---|---|---|---|---|
| ubefestet | 0.9295 | 0.9435 | 0.9460 | 0.9465 |
| solceller | 0.7612 | 0.8603 | 0.8876 | 0.9010 |
| asfalt | 0.5166 | 0.5354 | 0.5507 | 0.5568 |
| grus | 0.1615 | 0.1953 | 0.1950 | 0.2960 |
| fliser | 0.1284 | 0.1194 | 0.1553 | 0.1596 |
| drivhus | 0.0284 | 0.0255 | 0.0542 | 0.0674 |
| betonflade | 0.0000 | 0.0019 | 0.0106 | 0.0354 |
| **green_roof** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| **brosten** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |

`brosten` and `green_roof` are at exactly 0.0000 at every volume including the 100 % anchor. They
are drawn heavy red with a single shared annotation; the other seven are muted, because the figure
exists to make that contrast legible rather than to let nine lines compete.

`betonflade` is the informative contrast: it starts at a hard 0.0000 and rises monotonically off
it (0.0000 → 0.0019 → 0.0106 → 0.0354). So a zero at 25 % is not itself evidence of an
insurmountable class — which is what makes the two flat zeros mean something.

## The caution that must travel with the figure

The curve's accuracies must **not** be read against the pooled constant-majority floor of 0.8747.
These are fold-0-only numbers, and fold 0's held-out composition gives a fold-specific
constant-majority floor of ≈0.754. That line is drawn on panel (a) and labelled as such.

Every point is scored on a byte-identical held-out set, so **within-curve** comparisons are clean.
**Cross-cell** comparison against the 37-cell matrix is not, and none is made.

Single fold, one seed. Descriptive — outside every declared family in
`2026-08-25_pre_declarations.md`, and no inferential claim is attached to it.

## Provenance

Nested whole-route subsets: each smaller pool is a strict subset of the next, drawn at route
granularity so no route is split across the subset boundary. resnet34+UNet, rgb, weighted CE,
held-out fold 0.
