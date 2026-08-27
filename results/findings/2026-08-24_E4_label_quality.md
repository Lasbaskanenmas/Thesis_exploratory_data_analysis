# 2026-08-24 — E4, label quality: the 37 % ignore mass is coverage, not uncertainty

**Role.** Records the first real run of `eda_label_quality.py` (module E4), written and self-tested
on 12/8 and never executed until tonight. Closes gap G0.4 and supplies the artifact the SQ4
limitation text has been citing without one, per the 2026-08-23 post-scriptum. Every number is read
from `results/tables/label_quality.json` and `results/tables/boundary_error_profile.csv`.

## 1. Where the ignore mass comes from — the headline

Scanned all **799 parent label rasters**, 54,106,685,500 pixels, before tiling.

| value | pixels | share | meaning |
|---|---:|---:|---|
| 15 | 39,012,182,311 | 72.10 % | nodata sentinel, remapped to class 0 by the reclass step |
| 0 | 2,401,273,039 | 4.44 % | explicit `unknown` in the annotation |
| 1–14 | 12,693,230,150 | 23.46 % | annotated surface classes |

**Of everything that ends up as ignore, 94.20 % is the nodata sentinel and 5.80 % is explicit
unknown.** The 36.861 % ignore share of the tiled pool is therefore **annotation coverage, not
annotator uncertainty**. Annotators delineated selected polygons; everything outside them carries no
ground truth at all and is dropped from every metric silently. The effective supervised dataset is
roughly **12.2 of 19.3 billion pixels**.

This is a different sentence from the one the thesis was going to have to write. It is not label
noise to be cleaned or apologised for. It is the share of the imagery that was never labelled, and
it belongs in the dataset chapter as a property of the product.

**Independent cross-check.** The annotation GeoPackage holds **17,791 polygons** covering
**56.43 km²** against a 193.14 km² tile footprint, a **29.21 %** coverage ratio. The tile footprint
over-counts because tiles overlap (75.45 km² of unique ground, per the 2026-08-04 provenance
report), so the two figures are consistent in direction and magnitude and were computed from
independent sources.

## 2. The ignore mask is severely biased by geography — new, and it matters for Part B

| route | fold | tiles | unknown % | scored px |
|---|---|---:|---:|---:|
| 82-11 | 1 | 1,441 | 11.80 % | 1,271,017,403 |
| 82-13 | 1 | 1,043 | 17.16 % | 863,989,206 |
| 82-16 | 2 | 378 | 18.83 % | 306,829,357 |
| 82-19 | 2 | 591 | 26.70 % | 433,222,749 |
| 82-20 | 2 | 608 | 47.96 % | 316,377,560 |
| 82-21 | 1 | 313 | 37.79 % | 194,724,135 |
| 82-22 | 2 | 1,360 | 25.43 % | 1,014,086,946 |
| 82-24 | 2 | 1,529 | 57.95 % | 642,911,577 |
| 83-25 | 1 | 3,630 | 20.45 % | 2,887,492,462 |
| 83-26 | 2 | 1,228 | 22.96 % | 946,025,258 |
| 83-31 | 0 | 1,020 | 17.96 % | 836,776,918 |
| 83-34 | 1 | 10 | 89.33 % | 1,067,196 |
| 84-40 | 0 | 5,269 | 57.75 % | 2,226,168,062 |
| 84-41 | 2 | 343 | 47.18 % | 181,155,748 |
| 85-45 | 2 | 401 | 87.30 % | 50,907,628 |
| 85-48 | 0 | 150 | 85.41 % | 21,881,576 |

**Range 11.80 % to 89.33 %, a spread of 77.5 points. Per fold: fold 0 52.09 %, fold 1 18.93 %,
fold 2 39.55 %.**

Two consequences, both of which belong in the statistical-limitations subsection (Plan 3.1 §5.2) and
neither of which was on the list before tonight.

1. **Routes carry wildly unequal amounts of supervision, on top of the class-diversity problem
   already recorded.** Route 83-34 contributes 1.07 million scored pixels; route 83-25 contributes
   2.89 billion, a factor of 2,700. The route-level Wilcoxon weights every route equally by design,
   which is defensible — the route is the independent unit — but the declaration should say plainly
   that an equally weighted route test is not an equally informed one. This sharpens rather than
   changes the D2 recommendation: the sensitivity run dropping routes under 100 tiles removes 83-34,
   which is both the smallest and the most sparsely labelled route in the split.
2. **The folds differ in supervision density by a factor of nearly three.** Fold 0 is 52.09 %
   unlabelled against fold 1's 18.93 %. Per-fold macro-IoU values were already flagged as diagnostic
   only; this is the mechanism behind part of that spread and should be cited when the per-fold
   table is presented. It also bears on the learning curve, which is scored entirely on fold 0.

## 3. Boundary ambiguity — the SQ4 annotation-precision proxy now has a number

600 tiles, fixed seed, cell `convnext_upernet_rgb`. 370,151,418 scored pixels, 23,565,209 errors,
overall error rate **6.37 %**.

| distance to nearest label boundary | pixels | % of px | % of error | error rate |
|---|---:|---:|---:|---:|
| 0 (on boundary) | 2,190,173 | 0.59 % | 3.92 % | **42.17 %** |
| [1,2) | 2,907,920 | 0.79 % | 4.98 % | 40.34 % |
| [2,3) | 2,872,664 | 0.78 % | 4.67 % | 38.31 % |
| [3,5) | 4,211,273 | 1.14 % | 6.37 % | 35.67 % |
| [5,10) | 10,822,307 | 2.92 % | 14.25 % | 31.03 % |
| [10,20) | 15,672,765 | 4.23 % | 15.89 % | 23.90 % |
| [20,50) | 27,824,913 | 7.52 % | 18.68 % | 15.82 % |
| ≥50 | 303,649,403 | 82.03 % | 31.23 % | **2.42 %** |

The error rate falls monotonically with distance from an annotation boundary, from 42.17 % on the
boundary to 2.42 % more than 5 m away — a factor of **17**. **8.9 % of all error sits within 2 px
(20 cm) of a boundary, on 1.4 % of the pixels.**

**Read it in both directions, because both readings are load-bearing and they point opposite ways.**

- Error concentrated on boundaries at 42 % is as much annotation imprecision as model failure. At
  0.1 m GSD, 20 cm is within the plausible precision of a hand-drawn polygon. This is the available
  proxy for annotation precision that the post-scriptum's SQ4 limitation names, and it now exists
  as an artifact rather than a promise. It caps what any model can score.
- **But 31.23 % of all error sits more than 5 m from any boundary**, on 82 % of the pixels, at a
  2.42 % rate. That error cannot be explained by edge imprecision at all. It is genuine class
  confusion — which is the F5b reading, and it means the weak-category argument does *not* reduce to
  sloppy annotation. The two mechanisms are separable and both are present.

Boundary IoU (Plan 3.1 §5.4) is the metric that puts a number on the first mechanism per class; this
profile is the pooled version and the two should be presented together.

## 4. Selection, and the limitation that stands

60,123 label tiles exist on disk; **19,314 are in the pool (32.1 %)**; 757 of 799 parents are
represented. The excluded tiles are the all-zero-label and image-less ones already recorded in Plan
3.1 §5.6.5's first delimitation, plus the four corrupt tiles commented out of `all.txt`
(`O2021_82_13_1_0024_00141735_7000_0`, `O2021_82_13_1_0024_00141802_8150_2000`,
`O2021_82_13_1_0025_00144229_8160_1000`, `O2021_82_22_1_0031_00000390_8000_1000`). The pool is a
selected subset and any national-behaviour statement inherits that selection on top of the 16-route
footprint.

**Annotator churn remains unmeasurable, and E4 confirms it rather than assuming it.**
`labels/old_splitted_labels/` holds 16 tiles on the stride-960 grid, only one sharing a filename
with the current pool, and `labels/large_label/reclass/` was verified by cross-tabulation to be the
same rasters with sentinel 15 remapped to 0 rather than an older vintage. So the contribution of
`data_cleaning_based_on_newer_ground_truth.py` to the ignore mass cannot be estimated here and no
annotator-disagreement number is reported. This is the same confirmed absence the author established
with KDS on 24/8, now also established from disk, and §3 above is the declared proxy.

## Artifacts

- `results/tables/label_quality.json` — sections 1, 2 and 4, plus the boundary summary.
- `results/tables/boundary_error_profile.csv` — the §3 table.
- `exploratory_data_analysis/scripts/eda_label_quality.py` — regenerates both (`--selftest` passes).
