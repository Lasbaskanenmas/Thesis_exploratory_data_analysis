# SQ1 findings — the Befæstelsesdata task as a statistical-learning problem

**Status: live document. Generated from the artifacts in `results/tables/`, which are themselves
reproducible from `scripts/` against untouched source data.**

> **SQ1** — What are the properties of the Befæstelsesdata classification task as a
> statistical-learning problem, and what do these properties imply for which learning and
> evaluation methods are appropriate, and what performance is attainable?

Every number below is measured on the 19,314-tile labelled pool. The tile inventory that
underpins all of it reproduces the two pre-existing audits **exactly** — every per-class pixel
count, every per-class tile count, every per-route total, and the frozen fold sizes
6,439 / 6,437 / 6,438. Nothing here rests on a recomputation that disagrees with what was already
on disk.

---

## F1. A predictor that does nothing scores 0.87 accuracy

Constant "always answer ubefestet", computed through the same metric code as the real results:

| predictor | Macro-IoU | macro-F1 | overall accuracy |
|---|---:|---:|---:|
| constant majority | 0.0972 | 0.1037 | **0.8747** |
| prior-matched random | 0.0935 | 0.1111 | 0.7705 |
| best trained cell (`convnext_upernet_rgb`) | 0.3586 | 0.4593 | 0.9336 |
| production architecture (`unet_resnet34_rgb`) | 0.2695 | 0.3454 | 0.8973 |

The whole modelling effort is worth **5.9 percentage points of accuracy** over predicting one class
everywhere. On Macro-IoU the same comparison is a **3.7x** gap.

The binary befæstet/ubefæstet task that KDS's ~95 percent headline refers to has the same trivial
floor of 0.8747, so that headline is roughly **7.5 points above doing nothing**.

**Implication for method.** Accuracy conceals the model's contribution on this data; Macro-IoU
exposes it. That is the argument for per-category metrics established without appealing to any
literature, which is what Great Plan 3.0 §2 asks for.

## F2. A tile-level oracle beats every trained model on Macro-IoU

Painting each tile entirely with its own dominant class — an oracle, since it reads the labels —
scores **Macro-IoU 0.3993**, above the best trained cell's 0.3586.

| class | tile-oracle IoU | best cell IoU |
|---|---:|---:|
| betonflade | 0.6519 | 0.1526 |
| brosten | 0.3670 | 0.0809 |
| grus | 0.3017 | 0.2159 |
| asfalt | 0.4586 | 0.5755 |
| solceller | 0.5799 | 0.7525 |
| drivhus | 0.0000 | 0.2414 |

This is not a claim that the models are useless. It is a claim about **where** they add value.
Where the oracle wins (betonflade, brosten) the class is concentrated in tiles it dominates, and the
models still fail — so the failure is confusion, not dispersion. Where the models win (drivhus,
solceller) they are doing genuine sub-tile localisation: drivhus is never any tile's plurality
class, yet the models score 0.24 on it.

**This reframes finding F5b.** brosten is not failing because it is scattered or rare. Tiles
containing brosten are substantially brosten, and the models still miss it.

## F3. Tiles stay correlated for kilometres, so a random split cannot be honest

Semivariogram of tile composition (sealed fraction), over all 183,984,153 tile pairs:

- **Adjacent tiles (100 m apart) share about 90 percent of their variance** — γ/sill = 0.100,
  Moran's I = 1.13.
- Practical range at **50 percent** of the sill: **650 m**.
- Practical range at **95 percent** of the sill: **4,232 m** — about 42 tile widths.

A random tile-level split therefore puts near-duplicates on both sides of the train/test boundary.
This is the empirical warrant for spatial cross-validation, measured on KDS's own data rather than
cited from Roberts, Karasiak or Kattenborn.

## F4. Route blocking is required; parent blocking would not have been enough

Mean semivariance by spatial relationship, as a fraction of the sill. Below 1.0 means still
correlated:

| pairs are... | tile pairs | γ / sill |
|---|---:|---:|
| in the same parent orthophoto | 435,001 | **0.297** |
| in the same route, different parent | 25,258,787 | **0.868** |
| in different routes | 158,290,365 | **1.023** |

Only whole-route separation reaches independence. Blocking on the parent orthophoto — the unit Plan
2.0 originally specified — would have left same-route pairs at 0.868 of the sill straddling the
split. **The route-over-parent decision, previously justified by citing Karasiak et al. 2022, is
now demonstrated on this dataset.**

## F5. The naive 25-tile split measures memorisation

Distance from each held-out tile to its nearest *training* tile:

| split | median | within 150 m of a training tile |
|---|---:|---:|
| naive 25-tile valid set | **119 m** | **88.0 percent** |
| route-blocked folds | **22,417 m** | 6.9 percent |

At 119 m the semivariogram says those tiles share about 90 percent of their variance with training
data. The naive split is not merely optimistic; it largely measures recognition of places the model
has already seen.

**Stated honestly:** route blocking is not perfect either. 6.9 percent of route-blocked held-out
tiles still sit within 150 m of a training tile, where routes abut. That is the weak residual
between-route correlation Karasiak warns about, now quantified rather than asserted, and it belongs
in the statistical limitations subsection.

## F6. The routes are not interchangeable

Sealed-surface share ranges from **0.00 percent** (route 83-31, entirely unsealed) to **67.6
percent** (route 83-34). Route 85-45 is 53 percent solceller — effectively a solar-farm route.

The three folds inherit this. It is the mechanism behind the large per-fold variance already
visible in the existing artifacts, for example solceller per-fold IoU of 0.769 / 0.104 / 0.749:
fold 1 simply contains almost no solceller.

**Implication.** A pooled headline is an average over places that behave very differently, and
route-level pairing is the right unit for the formal test.

## F7. Most tiles hold one class, and 37 percent of all label pixels are unscored

- **59.3 percent** of tiles contain exactly one of the nine predicted classes; **0.7 percent**
  contain none. Only 4.6 percent contain five or more.
- **36.861 percent** of every label pixel is `unknown`, the ignore index, and is dropped silently
  from every metric. The effective supervised dataset is **12.19 of 19.31 billion pixels**.

Local homogeneity is why the tile-level oracle does so well (F2), and it constrains how much
within-tile context there is for a model to exploit.

## F8. The imagery is fine; the elevation channels are configured into near-uselessness

Measured over all 19,314 tiles and all 14 available bands.

**A correction to an earlier estimate.** A planning-stage figure of "242x" came from a single flat
tile's *within-tile* standard deviation. The full-pool number is smaller and the mechanism is more
specific — see below. The direction of the finding stands; the magnitude and the reason are now
measured rather than extrapolated.

Effective standard deviation each channel presents to the network after `(x/255 - mean)/std`:

| channel | measured mean | measured std | config mean | config std | effective std |
|---|---:|---:|---:|---:|---:|
| rgb_R | 100.28 | 50.68 | 0.485 | 0.229 | 0.868 |
| cir NIR | 111.67 | 62.71 | 0.408 | 0.152 | 1.620 |
| OrtoRGB_R | 115.59 | 53.34 | 0.485 | 0.229 | 0.913 |
| **DSM** | 21.74 | 23.71 | **0.5** | **1.0** | **0.093** |
| **DTM** | 18.19 | 21.81 | **0.5** | **1.0** | **0.086** |

A **19x spread** between the weakest and strongest channel, and the ranking is identical whether or
not the `/255` step is applied, so the conclusion does not depend on resolving where fastai's
`IntToFloatTensor` acts.

The DSM and DTM configuration constants (`mean 0.5, std 1.0`) are placeholders, not measured
statistics, while the tiles hold **raw metres above sea level**. The NIR constant is documented in
`sdfi_dataset.py:146` as *"based on a sample size of 1! OBS! UNKNOWN"* and propagated into every 6ch
and 10ch config.

**The sharper point — the variance decomposition.** Splitting each elevation channel's variance
into a between-tile part (differences between tile means) and a within-tile part (structure inside
a tile), over all 19,314 tiles:

| channel | total sd | between-tile sd | within-tile sd | between share |
|---|---:|---:|---:|---:|
| DSM | 23.707 m | 23.295 m | 4.403 m | **96.6 %** |
| DTM | 21.814 m | 21.773 m | 1.334 m | 99.6 % |
| nDSM | 8.977 m | 7.950 m | 4.169 m | 78.4 % |

Of everything the DSM channel varies by, **96.6 percent is between-tile terrain elevation** and
only 3.4 percent is structure inside the tile. Terrain elevation is largely a proxy for *where in
Denmark the tile is*, and route is the blocking unit — so the dominant component of the elevation
channel is a **spatial confound**, while object height is compressed to near nothing.

nDSM (= DSM − DTM) does not eliminate the between-tile component, since different areas genuinely
have different building and vegetation heights. What it does is change the ratio: nDSM carries
**21.6 percent** of its variance in the informative within-tile component against DSM's 3.4
percent, a roughly sixfold improvement, at a third of the total spread. That is why it is the
quantity worth feeding the network.

**And the discarded signal is real.** nDSM separates the classes cleanly:

| class | mean nDSM (m) |
|---|---:|
| betonflade | 0.19 |
| solceller | 0.26 |
| brosten | 0.78 |
| asfalt | 2.48 |
| ubefestet | 3.13 |
| drivhus | 4.11 |
| green_roof | **5.28** |

green_roof sits 5.3 m above ground — a distinctive signature — and the models score **0.008** on it.

**Implication for finding F4 of the handoff (RGB ≥ 10ch ≥ 6ch).** The honest conclusion is *not*
"the auxiliary channels do not earn their keep". It is that **the auxiliary channels as currently
normalised cannot contribute**, and that the physically meaningful derived quantity was never
formed. That is a different recommendation to KDS, and a stronger thesis result: it identifies a
concrete, cheap improvement lever rather than a dead end.

## F9. Two hypothesised defects that turned out not to matter

Reported because ruling things out is a result:

- **DSM/DTM nodata.** `ImageBlockReplacement.py:242` maps the `-9999` sentinel to `0 m`, which
  against real terrain is a fabricated cliff rather than a mask. Incidence is **70,200 of 19.31
  billion pixels (0.0004 percent)**. Real, documented, negligible.
- **The unused CIR bands.** The pipeline takes band 0 of `cir` and `OrtoCIR` as NIR and discards
  bands 1 and 2. Measured, those bands match `rgb` R and G almost exactly (100.26 / 103.82 against
  100.28 / 103.77), confirming the files are standard NIR-R-G composites and the pipeline's
  assumption is correct. **The 10-channel stack correctly avoids duplicating R and G.**

## F10. The paved classes are barely separable from pixel values, and context does all the work

A **context-free per-pixel linear discriminant**, trained on two folds and tested on the third
(same spatial protocol as the matrix) with class priors restored to the true pixel frequencies:

| | Macro-IoU | overall accuracy |
|---|---:|---:|
| trivial constant predictor (F1) | 0.0972 | 0.8747 |
| **per-pixel linear probe** | **0.1159** | 0.8634 |
| best trained CNN | 0.3586 | 0.9336 |

Pixel values alone get almost nowhere: ubefestet 0.874 IoU, solceller 0.098, and **every paved
class essentially zero** (asfalt 0.065, fliser 0.000, grus 0.000, brosten 0.003, betonflade 0.004).
The probe is barely above a predictor that does nothing.

Jeffries-Matusita separability (0 identical, 2 fully separable) says why:

| pair | JM |
|---|---:|
| asfalt vs grus | **0.366** |
| asfalt vs drivhus | 0.642 |
| grus vs drivhus | 0.649 |
| asfalt vs betonflade | 0.719 |
| asfalt vs brosten | 0.867 |

**This is the direct test of the F5b hypothesis, and it confirms it.** brosten, fliser, asfalt and
grus overlap heavily in pixel space. Their confusion is a property of the *data and the class
taxonomy*, not a defect of any particular architecture, and no architecture change will fix it.
What separates them is spatial pattern — texture, edges, context — which is exactly the gap between
0.116 and 0.359.

**Implication.** Nearly all measured model value on this task is contributed by spatial context
rather than by spectral discrimination. That is a strong argument for reporting per-category
metrics (the classes differ enormously in how much context helps) and it bounds what a
better-normalised channel stack could add for the paved classes specifically.

## F11. rgb and OrtoRGB are genuinely different acquisitions

Per-tile Pearson correlation between the two products, over 19,207 tiles: **mean 0.437, median
0.462**. Low enough that these are not two renderings of one image. This supports the decision to
add the 10-channel config as a distinct source rather than treating it as redundant.

**Open question for Rasmus.** Great Plan 2.2 §7.5 states `rgb`/`cir` are the GeoDanmark nadir
orthophoto and `OrtoRGB`/`OrtoCIR` the oblique LOD product; `multi_channel_dataset_creation/README.md`
lists the sources in the opposite order, and the Danish naming ("Orto" = ortofoto) points the other
way. The headline channel question is framed as a procurement decision, so which product is which
must be settled before any recommendation is made.

## F12. Route-level performance varies enormously, but mostly with class diversity, not place

Per-route out-of-fold scores for `convnext_upernet_rgb`, from the 16 route confusion matrices. The
route matrices sum to the existing pooled matrix **exactly, element by element, over all
19,314,000,000 pixels** — which validates the route-to-fold-to-prediction routing, the tile
partition and the accumulation in one assertion.

| route | tiles | classes present | route Macro-IoU | accuracy |
|---|---:|---:|---:|---:|
| 82-24 | 1,529 | 8 | **0.237** | 0.813 |
| 84-40 | 5,269 | 9 | **0.245** | 0.823 |
| 83-25 | 3,630 | 6 | 0.302 | 0.974 |
| … | | | | |
| 82-16 | 378 | 2 | 0.616 | 0.997 |
| 85-48 | 150 | 3 | 0.968 | 0.977 |
| 83-31 | 1,020 | **1** | **0.999** | 0.999 |

Median 0.490, range 0.237 to 0.999 — a spread of **0.76**, more than twice the entire range across
all 24 cells in the matrix (0.24 to 0.36).

**The caveat that matters, stated plainly.** Route-level Macro-IoU is averaged over the classes
*present in that route*, so it is **not comparable between routes**. Route 83-31 scores 0.999
because it contains exactly one class (it is entirely ubefestet), not because it is easy in any
interesting sense. The scores track class diversity almost monotonically: the two worst routes are
the class-rich urban ones (82-24 with 8 classes, 84-40 with 9).

That is precisely why the Great Plan 3.0 §5 test is **paired**: comparing two models on the *same*
route cancels the route's class composition, while comparing routes to each other does not. This
measurement is the concrete justification for that design, and the 384-matrix cache is what Part B
consumes.

---

## What this implies for the instrument (the bridge to SQ2)

| measured property | what it forces |
|---|---|
| Adjacent tiles share ~90% of variance; range 4.2 km (F3) | spatial cross-validation, with blocks larger than the range |
| Only different-route pairs reach independence (F4) | **route**-level blocking specifically, not parent-level |
| The naive split is 88% within 150 m (F5) | the existing KDS evaluation cannot measure generalisation |
| Trivial predictor scores 0.87 accuracy (F1) | per-category metrics; accuracy is uninformative here |
| Between-route composition varies 0–68% sealed (F6) | route-level pairing for the formal test; pooled numbers hide it |
| 572:1 class imbalance, 37% unscored (F7) | per-class reporting with support shown; absent-class rules |
| Paved classes overlap spectrally, JM 0.37 (F10) | per-category metrics; the taxonomy itself is a limitation to report |

Each row is a measurement on this dataset, not a citation. That is the bottom of the staircase
Great Plan 3.0 §2 says is missing.

---

## Still outstanding

- **E4** label quality — origin of the 37 percent ignore mass, and the boundary-error profile.
  Script written and self-tested; needs its run.
- **E6** per-route confusion matrices for all 24 cells — the geographic-variation decomposition and
  the cached input to the Part B route-level statistics. Script written, self-tested, and validated
  on one cell.

## Environment defect worth knowing about

`sklearn.discriminant_analysis.LinearDiscriminantAnalysis.fit` and
`sklearn.linear_model.LogisticRegression` with the lbfgs solver both **abort the process outright**
in this environment (Windows status `0xC0000409`), reproducibly, on a trivial 50,000 x 10 array of
random numbers. It is a broken LAPACK path in `c:\thesis\envs\ML_sdfi`, not a data problem, and a
hard abort cannot be caught in Python. The probe in `eda_separability.py` is therefore implemented
directly in numpy (`lda_fit_predict`), using only routines demonstrably sound here, and is verified
against separated Gaussians in `--selftest`. **Anything else in this project that reaches for
sklearn's iterative solvers will hit the same wall.**

## Limitation that cannot be closed from disk

The contribution of `data_cleaning_based_on_newer_ground_truth.py` (which sets pixels whose class
changed between two annotation vintages to ignore) **cannot be estimated**.
`labels/old_splitted_labels/` holds 16 tiles from a different tiling grid, only one of which shares
a filename with the current pool, and `labels/large_label/reclass/` was verified to be the same
rasters with the nodata sentinel 15 remapped to 0 — not an older vintage. No figure for annotator
disagreement is reported, and the boundary-error profile is used as the available proxy for
annotation precision.
