# 2026-09-08 — Post-hoc exploratory layer on the declared descriptive contrasts

**NOT CONFIRMATORY. Nothing in this document supports a claim.** Authorised by the author on
2026-09-08 after the results were seen, as "option (b)" — the alternative to overruling the locked
declaration and folding the arm cells into the Holm families.

**The pre-registered layer is untouched.** `wilcoxon_by_family.csv` is byte-identical before and
after this run, sha256 `21ab09d5d08132ee`, asserted programmatically on exit.

## What is and is not pre-registered

| | |
|---|---|
| **Pre-registered** | The twenty-one contrasts themselves. D1 fixed them on 2026-08-25, before any of the cells existed, and stated that *"no other descriptive contrast is permitted"*. There is no cherry-picking of which to test — the menu closed before the results arrived. Also the per-route metric (D3), the route and tie rules (D2), the seed and the bootstrap. |
| **Post-hoc** | The decision to compute significance tests on them at all. D1 declared these cells *"descriptive with CIs, outside every family… never tested for significance"*. That is what is being set aside. |

**One family of all twenty-one, not several.** Family structure chosen after seeing results is
precisely the failure mode this layer exists to avoid. Grouping by intervention or by architecture
would be a data-informed choice about which tests get the easier correction. A single family
containing every declared contrast cannot be accused of favourable selection, and it is the most
punishing: Holm's sharpest threshold is **0.05/21 = 0.00238**.

## The result

**Zero of twenty-one survive Holm.** The smallest uncorrected p is 0.00763.

| contrast | p uncorrected | Holm p | sign wins | rank-biserial | pre-registered CI excludes 0 |
|---|---:|---:|---:|---:|:--:|
| cnx `ortorgb` − `rgb` | **0.00763** | 0.1602 | 14/16 | +0.74 | — |
| rn34 `6ch_corrected` − `rgb` | 0.02899 | 0.5798 | 5/16 | −0.62 | **yes** |
| segf `ortorgb` − `rgb` | 0.02899 | 0.5798 | 12/16 | +0.62 | — |
| rn34 `armA` − `rgb` | 0.03354 | 0.6037 | 3/16 | −0.60 | **yes** |
| swin `ortorgb` − `rgb` | 0.11667 | 1.0000 | 11/16 | +0.46 | **yes** |
| rn34 `ortorgb` − `rgb` | 0.14386 | 1.0000 | 10/16 | +0.43 | — |
| … remaining 15 | 0.25–0.98 | 1.0000 | — | — | 3 of them yes |

Full table with both sensitivities in `results/tables/part_b/posthoc_exploratory_contrasts.csv`.

**The threshold was reachable.** The confirmatory layer's `cnx − segf` pair wins 15 of 16 routes at
p = 0.00076, which would clear 0.00238 comfortably. No arm contrast produces a route-level sweep
that consistent — the best is 14 of 16.

## What this actually shows, and it is the useful part

**The exploratory layer is strictly less informative than the pre-registered one.** Six contrasts
have a pre-registered bootstrap CI excluding zero; **none** reaches Holm-corrected significance. Had
the declaration been overruled and these cells folded into the families, the arms' evidence would
have come out **weaker**, not stronger.

That is not a quirk of the correction. It is D3's own warning appearing in the data:

- **cnx `ortorgb` − `rgb`** has the *smallest* p of all twenty-one — 14 of 16 routes, rank-biserial
  +0.74 — while its pooled difference is **−0.0012 with a CI spanning zero**. ConvNeXt's OrtoRGB cell
  wins on most routes by small margins that cancel in the pooled metric, because pooled counts the
  betonflade and solceller collapses in full and the route metric drops absent classes.
- **swin `ortorgb` − `rgb`** is the reverse: the largest pooled effect in the set, **+0.0497 with a
  CI excluding zero**, but only 11 of 16 routes and p = 0.117.

The two instruments disagree about which contrast is strongest, and the declaration said so in
advance: route-level values are *"comparable between models on the same route only, never between
routes, and never as standalone quality evidence"*, with the pooled metric as the headline quantity.

**D1's original choice is vindicated by its own test.** Declaring the arms descriptive with CIs was
not a limitation imposed for propriety — it selected the instrument that actually carries their
evidence.

## How to use this in the thesis

Use it, briefly, as a **robustness note in the limitations or an appendix**, not in the results
chapter. The honest sentence is roughly:

> A post-hoc exploratory analysis applied the same paired route-level tests to the twenty-one
> pre-declared descriptive contrasts under a single Holm family. None reached corrected
> significance, while six carried pre-registered bootstrap intervals excluding zero. The two layers
> disagree because the per-route metric drops classes absent from a route while the pooled metric
> counts them in full, and the arms' effects are concentrated in exactly those classes. The
> pre-registered estimation-based reading is therefore retained as the primary one.

Do **not** report any p-value from this layer as evidence for an arm's effect. Every row of the
artifact carries `status = POST-HOC EXPLORATORY -- not pre-registered, supports no claim`.

## Artifacts

- `results/tables/part_b/posthoc_exploratory_contrasts.csv` — 21 rows, uncorrected and Holm p under
  the primary rule and both declared sensitivities, sign test, rank-biserial, median route
  difference with CI, and the pre-registered pooled-difference CI joined on for comparison.
- `results/tables/part_b/posthoc_provenance.json` — the pre-registered / post-hoc split, the family
  rationale, and the confirmatory-layer hash.
- `exploratory_data_analysis/scripts/partb_posthoc.py`.
