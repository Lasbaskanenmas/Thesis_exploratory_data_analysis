# 2026-09-11 — The qualitative error analysis: results

**Executes locked declaration D4 with its two dated amendments. Both scoring passes complete, keys
unsealed after scoring, nothing re-scored. This is the second evaluation leg and the LO7 vehicle.**

Declaration: `2026-08-25_pre_declarations.md`, LOCKED 2026-08-25 02:51.
Amendments: `2026-09-08_D4_amendment_mode_scoping_rule.md` (rule A, `not_assessable`) and
`2026-09-08_D4_section_4_6_drop_invoked_and_withdrawn.md` (sample stayed at 96).

## 0. What was scored, and validation

96 tiles, 32 per fold, half weak-class-present. Two cells scored blind with identity hidden and
order shuffled: `unet_resnet34_rgb` (production) and `convnext_upernet_rgb` (best). **192 pass-1
items, 24 pass-2 items, every row complete.**

Validation passed before anything was computed: row counts, all mode columns in {0, 1},
`dominant_mode` inside the closed vocabulary, and the consistency rule that a dominant mode must
itself be marked present while `none` and `not_assessable` require all nine absent. No exceptions.

**One tile was marked `not_assessable`** under the amendment, in both cells — so N = 95 assessable
items per cell, and every proportion below is reported against that denominator. See §4; that single
tile turned out to matter more than its count.

## 1. Incidence, mode × cell, over assessable items

| mode | ConvNeXt (best) | resnet34 (production) |
|---|---:|---:|
| m1 speckle | 4 / 95 (4.2 %) | **20 / 95 (21.1 %)** |
| m2 boundary bleeding | **52 / 95 (54.7 %)** | 43 / 95 (45.3 %) |
| m3 spurious minority patch | 36 / 95 (37.9 %) | **57 / 95 (60.0 %)** |
| m4 paved-class swap | 31 / 95 (32.6 %) | 44 / 95 (46.3 %) |
| m5 road/driveway continuation | 43 / 95 (45.3 %) | 42 / 95 (44.2 %) |
| m6 vegetation occlusion | 12 / 95 (12.6 %) | 10 / 95 (10.5 %) |
| **m7 misregistration echo** | **0 / 95 (0.0 %)** | **0 / 95 (0.0 %)** |
| m8 large-object failure | 3 / 95 (3.2 %) | 5 / 95 (5.3 %) |
| m9 other | 9 / 95 (9.5 %) | 10 / 95 (10.5 %) |

**Dominant mode**, the single most consequential failure per item:

| dominant | ConvNeXt | resnet34 |
|---|---:|---:|
| **none (clean)** | 26 / 95 (27.4 %) | 24 / 95 (25.3 %) |
| m2 boundary bleeding | **33 (34.7 %)** | 23 (24.2 %) |
| m3 spurious patch | 8 (8.4 %) | **22 (23.2 %)** |
| m1 speckle | **0 (0.0 %)** | **11 (11.6 %)** |
| m4 paved swap | 11 (11.6 %) | 6 (6.3 %) |
| m6 vegetation occlusion | 7 (7.4 %) | 3 (3.2 %) |
| m5 road continuation | 6 (6.3 %) | 5 (5.3 %) |
| m9 other | 4 (4.2 %) | 1 (1.1 %) |
| m7, m8 | 0 | 0 |

### What this says

**The two architectures fail differently, and the difference is legible.** resnet34 is *noisy*:
speckle at 21.1 % against ConvNeXt's 4.2 %, and spurious minority patches at 60.0 % against 37.9 %.
Between them those two account for 34.8 % of resnet34's dominant tags and 8.4 % of ConvNeXt's.
ConvNeXt is *cleaner but blunter*: its dominant failure is boundary bleeding at 34.7 %, and it is
worse than the production model on that mode in absolute terms (54.7 % against 45.3 %).

**Roughly a quarter of items are clean under both models** — 27.4 % and 25.3 %. That is the honest
headline for a product that scores 0.93 overall accuracy: on three quarters of assessable tiles a
trained scorer can see at least one named failure.

**Nobody is failing at large objects, and nobody is misregistered.** m8 is never a dominant mode in
either cell, and **m7 is literally zero in 190 assessable items**. The m7 result is coherent rather
than surprising: both scored cells are RGB-only, so there is no elevation channel to produce a
misregistration echo. The 671 misregistered DSM/DTM tiles cannot express themselves here. **This is
a scoped null, not evidence that the product is geometrically sound**, and it must be written that
way.

### Triangulation with the quantitative leg

- **m2 at 45–55 % corroborates the Boundary IoU finding.** Boundary IoU showed bulk classes losing
  about a third of their quality at the edges (ubefestet 0.93–0.96 mask against 0.60–0.66 boundary)
  and betonflade collapsing 0.200 → 0.037. Boundary bleeding being the single most common failure
  mode, and ConvNeXt's dominant one, is the same phenomenon seen by eye.
- **m3 at 60 % on the production model is exactly D3's declared blindness.** The per-route metric
  averages only the classes present in a route, so a false-positive island of a class with no route
  support vanishes from the route score. D4 named mode 3 the "D3 blindness catcher" in advance. It
  caught something substantial: the most common failure of the production model is invisible to the
  route-level statistic that carries the formal tests.
- **m4 at 32.6 % and 46.3 % corroborates F10 and the JM separability matrix.** Paved classes overlap
  heavily in pixel space (asfalt vs grus JM 0.366 of 2.0); the scorer sees that as swaps on roughly a
  third to a half of tiles.

## 2. Self-consistency — reported as-is

24 re-scored items over 12 tiles, second pass at least 24 hours after the first, under fresh ids.

| mode | pass 1 | pass 2 | raw agreement | Cohen's κ |
|---|---:|---:|---:|---:|
| m1 speckle | 6 | 5 | 95.8 % | **0.882** |
| m3 spurious patch | 12 | 14 | 83.3 % | 0.667 |
| m2 boundary bleeding | 13 | 10 | 79.2 % | 0.589 |
| m4 paved swap | 8 | 11 | 79.2 % | 0.571 |
| m5 road continuation | 15 | 16 | 79.2 % | 0.545 |
| m9 other | 4 | 3 | 87.5 % | 0.500 |
| m6 vegetation occlusion | 2 | 0 | 91.7 % | prevalence too low |
| m8 large object | 2 | 0 | 91.7 % | prevalence too low |
| m7 misregistration | 0 | 0 | 100 % | undefined, no variation |

**Dominant-mode exact agreement: 54.2 %. Clean / not-clean agreement: 100 %.**

That pair of numbers is the most important thing in this section and it must be reported together.
**The scorer never once disagreed with himself about whether a tile had a problem.** All 24
re-scored items were placed on the same side of the clean / not-clean line both times. What is
unstable is *which* failure dominates when several are present — agreement on that is barely better
than a coin flip among the modes that actually occur.

**Consequence for how the results are read.** The presence columns are reliable enough to support
the incidence table. The `dominant_mode` ranking is not, and **should be reported descriptively
rather than used to claim one mode matters more than another**. Where the two architectures differ
on dominance — ConvNeXt 34.7 % boundary bleeding against resnet34's 23.2 % spurious patch — that
difference is large enough to survive the instability, but a 6.3 % against 5.3 % difference is not.

The κ values also degrade in a readable way: the mode with the sharpest visual signature, speckle,
is the most repeatable at 0.882, while the modes that require a judgement about where one surface
ends and another begins sit at 0.55–0.67. That is a property of the taxonomy, not of the scorer.

**As declared, nothing was re-scored to improve these figures.** D4 item 6 said they would be
reported as-is and they are.

## 3. Exemplars, chosen by the fixed rule

D4 item 7: the first tile in sampled order exhibiting the mode in the ConvNeXt cell, falling back to
the production cell, never hand-picked. Panels copied to `analysis/exemplars/`.

| mode | item | cell | tile |
|---|---|---|---|
| m1 speckle | 0060 | ConvNeXt | `O2021_84_40_1_0039_00062715_6000_0` |
| m2 boundary bleeding | 0147 | ConvNeXt | `O2021_84_40_1_0038_00060852_0_5000` |
| m3 spurious patch | 0142 | ConvNeXt | `O2021_84_40_1_0047_00073810_3000_2000` |
| m4 paved swap | 0142 | ConvNeXt | `O2021_84_40_1_0047_00073810_3000_2000` |
| m5 road continuation | 0147 | ConvNeXt | `O2021_84_40_1_0038_00060852_0_5000` |
| m6 vegetation occlusion | 0125 | ConvNeXt | `O2021_84_40_1_0037_00084312_5000_2000` |
| m8 large-object failure | 0095 | ConvNeXt | `O2021_84_40_1_0038_00060852_4000_2000` |
| m7 misregistration | — | — | no item exhibits this mode |

**Every exemplar comes from route 84-40 and fold 0.** That is a direct consequence of "first tile in
sampled order" — fold 0 is drawn first and 84-40 dominates it. The rule was fixed in advance and was
followed exactly; the concentration is a property of the rule, not a selection. If the figures need
geographic variety, the honest fix is a **dated amendment changing the rule**, not a quiet
substitution.

## 4. The `[mismatch]` tile — independent corroboration of an open finding from another agent

The one tile marked `not_assessable`, in both cells, carries the scorer's note `[mismatch]`. It is
**`O2021_82_24_1_0021_00002048_0_4000.tif`**.

**That tile is in Agent 4's list of 273 tiles whose label and rgb geotransforms disagree.**

| | |
|---|---:|
| tiles in Agent 4's mismatch list | 273 of 19,314 (1.41 %) |
| expected overlap with a 96-tile sample | 1.36 |
| actual overlap | **1** |
| of those, independently flagged by the blind scorer | **1 of 1** |

Agent 4's handoff §5.1 states the limit of what they established: *"What is established: the
georeferencing headers disagree. What is NOT established, and is the open question: whether the
content of those 273 label rasters is aligned with the corresponding image content. If it is not,
those tiles are mislabelled training and evaluation data."*

**A scorer with no knowledge of that list, looking only at a rendered panel, flagged the one
affected tile in the sample as unusable because the label and the image did not correspond.** That
is independent visual evidence that the mismatch reaches the content, not only the headers.

**Stated with its limits.** n = 1. The sample was not designed to test this and one hit is exactly
the base rate, so this establishes nothing about the other 272. What it does is convert Agent 4's
open question from "unknown" to "at least one confirmed case, found blind". That is enough to
justify the small work order they recommended, and enough to state in the limitations as a known
data-integrity issue with a confirmed instance rather than a header anomaly.

## 5. The `[miss]` theme — a taxonomy gap, recorded as post-hoc

23 of 192 items carry a free-text note. Tagged by the scorer:

| tag | items | what it marks |
|---|---:|---|
| `[miss]` | **12** | the model omitted a whole class that is present in the ground truth |
| `[unannot]` | 3 | observation about unannotated ground |
| `[label]` | 2 | the annotation looks wrong, not the prediction |
| `[mismatch]` | 2 | the §4 tile, both cells |

**This is post-hoc thematic coding of free text. It is not a declared mode and no incidence claim
rests on it.**

It does carry one finding worth stating. **`[miss]` at 12 items is more frequent than the declared
m8 large-object mode at 8 items across both cells.** The closed taxonomy has a mode for *fragmenting
or hallucinating* a large installation, but none for *omitting an entire class that is present* —
which is exactly the brosten and green_roof behaviour the learning curve measured at IoU 0.0000 at
every training volume. The scorer met it repeatedly and had nowhere to put it except `m9_other`.

**Recommendation for the D4 taxonomy, as future work:** add a "class omission" mode. The closed
taxonomy was correct to stay closed mid-study — that is what makes the incidence counts meaningful —
but the gap is real and the thesis should say so rather than let 12 notes sit unexplained.

The 2 `[label]` items are a second, smaller theme: cases where the scorer judged the *annotation*
wrong rather than the prediction. That is E4's coverage and quality finding appearing from the
qualitative side, and it belongs beside the annotation-ceiling limitation.

## 6. Operational consequences, per D4 item 8

The mapping was fixed in advance and no consequence may be attached post hoc to a mode not listed.
Reading it against the incidence:

| dominant failure | share of items | operational consequence for KDS |
|---|---:|---|
| boundary bleeding | 34.7 % cnx / 24.2 % rn34 | systematic area misestimation at parcel scale |
| spurious patches + speckle | 8.4 % cnx / **34.8 % rn34** | false imperviousness change signals; noise in parcel-level statistics |
| paved-class swap | 11.6 % / 6.3 % | wrong surface-type input to drainage coefficients |
| road/driveway + occlusion | 13.7 % / 8.5 % | connectivity and access-surface errors in municipal use |

**The production model's dominant operational risk is different from the best model's.** resnet34's
noise profile maps to *false change signals* — the failure mode that matters most for a product used
to monitor imperviousness over time, because spurious patches and speckle appear and disappear
between vintages and read as real change. ConvNeXt's maps to *area misestimation*, which is a
systematic bias rather than a noise source and is the easier of the two to correct downstream.

That is a concrete, evidence-backed recommendation: **switching architecture would change the
character of the error, not merely its quantity**, and for change detection specifically the
noisier production model is the worse of the two in a way overall accuracy does not show.

## Artifacts

```
results/qualitative_pack/analysis/
  incidence_by_mode.csv          mode x cell, counts, N, proportions, declared consequence
  dominant_mode_by_cell.csv      dominant-mode distribution per cell
  self_consistency.csv           per-mode agreement, kappa, reportability flag
  exemplars.csv                  the fixed-rule selection, with fold and stratum
  exemplars/exemplar_m*.png      8 panels
  note_themes.csv                post-hoc tag coding, item ids, status label
  analysis_provenance.json       declaration, amendments, counts, validation record
```

Regenerable by `scripts/wo2_analyse.py` against the two sheets and the sealed keys.
