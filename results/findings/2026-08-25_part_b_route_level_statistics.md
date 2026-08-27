# 2026-08-25 — Part B: the route-level statistics, run against the locked declaration

**Run under `2026-08-25_pre_declarations.md`, Status LOCKED 2026-08-25 02:51, on the author's
"declarations locked, go". Every quantity here was computed after the lock; none existed before it.
Script: `exploratory_data_analysis/scripts/partb_statistics.py`. Artifacts:
`results/tables/part_b/`.**

This closes the P0 item open since 28 July. It is the last piece of Level 1.

**Scope actually run.** All 8 family cells were scored, so all three families ran complete — no
Holm threshold is affected by a missing cell. Ten of the thirteen declared descriptive cells are
still in or ahead of the GPU queue and were skipped with notice; **5 of the 21 declared descriptive
contrasts were runnable** and are reported below. The run is repeatable and additive: re-running
after the Option 2 cells land fills in the remaining 16 contrasts and changes nothing computed here
(fixed seed 20260825, B = 10,000).

**One declared item not yet produced:** McNemar on pooled OOF pixels for the ten family pairs. It
needs per-pixel agreement between two cells, which confusion matrices cannot supply, so it requires
a streaming pass over the prediction rasters. It is context-only by declaration and its p-value is
explicitly not interpreted. Flagged here rather than quietly omitted.

---

## 1. Family 1 — model ranking within rgb, weighted

**Friedman omnibus: χ² = 15.075, df = 3, p = 0.00175, Kendall's W = 0.314. Rejects at α = 0.05,
so the family's narrative claim stands.** Mean ranks over the 16 routes (higher is better):

| cell | mean rank |
|---|---:|
| `convnext_upernet_rgb` | **3.500** |
| `swin_upernet_rgb` | 2.562 |
| `unet_resnet34_rgb` | 2.000 |
| `segformer_b1_rgb` | 1.938 |

Kendall's W of 0.314 is the honest reading of "how strongly do the 16 routes agree on the
ordering": they agree, but not overwhelmingly. ConvNeXt's mean rank of 3.500 against a ceiling of
4.000 is the strongest single number in the family.

**Pairwise, Holm-corrected within the family, primary route set:**

| pair | n′ | p exact | Holm p | reject | sign | sign p | median Δ | 95 % CI | rank-biserial | fragile |
|---|---:|---:|---:|:--:|---:|---:|---:|---|---:|:--:|
| cnx − segf | 16 | 0.00076 | **0.0046** | **YES** | 15/16 | 0.0005 | +0.0203 | [+0.0108, +0.0757] | +0.88 | — |
| cnx − rn34 | 16 | 0.00919 | **0.0459** | **YES** | 13/16 | 0.0213 | +0.0247 | [+0.0049, +0.1105] | +0.72 | **YES** |
| cnx − swin | 16 | 0.01309 | 0.0524 | no | 12/16 | 0.0768 | +0.0198 | [+0.0032, +0.0957] | +0.69 | — |
| swin − segf | 16 | 0.25223 | 0.5784 | no | 10/16 | 0.4545 | +0.0131 | [−0.0075, +0.0431] | +0.34 | — |
| swin − rn34 | 16 | 0.19281 | 0.5784 | no | 11/16 | 0.2101 | +0.0161 | [−0.0030, +0.0476] | +0.38 | — |
| segf − rn34 | 16 | 0.86026 | 0.8603 | no | 8/16 | 1.0000 | −0.0014 | [−0.0141, +0.0267] | +0.06 | — |

**Holm decision under each declared route and tie rule**, each corrected within its own family:

| pair | primary (n = 16) | Sensitivity A (n = 14) | Sensitivity B (near-tie) |
|---|---|---|---|
| cnx − segf | **REJECT** 0.0046 | **REJECT** 0.0139 | **REJECT** 0.0070 |
| cnx − rn34 | **REJECT** 0.0459 | — 0.0830 | — 0.0623 |
| cnx − swin | — 0.0524 | — 0.0981 | — 0.0623 |
| swin − segf | — 0.5784 | — 1.0000 | — 0.6233 |
| swin − rn34 | — 0.5784 | — 1.0000 | — 0.6233 |
| segf − rn34 | — 0.8603 | — 1.0000 | — 0.8469 |

**What Family 1 supports, stated at the strength the evidence carries.**

1. **ConvNeXt over SegFormer is the one robust pairwise architectural claim.** It survives Holm
   under all three declared route rules, the sign test agrees at 15 of 16 routes (p = 0.0005), the
   rank-biserial is +0.88, and the paired route-difference CI excludes zero.
2. **ConvNeXt over resnet34 holds in the primary analysis and fails under both sensitivities**, so
   by D2's interpretation rule it is **reported fragile** and never smoothed. The direction is
   consistent everywhere (13 of 16 routes, rb +0.72, CI excluding zero); what is not robust is the
   Holm-corrected significance.
3. **ConvNeXt over Swin does not reach Holm significance anywhere** (primary 0.0524, a near miss),
   though the direction is consistent at 12 of 16 routes with a CI excluding zero.
4. **Swin, SegFormer and resnet34 are not separable from one another.** SegFormer against resnet34
   is an essentially perfect null: 8 of 16 routes, p = 1.0000, rank-biserial +0.06.

This **narrows the handoff's F2** ("ConvNeXt > Swin ≈ SegFormer > resnet34"). The route-level
evidence supports the top of that ordering and nothing below it. Combined with the handoff §9.1
consistency rule, only ConvNeXt over SegFormer qualifies as a "consistent advantage" without
qualification.

## 2. Family 2 — channel effect within ConvNeXt, weighted, frozen configs

**Friedman omnibus: χ² = 2.375, df = 2, p = 0.30498, Kendall's W = 0.074. FAILS TO REJECT.** By
D1's own rule, **this family's narrative claim is downgraded to descriptive.** The runner records
`narrative_downgraded_to_descriptive: true` in `friedman_F2_channel_within_convnext_weighted.json`.

| pair | n′ | p exact | Holm p | reject | sign | median Δ | 95 % CI | rb | fragile |
|---|---:|---:|---:|:--:|---:|---:|---|---:|:--:|
| rgb − 6ch | 16 | 0.63217 | 0.9275 | no | 7/16 | −0.0008 | [−0.0124, +0.0326] | +0.15 | — |
| rgb − 10ch | 16 | 0.46375 | 0.9275 | no | 6/16 | −0.0039 | [−0.0759, +0.0099] | −0.22 | — |
| 6ch − 10ch | 16 | 0.05066 | 0.1520 | no | 5/16 | −0.0190 | [−0.0499, +0.0001] | −0.56 | **YES** |

**Two consequences, and both are corrections to things the thesis was going to assert.**

1. **The pooled `RGB ≥ 10ch ≥ 6ch` ordering has no route-level support.** `rgb` against `6ch` is a
   null (p = 0.632, median Δ −0.0008, 7 of 16 routes) and `rgb` against `10ch` is a null in the
   *opposite* direction to the pooled figure (p = 0.464, rb −0.22, 6 of 16 routes). At the route
   level the three frozen channel configurations are statistically indistinguishable for ConvNeXt.
   SQ3's channel conclusion must be restated accordingly.
2. **The "carrying both acquisitions earns its keep" finding does not reach significance in the
   primary analysis.** `6ch` against `10ch` gives Holm p = 0.1520 primary; it rejects **only** under
   Sensitivity A (Holm 0.0322), so the pair is reported **fragile**. The direction does favour the
   second acquisition throughout (11 of 16 routes, rb −0.56, median Δ −0.0190), and the family
   omnibus fails, so under the declared precedence this is **descriptive evidence of a direction,
   not a significant effect**. The handoff's 4-of-4-architectures pooled statement stands as a
   pooled statement and must be presented as one, with this route-level result beside it.

## 3. Family 3 — loss weighting within ConvNeXt × rgb

Single pair, no omnibus by declaration.

| pair | n′ | p exact | Holm p | reject | sign | median Δ | 95 % CI | rb |
|---|---:|---:|---:|:--:|---:|---:|---|---:|
| rgb − rgb_unw | 16 | 0.70572 | 0.7057 | no | 8/16 | +0.0073 | [−0.0122, +0.0217] | +0.12 |

**No weighting effect at route level.** Exactly 8 of 16 routes, sign p = 1.0000, rank-biserial
+0.12, CI straddling zero. The effective-number weighting decision is not visible in the per-route
metric. This is consistent with the saturation limitation already on record (handoff §11.2) and
should be written as a measured null rather than left implicit.

## 4. The effective-N question, answered against the prediction

D2 and the tracker predicted the effective N would collapse to roughly 9–12 once ties were dropped,
because seven of sixteen routes carry three classes or fewer and 83-31 is single-class.

**It did not. Every one of the ten pairs has n′ = 16 in the primary analysis, with zero exact
ties.** The declaration anticipated the mechanism correctly — "exact zeros will be rare in floating
point" — and that is exactly what happened: the low-diversity routes produce differences of order
10⁻⁴ rather than identically zero, so the classical zero-drop rule removes nothing.

Sensitivity B, the declared δ = 0.001 near-tie rule, is what actually does the work: it removes one
or two routes per pair, giving n′ = 14–15. So the power concern was real but the instrument for it
is Sensitivity B, not the exact-tie rule. **The "powered at n′ = 16" statement of D2 holds for the
primary analysis as run**, and the honest caveat is that one or two routes contribute differences
too small to be evidence.

## 5. Pooled Macro-IoU with whole-route block-bootstrap CIs

B = 10,000, seed 20260825, routes resampled with replacement, the statistic recomputed by summing
the resampled routes' confusion matrices.

| cell | Macro-IoU | 95 % CI | role |
|---|---:|---|---|
| `convnext_upernet_rgb_ndsm` | 0.3625 | [0.3140, 0.4659] | descriptive |
| `convnext_upernet_rgb` | 0.3586 | [0.3091, 0.4509] | family |
| `convnext_upernet_10ch` | 0.3447 | [0.2759, 0.4900] | family |
| `convnext_upernet_6ch` | 0.3374 | [0.2926, 0.4388] | family |
| `convnext_upernet_rgb_unw` | 0.3336 | [0.2796, 0.4373] | family |
| `convnext_upernet_6ch_corrected` | 0.3110 | [0.2427, 0.4576] | descriptive |
| `swin_upernet_rgb` | 0.2982 | [0.2359, 0.3709] | family |
| `segformer_b1_rgb` | 0.2976 | [0.2337, 0.3513] | family |
| `unet_resnet34_rgb` | 0.2695 | [0.1962, 0.3298] | family |
| `unet_resnet34_6ch_corrected` | 0.2025 | [0.1647, 0.2722] | descriptive |

**Read these intervals for what they are, and warn the reader in the text.** They are wide and they
overlap heavily — `convnext_upernet_rgb` [0.3091, 0.4509] against `unet_resnet34_rgb` [0.1962,
0.3298] barely separate, and most pairs overlap outright. That is the expected consequence of
resampling only 16 spatially blocked routes, and it is **not** evidence of no difference. Comparing
two overlapping independent CIs is the classic error here; the paired test of §1 is the instrument
that answers the comparison question, because it holds the route fixed. The intervals belong in the
thesis as an honest statement of how much a 16-route bootstrap can pin down a single cell.

Note also the asymmetry: every interval is skewed upward, because a resample that draws more of the
class-rich routes raises Macro-IoU more than a resample of the poor ones lowers it.

## 6. Descriptive contrasts — paired whole-route bootstrap on the pooled difference

**No p-values, by declaration.** Same route resample applied to both cells in every replicate.
Five of the twenty-one declared contrasts were runnable; the other sixteen await cells still in the
queue.

| contrast | difference | 95 % CI | excludes 0 |
|---|---:|---|:--:|
| `rgb_ndsm` − `rgb` (cnx) | +0.0039 | [−0.0050, +0.0184] | no |
| `6ch_corrected` − `6ch` (cnx) | −0.0264 | [−0.0661, +0.0423] | no |
| `6ch_corrected` − `rgb` (cnx) | −0.0476 | [−0.0739, +0.0166] | no |
| `6ch_corrected` − `6ch` (rn34) | −0.0452 | [−0.0758, −0.0032] | **yes** |
| `6ch_corrected` − `rgb` (rn34) | −0.0669 | [−0.1031, −0.0009] | **yes** |

**The channel-axis harm claim now has an interval, and it is asymmetric across architectures.**
The correction's damage is interval-supported on the production architecture — resnet34's
`6ch_corrected` sits below both its `6ch` and its `rgb` cell with CIs excluding zero — but **not**
on ConvNeXt, where both intervals straddle zero despite point differences of −0.0264 and −0.0476.

This does not overturn `2026-08-19_channel_axis_findings.md`. That verdict rests on a mechanism
measured independently of any single run (the F8 variance decomposition, the nDSM control, the
per-fold signature, the betonflade collapse), and the mechanism is untouched. What changes is the
strength claimable from the pooled numbers alone: **on ConvNeXt the drop is a point estimate whose
interval includes zero, and the write-up must say so.** The betonflade and fold-0 evidence carries
the ConvNeXt half of the argument, not the pooled delta.

`rgb_ndsm` − `rgb` at +0.0039 with CI [−0.0050, +0.0184] is exactly the "inside noise" reading
already on record, now with an interval behind it.

## 7. A bug found and fixed while reading the first run — recorded because it changed a verdict

The first execution implemented D2's fragility rule as "primary Holm decision versus each
sensitivity's **raw** p against α". D2 says "primary and either sensitivity disagree in significance
**after Holm**". Those are different tests, and the difference was not cosmetic:

- the raw-p version flagged **cnx − swin** as fragile, which it is not — that pair fails to reject
  consistently under all three rules;
- and it **missed cnx − rn34**, which rejects in the primary and fails under both sensitivities once
  each sensitivity is Holm-corrected within its family. That is a genuine fragility attached to a
  headline architectural claim.

Each route set is now Holm-corrected within its own family and the comparison is decision against
decision, with the reason recorded per pair in `fragile_reason`. The numbers in §1–§3 are from the
corrected run. Both runs used the same seed and the same locked declaration; only the fragility
column changed.

A second, harmless failure on the first run: the provenance JSON dump raised because YAML parses the
unquoted `declaration_version: 2026-08-25` as a `datetime.date`. Fixed with `default=str` in the
runner rather than by editing the locked declaration to suit the tool.

## 8. What this changes in the thesis

1. **SQ3's model claim narrows.** Report ConvNeXt on top with the Friedman omnibus and mean ranks,
   ConvNeXt over SegFormer as the one robust pair, ConvNeXt over resnet34 as fragile with its
   direction consistent, and the bottom three as not separable.
2. **SQ3's channel claim is downgraded to descriptive** by the declaration's own omnibus rule, and
   the pooled `RGB ≥ 10ch ≥ 6ch` ordering must be presented as a pooled ordering with an explicit
   statement that it has no route-level support.
3. **The weighting result is a measured null** at route level and should be written as one.
4. **The statistical-limitations subsection gains a paragraph** on the wide, overlapping,
   upward-skewed bootstrap intervals and why the paired test rather than interval overlap answers
   the comparison question, alongside E4's supervision-range finding and the effective-N outcome
   of §4.

## Artifacts

- `results/tables/part_b/wilcoxon_by_family.csv` — 30 rows, 10 pairs × 3 route sets, every declared
  reporting field including n′, exact ties, near ties, Holm p, sign test, effect sizes, fragility.
- `results/tables/part_b/pooled_macro_iou_route_bootstrap.csv` — 10 cells with CIs.
- `results/tables/part_b/descriptive_contrast_paired_bootstrap.csv` — 5 contrasts.
- `results/tables/part_b/friedman_F1_model_ranking_rgb_weighted.json`,
  `friedman_F2_channel_within_convnext_weighted.json`.
- `results/tables/part_b/run_provenance.json` — declaration version and status, route sets, which
  descriptive cells were skipped and which contrasts ran.
