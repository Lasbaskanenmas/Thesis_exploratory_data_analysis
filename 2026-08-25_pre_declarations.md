# 2026-08-25 — Pre-declarations for the inferential and qualitative legs (D1–D4)

**Status: LOCKED, 2026-08-25 02:51. Prepared 2026-08-24 evening under Great Plan 3.1
§5.1; locked by the author before any Part B statistic was computed and before any G/A arm
score was read.**

**Author: Lasse. Drafted by the cowork session from tracker §4, Plan 3.1 §5.1 and the E6/route
artifacts. The author owns every choice below; bracketed [AUTHOR CONFIRM] items must be resolved
at lock.**

**Role.** These four declarations fix, in writing and in advance, every analytic degree of freedom
in the route-level statistics (Part B) and the qualitative error analysis: the test family, the
route and tie rules, the per-route metric, and the sampling frame. Plan 3.0 §9 and the tracker §4
require them before any p-value exists or any tile is inspected. The Part B runner (VM work order
of 2026-08-24, Task 3) reads the machine-readable block in §6 verbatim.

## 0. Transparency clause — what has and has not been seen

Declared plainly so the pre-registration claim is honest:

- **Seen before this declaration:** all descriptive pooled scores and per-fold diagnostics of the
  27 scored cells; the E6 per-route score cache including route medians, minima, maxima and
  spreads (landed 2026-08-23); the learning-curve results. In particular, the direction-relevant
  facts that route medians can invert pooled orderings (`convnext_upernet_6ch_corrected` median
  0.4982 vs `convnext_upernet_rgb` 0.4902) were known when D3 was written.
- **Not seen, and not in existence:** any test statistic, p-value, confidence interval, bootstrap
  replicate or Holm-adjusted quantity on real data; any score for the two arms now training
  (`convnext_upernet_ortorgb`, `convnext_upernet_rgb_dsm_dtm_corrected`).
- Therefore: the declarations precede **all** inferential quantities, and precede the two new
  cells entirely. Where a choice could have been steered by a seen descriptive number, the
  steering-relevant fact is named here rather than left discoverable.
- **Amendment 25/8, before lock (Plan 3.1 §11.5, Option 2):** the descriptive cell set expanded
  to the full four-model roster. Of the thirteen descriptive cells now declared, three have been
  seen as stated above (`rgb_ndsm` and the two scored `6ch_corrected` cells); the remaining ten
  have no results in existence — several are not yet trained.

## 1. D1 — The test family and the Holm–Bonferroni structure

**α = 0.05, two-sided, throughout.** Three families, ten tests, weighted cells only, corrected by
Holm–Bonferroni **within each family**. Weighting is held constant inside families 1 and 2 so that
each family varies exactly one design axis.

**Family 1 — model ranking within the best channel config (rgb, weighted), 6 pairs:**

1. `convnext_upernet_rgb` vs `swin_upernet_rgb`
2. `convnext_upernet_rgb` vs `segformer_b1_rgb`
3. `convnext_upernet_rgb` vs `unet_resnet34_rgb`
4. `swin_upernet_rgb` vs `segformer_b1_rgb`
5. `swin_upernet_rgb` vs `unet_resnet34_rgb`
6. `segformer_b1_rgb` vs `unet_resnet34_rgb`

**Family 2 — channel effect within the best model (ConvNeXt, weighted), frozen configs only,
3 pairs:**

7. `convnext_upernet_rgb` vs `convnext_upernet_6ch`
8. `convnext_upernet_rgb` vs `convnext_upernet_10ch`
9. `convnext_upernet_6ch` vs `convnext_upernet_10ch`

**Family 3 — loss weighting within best model × best channel, 1 pair:**

10. `convnext_upernet_rgb` vs `convnext_upernet_rgb_unw`

**Omnibus layer (added 24/8 evening on the author's robustness query).** For the two multi-pair
families, a **Friedman test across the 16 routes** (F1: 4 models × 16 routes; F2: 3 channel
configs × 16 routes) is computed per the Demšar (2006) protocol for comparing classifiers over
multiple data sets, reported with the mean rank per treatment and **Kendall's W** as the
concordance effect size ("how strongly do the 16 routes agree on the ordering"). Role: family-level
omnibus context, and the canonical literature anchor for the whole design. The Holm-corrected
pairwise Wilcoxons remain the confirmatory instrument and run regardless of the omnibus outcome;
**if a family's Friedman fails to reject at α = 0.05, that family's narrative claim is downgraded
to descriptive** even where an individual Holm-corrected pair is significant, and the disagreement
is reported as such. Family 3 is a single pair and gets no omnibus.

"Best model" and "best channel config" are fixed by the frozen pooled ordering on record since
2026-07-28 (ConvNeXt; rgb). Family 2 tests the frozen 6ch/10ch cells **as the production pipeline
configured them**, defects included; the corrected arms carry that story descriptively.

**Declared outside every family, descriptive only, never tested for significance (amended 25/8
to the full roster):** the thirteen arm cells — `convnext_upernet_rgb_ndsm`; `6ch_corrected`,
`ortorgb` and `rgb_dsm_dtm_corrected` for each of `convnext_upernet`, `swin_upernet`,
`segformer_b1` and `unet_resnet34`. Their pre-declared descriptive contrasts (paired whole-route
bootstrap CIs on the pooled-metric difference, no p-values) are fixed by rule, and **no other
descriptive contrast is permitted**: for every model m in the roster,

- `<m>_ortorgb` − `<m>_rgb`
- `<m>_rgb_dsm_dtm_corrected` − `<m>_6ch_corrected`, and − `<m>_rgb`
- `<m>_6ch_corrected` − `<m>_6ch`, and − `<m>_rgb`

plus `convnext_upernet_rgb_ndsm` − `convnext_upernet_rgb`. Twenty-one contrasts in total,
enumerated exhaustively in the §6 machine block.

**Power arithmetic, declared in advance so the outcome cannot be reframed later.** Wilcoxon
signed-rank at n′ non-zero pairs has minimum attainable two-sided p = 2^(1−n′): 0.0000305 at
n′ = 16, 0.0039 at n′ = 9, 0.0078 at n′ = 8. Holm's sharpest threshold across a 6-test family is
0.05/6 ≈ 0.0083, across the 3-test family 0.0167, across the whole ten if ever pooled 0.005.
Consequently, at small effective N some pairs **cannot** reach Holm-corrected significance even
under a perfect sweep. **Non-significance under Holm at small effective N is declared here as an
expected, fully reportable outcome**, and the "powered without seeds" wording of Plan 2.2 §7.6 is
restated as: powered at n′ = 16; power degrades as ties and near-ties shrink n′, which is why n′
is reported beside every p-value. The headline framing rule from the handoff §9.1 stands: an
architectural claim is a "consistent advantage" only when the pooled direction holds in all three
fold diagnostics **and** the route-level Wilcoxon agrees after Holm.

## 2. D2 — The route set, the tie rules, and the reporting duty

**Primary analysis:** two-sided Wilcoxon signed-rank on the 16 paired route scores per declared
pair, exact distribution (n′ ≤ 16 always permits it), zero-differences removed per the standard
procedure.

**Reported beside every p-value, without exception:** n′ (pairs entering the test after zero
removal), the zero count, the near-tie count as defined below, and the effect sizes of §4.

**Sensitivity A — route size:** re-run each test dropping routes with fewer than 200 tiles — 83-34 (10 tiles) and 85-48 (150 tiles), the two routes an order of magnitude below the rest of the field (next candidate 82-21 at 313 is retained); 85-48 additionally shares ground with a route in another fold across 100 % of its tiles (2026-08-04 report), an independent reason it is the right second drop. n = 14. Threshold corrected from 100 before lock; the inherited figure could not have dropped 85-48.

**Sensitivity B — near-ties:** re-run each test treating |d| < **0.001** Macro-IoU as a tie
(dropped). Rationale, declared: on the single-class and near-single-class routes every cell scores
≈ 0.999 and paired differences there are of order 10⁻⁴ — bounded-scale noise, not evidence; exact
zeros will be rare in floating point, so without this sensitivity the formal n′ = 16 would
overstate the information content the tracker's effective-N analysis (7 of 16 routes carry ≤ 3
classes; 83-31 is single-class) says is really 9–12. The primary analysis stays the untouched
standard test precisely so that δ = 0.001 is a labelled sensitivity, not a tunable knob.

**Interpretation rule:** a pair's result is reported fragile if primary and either sensitivity
disagree in significance after Holm; fragility is stated, never smoothed.

**Companion robustness test (added 24/8 evening):** beside each Wilcoxon, the **exact two-sided
sign test** on the same paired route differences (zeros dropped; the δ = 0.001 near-tie rule of
Sensitivity B applies to it identically). The sign test uses direction only, so it is immune to
the one assumption the signed-rank test does lean on here — that |d| magnitudes are comparable
across routes when ranked, which is doubtful when route class composition differs this much. It is
also the formal version of the "wins in x of 16 routes" sentence the results chapter will use
anyway. Role: robustness check, never confirmatory; reported as wins/n′ with the exact binomial p,
uncorrected. Reference power, declared now: at n′ = 16, 13/16 wins gives p ≈ 0.021 and 14/16 gives
p ≈ 0.004. A pair where the sign test and the Wilcoxon disagree in significance direction is
reported fragile, same rule as the sensitivities.

## 3. D3 — The per-route metric (the declaration that fixes a direction)

**Definition (the implemented, self-tested one, now declared):** for route r and cell c, the
per-route score is **macro-IoU over the classes present in route r's held-out ground truth**,
computed from the (cell, route) confusion matrix; absent classes are dropped, never scored 0;
class presence means **support_pixels > 0** in the route's ground truth
[verified at lock against eda_route_cell_metrics.py: the implemented presence rule is support greater than zero, with no minimum-pixel threshold anywhere in the path, matching this declaration as written];
`ignore_index` 0 excluded; `unknown2` report-only. Both cells in a pair are scored on the
identical route set with the identical present-class list, so the paired difference
d_r = m_A(r) − m_B(r) cancels route composition by construction.

**Declared consequences, accepted with mitigations:**

1. **The metric is blind to false positives on absent classes** (a spurious brosten patch on a
   route with no brosten costs nothing at route level). Accepted because the route level exists
   solely to carry the paired test; the **pooled OOF metric remains the headline quantity**, and
   it counts every false positive in full. The qualitative taxonomy (D4, mode 3) is the declared
   instrument for exactly this failure mode.
2. **Route-level values are never comparable between routes and never standalone quality
   evidence.** Worked example, known before this declaration: `convnext_upernet_6ch_corrected`
   posts a higher route median than `convnext_upernet_rgb` (0.4982 vs 0.4902) while losing pooled
   by 4.8 points, because its collapsed classes (betonflade) drop out of the routes that lack
   them. Route medians are therefore reported only as descriptive context beneath the paired
   tests, with this mechanism stated wherever a median appears.
3. The 82-20 zero-fill caveat (333 of 608 tiles without elevation signal) applies to the
   `rgb_ndsm` arm's route contribution only and is restated in the statistical-limitations
   subsection.

## 4. Effect sizes, intervals, and context statistics (all pre-specified)

- **Effect size per tested pair:** the median paired route difference with its 95 % CI from the
  paired whole-route bootstrap below, plus the matched-pairs rank-biserial correlation
  r = (W⁺ − W⁻) / (n′(n′+1)/2) as the secondary standardised measure.
- **Block bootstrap, whole routes:** B = 10,000 replicates, seed 20260825. Each replicate draws 16
  routes with replacement; the pooled metric is recomputed by **summing the resampled routes'
  confusion matrices** and recomputing Macro-IoU from the summed matrix (never by averaging route
  scores). Percentile 95 % intervals. For paired differences (the descriptive contrasts of §1 and
  the effect sizes above), the **same route resample is applied to both cells** in each replicate.
  Cells receiving standalone CIs: the eight family cells of §1 plus the thirteen descriptive arm
  cells (amended 25/8).
- **McNemar on pooled OOF pixels:** computed for the ten family pairs only, reported as context —
  discordant-pixel direction and magnitude — with its p-value explicitly not interpreted
  (n ≈ 12.2 billion spatially autocorrelated pixels makes it near-automatic by construction).

**Precedence, declared.** Confirmatory: Holm-corrected exact Wilcoxon within the three families.
Omnibus context: Friedman with mean ranks and Kendall's W. Robustness: the sign test and
Sensitivities A and B. Estimation: paired route-bootstrap CIs and rank-biserial. Context only:
McNemar. Above all of it sits the handoff §9.1 consistency rule: a "consistent advantage" claim
requires pooled direction in all three fold diagnostics plus a Holm-surviving Wilcoxon.
Disagreement anywhere in this stack is reported as fragility; **it never triggers a search for a
friendlier statistic.**

**Declared exclusions — robustness by restraint (added 24/8 evening).** No significance test is
computed on any metric other than D3's per-route metric: accuracy, macro-F1, FW-IoU, the
worst-case metric, Boundary IoU and the binary collapse are descriptive everywhere they appear.
No per-class significance tests (weak-class route support collapses to n ≤ 7, where the §1 power
arithmetic makes p-values uninterpretable); per-class results stay descriptive with support shown.
No TOST equivalence tests on the arm contrasts — the paired CIs carry the "no material
difference" reading, a defensible equivalence margin does not exist a priori, and inventing one
now would be a researcher degree of freedom; the absence is acknowledged in the limitations. No
Bayesian re-analysis (noted as future work only). **These exclusions are declarations too: adding
any of them later requires a dated amendment to this file.**

## 5. D4 — The qualitative error analysis: pre-registered sampling frame and protocol

**Nothing below may be executed, and no held-out tile inspected for this purpose, before this
document is locked.** Seed for every random step: **20260825**.

1. **Population:** the 19,314 held-out tiles (each tile is held out exactly once under the frozen
   3-fold OOF structure).
2. **Sample: n = 96 tiles, 32 per fold.** Within each fold: 16 tiles drawn uniformly at random
   from the stratum whose ground truth contains at least one pixel of a **weak class**, declared
   as {green_roof, drivhus, betonflade, brosten} [confirmed by the author at lock as the declared weak-class set], and 16 from
   the complement. Sampling without replacement, by the seed, script-drawn, list persisted before
   any image is rendered.
3. **Cells scored: 2** — `unet_resnet34_rgb` (production) and `convnext_upernet_rgb` (best)
   [confirmed by the author at lock: two cells, these two, per Plan 3.1 decision D]. 192 scoring items (96 tiles × 2 cells).
4. **Blinding:** for each item the scorer sees the rgb tile, the ground-truth overlay, and the
   prediction overlay, with the cell identity hidden and item order shuffled by the seed. The
   VM builds the panels and the scoring sheet; the mapping item → (tile, cell) is revealed only
   after all scoring is complete.
5. **Closed failure taxonomy — scored present/absent per item; no new modes may be added after
   lock (anything else goes to mode 9's one-line note):**
   1. salt-and-pepper speckle (isolated single-class pixels/specks);
   2. boundary bleeding (class runs across a visible surface edge);
   3. spurious minority-class patch (a false-positive island of a class absent or negligible in
      the tile's ground truth — the D3 blindness catcher);
   4. paved-class swap (asfalt/fliser/grus/brosten/betonflade confusion over paved ground);
   5. road/driveway continuation error (carriageway class extended into entrances, or cut);
   6. vegetation-occlusion error (surface under canopy mislabelled — leaf-on rgb makes this a
      declared candidate mode);
   7. misregistration echo (prediction pattern visibly offset from image structure);
   8. large-object failure (solar farm, greenhouse, green roof missed, fragmented or
      hallucinated);
   9. other (one-line free note).
   Per item, additionally one **dominant-mode tag** (the single mode judged most consequential).
6. **Self-consistency:** 12 tiles (both cells, 24 items, 12.5 %) re-scored in a second shuffled
   pass at least 24 h later. Reported per mode as raw percent agreement, plus Cohen's κ where
   prevalence permits a stable estimate; reported as-is, never re-scored to improve.
7. **Outputs:** incidence table (mode × cell, over n = 96 each) with per-mode counts and
   proportions; the self-consistency table; and one exemplar panel per mode chosen by fixed rule
   — the first tile in sampled order exhibiting the mode in the ConvNeXt cell, falling back to
   the production cell — never hand-picked.
8. **Operational-consequence mapping, fixed now:** speckle and spurious patches → false
   imperviousness change signals and noise in parcel-level statistics; boundary bleeding → systematic area
   misestimation at parcel scale; paved-class swaps → wrong surface-type input to drainage
   coefficients; road/driveway and occlusion errors → connectivity and access-surface errors in
   municipal use; misregistration echo → geometric distrust of the product; large-object failure
   → wrong classification of exactly the installations (solar, greenhouse, green roof) policy
   monitoring targets. Each incidence count is reported with its mapped consequence; no
   consequence may be attached post hoc to a mode not listed here.
9. **Deviation rule:** any departure from this protocol is recorded in a dated amendment before
   the results are used.

## 6. Machine-readable block (consumed verbatim by the Part B runner)

```yaml
declaration_version: 2026-08-25
status: LOCKED            # runner must refuse to execute unless this reads LOCKED
alpha: 0.05
sidedness: two-sided
test: wilcoxon_signed_rank_exact
zero_handling: drop      # standard; count reported as n_zeros
near_tie_delta: 0.001    # sensitivity B only
route_set_primary: all16
route_min_tiles_sensitivity: 200   # drops 83-34, 85-48
per_route_metric:
  name: macro_iou_present_classes
  presence_rule: support_pixels_gt_0    # verified against eda_route_cell_metrics.py at lock
  ignore_index: 0
  report_only: [unknown2]
families:
  F1_model_ranking_rgb_weighted:
    - [convnext_upernet_rgb, swin_upernet_rgb]
    - [convnext_upernet_rgb, segformer_b1_rgb]
    - [convnext_upernet_rgb, unet_resnet34_rgb]
    - [swin_upernet_rgb, segformer_b1_rgb]
    - [swin_upernet_rgb, unet_resnet34_rgb]
    - [segformer_b1_rgb, unet_resnet34_rgb]
  F2_channel_within_convnext_weighted:
    - [convnext_upernet_rgb, convnext_upernet_6ch]
    - [convnext_upernet_rgb, convnext_upernet_10ch]
    - [convnext_upernet_6ch, convnext_upernet_10ch]
  F3_weighting_within_convnext_rgb:
    - [convnext_upernet_rgb, convnext_upernet_rgb_unw]
correction: holm_within_family
friedman:
  families: [F1_model_ranking_rgb_weighted, F2_channel_within_convnext_weighted]
  report: [statistic, p, mean_ranks, kendalls_w]
  role: omnibus_context      # fail-to-reject downgrades the family narrative to descriptive
sign_test:
  role: robustness_companion # never confirmatory, uncorrected
  zero_handling: drop
  near_tie_sensitivity: true # same delta as near_tie_delta
excluded_tests: [secondary_metric_significance, per_class_significance, tost_equivalence,
                 bayesian_reanalysis]
report_per_pair: [p_exact, n_prime, n_zeros, n_near_ties, median_paired_diff, diff_ci95,
                  rank_biserial, holm_adjusted_p, sign_wins, sign_p_exact,
                  sensitivity_A, sensitivity_B]
descriptive_cells: [convnext_upernet_rgb_ndsm,
                    convnext_upernet_6ch_corrected, swin_upernet_6ch_corrected,
                    segformer_b1_6ch_corrected, unet_resnet34_6ch_corrected,
                    convnext_upernet_ortorgb, swin_upernet_ortorgb,
                    segformer_b1_ortorgb, unet_resnet34_ortorgb,
                    convnext_upernet_rgb_dsm_dtm_corrected, swin_upernet_rgb_dsm_dtm_corrected,
                    segformer_b1_rgb_dsm_dtm_corrected, unet_resnet34_rgb_dsm_dtm_corrected]
descriptive_contrasts:
  - [convnext_upernet_rgb_ndsm, convnext_upernet_rgb]
  - [convnext_upernet_6ch_corrected, convnext_upernet_6ch]
  - [convnext_upernet_6ch_corrected, convnext_upernet_rgb]
  - [swin_upernet_6ch_corrected, swin_upernet_6ch]
  - [swin_upernet_6ch_corrected, swin_upernet_rgb]
  - [segformer_b1_6ch_corrected, segformer_b1_6ch]
  - [segformer_b1_6ch_corrected, segformer_b1_rgb]
  - [unet_resnet34_6ch_corrected, unet_resnet34_6ch]
  - [unet_resnet34_6ch_corrected, unet_resnet34_rgb]
  - [convnext_upernet_ortorgb, convnext_upernet_rgb]
  - [swin_upernet_ortorgb, swin_upernet_rgb]
  - [segformer_b1_ortorgb, segformer_b1_rgb]
  - [unet_resnet34_ortorgb, unet_resnet34_rgb]
  - [convnext_upernet_rgb_dsm_dtm_corrected, convnext_upernet_6ch_corrected]
  - [convnext_upernet_rgb_dsm_dtm_corrected, convnext_upernet_rgb]
  - [swin_upernet_rgb_dsm_dtm_corrected, swin_upernet_6ch_corrected]
  - [swin_upernet_rgb_dsm_dtm_corrected, swin_upernet_rgb]
  - [segformer_b1_rgb_dsm_dtm_corrected, segformer_b1_6ch_corrected]
  - [segformer_b1_rgb_dsm_dtm_corrected, segformer_b1_rgb]
  - [unet_resnet34_rgb_dsm_dtm_corrected, unet_resnet34_6ch_corrected]
  - [unet_resnet34_rgb_dsm_dtm_corrected, unet_resnet34_rgb]
bootstrap:
  B: 10000
  seed: 20260825
  unit: route
  n_resample: 16
  recompute: sum_confusion_matrices   # never average route scores
  ci: percentile_95
  paired: same_resample_both_cells
mcnemar: {pairs: families_only, role: context_only}
qualitative:
  seed: 20260825
  n_tiles: 96
  per_fold: 32
  strata: {weak_present: 16, complement: 16}
  weak_classes: [green_roof, drivhus, betonflade, brosten]
  cells: [unet_resnet34_rgb, convnext_upernet_rgb]
  double_scored_tiles: 12
  taxonomy_modes: 9
```

## 7. Lock procedure

1. The author reads the whole document, resolves the three [AUTHOR CONFIRM] items (D3 presence
   threshold against the code; D4 stratum set; D4 sample/cell/double-scoring numbers), and edits
   anything else at will — this draft binds nobody until signed.
2. The author sets the header to `Status: LOCKED, 2026-08-25 <time>` and saves the file into
   `prompts/` (log-of-record) and `c:\thesis\exploratory_data_analysis\` (runner input), sets
   `status: LOCKED` in the YAML block, and tells the VM session: "declarations locked, go".
3. From that moment: Part B may execute; arm scores may be read; the qualitative sampling script
   may run. Plan 3.1 decisions B–E are confirmed by this same act unless the author states
   otherwise.

*Drafted 2026-08-24 under Great Plan 3.1 §5.1 and §11. Amended the same evening on the author's
robustness query: Friedman omnibus + Kendall's W added for the multi-pair families (Demšar 2006
protocol), the exact sign test added as the declared robustness companion, and the declared
exclusions recorded (no secondary-metric or per-class significance, no TOST, no Bayesian
re-analysis). Amended again 25/8 morning, still before lock: the descriptive cell set and
contrast list expanded to the full four-model roster per Plan 3.1 §11.5 (Option 2). Sources:
tracker §4 (recommendations), handoff §9.1 (framing rule), SQ1 F12 and the E6 cache (route
structure), channel-axis findings §6 (the D3 worked example). No inferential quantity existed
when this was written.*
