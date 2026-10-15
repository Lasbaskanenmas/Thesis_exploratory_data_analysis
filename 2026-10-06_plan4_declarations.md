# 2026-10-06 — Plan 4 declarations: the five final arms and the production pick (P1 to P5)

**Status: LOCKED, 2026-10-06 05:45. To be locked by the author before the queue is launched.**

**Author: Lasse. Drafted by the VM session from Great Plan 4 Phase 1, whose configurations were
narrowed with the author one question at a time on 2026-10-06. The author owns every choice.**

**Role.** These declarations fix, in writing and in advance, every degree of freedom in the last
experiment of the project: which cells run, how they are configured, and the rule that turns their
scores into one production model for KDS. The pick script
(`scripts/plan4_pick_production.py`) reads the machine-readable block in section 6 verbatim and
refuses to judge any cell unless that block reads `status: LOCKED`.

## 0. Transparency clause — what has and has not been seen

- **Seen before this declaration:** every all-tiles pooled score and per-fold diagnostic of the
  37 scored cells; the clean (18,033-tile) pooled scores of 28 of them
  (`contamination_decision/task2_rescore_table.csv`); every Part B quantity, which is all-tiles,
  including the descriptive paired CIs. In particular, `convnext_upernet_rgb_ndsm` minus
  `convnext_upernet_rgb` is +0.0039 pooled all-tiles, 95% CI [-0.0050, +0.0184], route
  Wilcoxon p = 0.298. That number was known when the pick rule (P2) was chosen, and is named here
  for that reason.
- **Not seen, and not in existence:** any score of the five new cells, none of which has been
  trained; any CLEAN-score confidence interval for any pair of cells.
- Therefore the rule precedes every score of every new cell, and every clean CI of every cell.

## 1. P1 — The five new cells

All ConvNeXt-base + UPerNet (V1), weighted cross-entropy with the locked effective-number vector,
frozen route-blocked 3-fold split, 10 epochs, bs 4, lr 0.002, fit_one_cycle, the 5-transform set,
bf16, TF32 off. Every protocol line is byte-identical to the frozen `convnext_upernet_rgb_fold<k>.ini`
except the channel block and `job_name` (built by `scripts/plan4_build_configs.py`; checked by
`scripts/gate_plan4.py` G-diff). Queue order is the order below.

| # | cell | datatypes | channels | n_in | one change from |
|---|---|---|---|---:|---|
| 1 | `convnext_upernet_ortorgb_ndsm` | OrtoRGB, nDSM | [0,1,2], [0] | 4 | `ortorgb` (+nDSM) |
| 2 | `convnext_upernet_cir` | cir | [0,1,2] = NIR, R, G | 3 | `rgb` (blue -> NIR) |
| 3 | `convnext_upernet_ortocir` | OrtoCIR | [0,1,2] = NIR, R, G | 3 | `ortorgb` (blue -> NIR) |
| 4 | `convnext_upernet_ortocir_ndsm` | OrtoCIR, nDSM | [0,1,2], [0] | 4 | `ortocir` (+nDSM) |
| 5 | `convnext_upernet_cir_ndsm` | cir, nDSM | [0,1,2], [0] | 4 | `cir` (+nDSM) |

Constants: ImageNet `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` for every 3-band image; nDSM
`0.010076483036168348` / `0.022261953346304587`, verbatim from `convnext_upernet_rgb_ndsm`. Together
with the existing `rgb`, `rgb_ndsm` and `ortorgb` they fill a 2x2x2 design: base product
{skraafoto, orthophoto} x colour {RGB, CIR} x nDSM {no, yes}.

**All five are descriptive, outside every Holm family**, exactly as every arm was under D1 of the
2026-08-25 declaration. No new significance test is declared on them. Their only formal role is the
production pick, P2.

## 2. P2 — The production pick rule

**Incumbent:** `convnext_upernet_rgb`, the winner of declared family F1 and KDS's own input.

**A challenger replaces the incumbent only if both hold, on CLEAN scores** (the 18,033 tiles not in
`contamination_decision/raw/contaminated_tiles.txt`):

1. its clean pooled Macro-IoU minus the incumbent's is greater than zero, and
2. the 95% route-paired bootstrap CI of that difference excludes zero.

The CI is `bootstrap_paired_pooled_diff` (`scripts/partb_statistics.py`): whole routes resampled with
replacement, the same resample applied to both cells, pooled Macro-IoU recomputed from the
resampled routes' summed confusion matrices. B, seed and ci are inherited unchanged from the locked
2026-08-25 block: B = 10000, seed 20260825, ci 0.95. The per-route confusion matrices are built from
the clean tiles only.

**Candidates:** every weighted ConvNeXt cell scored by the pick time. That is six existing
challengers (`rgb_ndsm`, `ortorgb`, `rgb_dsm_dtm_corrected`, `10ch`, `6ch`, `6ch_corrected`) plus the
five of P1.

**If several challengers qualify:** take the one needing the fewest input products (P3), then the
higher clean pooled Macro-IoU. **If none qualifies, the incumbent stays.** The verdict is written to
`results/tables/plan4/pick_verdict.json` with every candidate's difference and CI, whatever the
outcome.

## 3. P3 — Input products, the tie-break unit

The number of distinct source data products a cell needs at inference time. nDSM counts as its two
sources, DSM and DTM.

| cells | products | count |
|---|---|---:|
| `rgb`, `cir`, `ortorgb`, `ortocir` | one image product | 1 |
| `rgb_ndsm`, `cir_ndsm`, `ortorgb_ndsm`, `ortocir_ndsm`, `rgb_dsm_dtm_corrected` | image + DSM + DTM | 3 |
| `6ch`, `6ch_corrected` | rgb + cir + DSM + DTM | 4 |
| `10ch` | rgb + cir + OrtoRGB + OrtoCIR + DSM + DTM | 6 |

## 4. P4 — Schedule cut-off and drop rules

- **The production retrain must START by Mon 12/10 12:00** (it takes about 10 h; the VM must be
  empty by Tue 13/10 end of day).
- A failed arm is re-run once, and only if the re-run can finish before that time. Otherwise it is
  excluded and reported as initiated-not-complete.
- A cell not scored when the pick runs is excluded from the pick and reported as such.
- The pick runs once. It is not re-run to include a late cell.

## 5. P5 — Recorded deviation from Great Plan 3.1 section 11.5

The full-roster rule ("every arm intervention is replicated across the entire declared roster") is
**knowingly not applied**. Replicating the five arms on resnet34, Swin and SegFormer would cost about
250 GPU-h (5 x (14.8 + 22.4 + 11.3 + inference)), more than the window holds. The arms are ConvNeXt
only, at one seed, like every arm before them. Any claim about the new cells is a claim about
ConvNeXt.

## 6. Machine-readable block (consumed verbatim by the pick script)

```yaml
declaration_version: 2026-10-06
status: LOCKED            # pick script refuses to judge any cell unless this reads LOCKED
inherits_from: 2026-08-25_pre_declarations.md

incumbent: convnext_upernet_rgb
new_cells:
  - convnext_upernet_ortorgb_ndsm
  - convnext_upernet_cir
  - convnext_upernet_ortocir
  - convnext_upernet_ortocir_ndsm
  - convnext_upernet_cir_ndsm
existing_challengers:
  - convnext_upernet_rgb_ndsm
  - convnext_upernet_ortorgb
  - convnext_upernet_rgb_dsm_dtm_corrected
  - convnext_upernet_10ch
  - convnext_upernet_6ch
  - convnext_upernet_6ch_corrected

statistic: paired_route_block_bootstrap_of_clean_pooled_macro_iou_difference
clean_tiles_exclude: ../logs_and_models/contamination_decision/raw/contaminated_tiles.txt
expected_clean_tiles: 18033
B: 10000
seed: 20260825
ci: 0.95
replace_if: diff_gt_zero_and_ci_excludes_zero
tie_break: [fewest_input_products, higher_clean_macro_iou]

input_products:
  convnext_upernet_rgb: 1
  convnext_upernet_cir: 1
  convnext_upernet_ortorgb: 1
  convnext_upernet_ortocir: 1
  convnext_upernet_rgb_ndsm: 3
  convnext_upernet_cir_ndsm: 3
  convnext_upernet_ortorgb_ndsm: 3
  convnext_upernet_ortocir_ndsm: 3
  convnext_upernet_rgb_dsm_dtm_corrected: 3
  convnext_upernet_6ch: 4
  convnext_upernet_6ch_corrected: 4
  convnext_upernet_10ch: 6

production_start_deadline: "2026-10-12T12:00"   # AMENDED 2026-10-09, see section 8: ~2026-10-13T00:00
pick_runs_once: true
```

## 7. Lock procedure

1. The author reads sections 0 to 5 and amends anything.
2. The author sets the header to `Status: LOCKED, 2026-10-06 <time>` and `status: LOCKED` in the
   YAML block, then saves.
3. Only then is the queue launched. The pick script checks the YAML before computing anything on
   clean scores; the machinery acceptance test (reproducing already-published all-tiles Part B
   numbers) is exempt because it reveals nothing new.

## 8. Amendment, 2026-10-09 04:30 (author decision, before any new cell was scored)

**What happened.** `convnext_upernet_ortorgb_ndsm` fold 2 froze at epoch 3, batch 962 of 3,219, on
Wed 07/10 12:23 and was found frozen on Fri 09/10 04:14. The process was alive, holding 22 GB at 0 %
GPU, with no file written in about 40 h. Folds 0 and 1 completed normally (6,439 and 6,437
predictions). Probable cause: QuickEdit mode pausing the console. Not yet confirmed.

**What changes.** The author keeps all five arms rather than invoking the P4 drop rule. The
production start deadline of P4 moves from Mon 12/10 12:00 to **about Tue 13/10 00:00**, which puts the
production retrain at about Tue 13/10 10:00 and leaves the rest of Tuesday for the E6 sweep and the HF
push. Nothing else in P1 to P5 changes: same cells, same order, same pick rule, and the pick still
runs once.

**Seen at the time of this amendment.** No score of any new cell exists. `ortorgb_ndsm`'s fold 0
and fold 1 rasters are on disk, but the cell is not scored until fold 2 lands. The training log of
fold 2's first two epochs shows only the running valid accuracy that fastai prints (0.873, 0.898),
not a pooled score.
