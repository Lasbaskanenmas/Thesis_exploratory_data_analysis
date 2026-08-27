# 2026-08-24 — Work order: arms G + A, CPU backlog, Part B build (VM, fresh Claude Code session)

You are a fresh Claude Code session on the thesis VM (H6-GPU-WIN-03). You have no memory of prior
sessions. This work order is your complete instruction; it was written under The Great Plan 3.1
(2026-08-24) before execution, per project practice. Save this file verbatim to
`c:\thesis\exploratory_data_analysis\results\findings\2026-08-24_work_order_arms_G_A.md` as your
first action, then follow it top to bottom.

## Read first (all on this machine)

1. `c:\thesis\exploratory_data_analysis\results\findings\2026-08-15_stage1_ndsm_arms.md` — the arms
   machinery, gates, measured constants, operating precedent.
2. `...\findings\2026-08-19_channel_axis_findings.md` — the channel verdict the new arms extend.
3. `...\findings\2026-08-23_learning_curve_launch_block.md` — launch-block format and scoring
   pattern to imitate.
4. `c:\thesis\ML_sdfi_fastai2\configs\matrix_configs\MANIFEST.md` and `MANIFEST_arms_2026_08.md` —
   config conventions. The frozen 72+72 configs and the frozen split are LOCKED.

## Binding rules

- **You never start GPU jobs.** You prepare configs, run cheap CPU gates, and print launch blocks;
  the author pastes them. Exception: if the author writes "you may launch" in this session, you may
  run the printed blocks yourself, in the stated order, and must say so in the end-of-run record.
- Additive only. Never delete, move, rename or truncate. `logs_and_models\` is written only by the
  training/inference the author launches. Never regenerate the frozen split or frozen configs;
  `validate_matrix_configs.py` must keep asserting the frozen 72+72 unchanged.
- **No inferential statistics may be executed** (nothing that produces a p-value on real data)
  until the author drops the locked `2026-08-25_pre_declarations.md` into
  `exploratory_data_analysis\` and says go. Building and self-testing the machinery on synthetic
  data is required (Task 3); running it on real scores is forbidden until then.
- Never print or copy the HF token (`c:\thesis\hftoken_write.txt`). Windows PowerShell 5.1: no
  `&&`; chain with `; if ($?) { }`. Do not touch `_monitor.py` / `_monitor_state.json`. Never parse
  the route audit's terminal scrollback; read its CSV/JSON.
- Terminology in anything you write: `OrtoRGB`/`OrtoCIR` are the **spring leaf-off orthophoto**;
  `rgb`/`cir` are the **skråfoto-programme nadir product** (leaf-on, ~3-year cadence). All four are
  geometrically nadir. The label "oblique source" is retired (correction note of 2026-08-24).
- Disclose every deviation and exception in the end-of-run record. Finish with a manifest of every
  file you created.

## Task 1 — build and gate two GPU arms (highest priority, do this first)

Both arms: ConvNeXt+UPerNet (trainer `train.py` — resnet-style trainers cannot build UPerNet),
weighted CE with the locked 11-element class-weight vector, frozen split, fresh train + inference
configs under new filenames only, pure bf16 flags exactly as the frozen matrix (`to_bf16` true,
`tf32` false, `to_fp16` false, `cudnn_benchmark` true, `pin_memory` true), 10 epochs, batch and lr
as the frozen convnext cells. Extend `MANIFEST_arms_2026_08.md` additively (or add a dated sibling
manifest) and a `run_arms_G_A.cmd` in the configs directory.

**Arm G (first in the GPU queue): `convnext_upernet_ortorgb`** — 3 channels from the OrtoRGB
source. **Normalisation: the ImageNet vectors copied programmatically from the frozen
`convnext_upernet_rgb` configs — deliberately NOT measured Orto constants.** The cell exists to
swap exactly one thing against the frozen rgb cell (the image source); identical normalisation
treatment is part of the design. Do not "improve" this. (Under ImageNet constants the Orto bands
sit at effective std ≈ 0.91 — healthy, per the EDA channel audit.)

**Arm A (second): `convnext_upernet_rgb_dsm_dtm_corrected`** — 5 channels: RGB at ImageNet
constants + DSM + DTM at the **measured corrected constants, loaded from the same measured-stats
JSONs the `6ch_corrected` configs load** (never transcribed by hand). **No CIR band.** The cell
isolates corrected absolute elevation without the simultaneous NIR change that `6ch_corrected`
bundled; its purpose is to test whether full-strength corrected DSM/DTM alone reproduces the
fold-0 collapse and the betonflade zero.

**Gates, all before anything is launched (CPU except the smoke):**

1. `validate_matrix_configs.py` extended to assert the new configs (n_in 3 and 5, correct pools,
   locked weight vector, bf16 flags) while still asserting frozen 72+72 unchanged and the existing
   arms and learning-curve configs intact. PASS required.
2. G-A forward gates at the real geometry: `1×3×1000×1000` and `1×5×1000×1000 → 1×11×1000×1000`
   through the actual ConvNeXt+UPerNet build path. n_in=5 has never been tested; it must pass.
3. Split dry-run: clean partition, 6,439/6,437/6,438, no route leakage.
4. Dataloader sanity for G: load a few batches from the OrtoRGB folder, assert shapes, dtype, and
   per-band post-normalisation statistics in a sane range; confirm all 19,314 tiles resolve.
5. **G-D 1-epoch smoke for arm A only** (new width + new constants): prepare
   `smoke_rgb_dsm_dtm_corrected_convnext.ini`. This is a GPU job, so it goes into the launch block,
   not run by you. Arm G needs no smoke (standard 3-channel width; gates 1–4 cover it).

**Launch blocks — print both fully expanded for cmd.exe, in the learning-curve block's format, and
stop there:**

- **Block 1 (author pastes tonight):** the A smoke (1 epoch, ~40–60 min) first, then G training
  fold 0, 1, 2 (~7–9 h each), then G inference fold 0, 1, 2. Smoke-first means the author can
  review the smoke log on Tuesday while G still trains, long before A's training would start.
- **Block 2 (author pastes after confirming the smoke: finite losses, sane epoch time):** A
  training fold 0, 1, 2, then A inference fold 0, 1, 2.
- State the expected wall-clock for each block and the smoke pass-criteria in one line each.

**After both arms' inference exists (later this week, when the author says so):** score each cell
with `pooled_oof_metrics.py` per the handoff template into `oof_convnext_upernet_ortorgb` and
`oof_convnext_upernet_rgb_dsm_dtm_corrected`; append both rows **additively** to
`cross_cell_summary.csv` (27 → 29 rows); run E6 for both cells and validate element by element
against their pooled matrices; HF-backup the six final `.pth` and the two oof JSONs. Both cells are
**descriptive, outside every Holm family** — no significance tests on them, ever, and no reading
of their scores against the frozen cells in writing until the author confirms the pre-declarations
are locked.

## Task 2 — CPU backlog, run tonight while the GPU trains

1. Extract the frozen `unet_resnet34_rgb` **fold-0 per-class IoUs** from its
   `pooled_oof_metrics.json` `per_fold_diagnostic` into
   `exploratory_data_analysis\results\tables\learning_curve\lc100_per_class.json`, so the
   per-class learning curves have their 100 % points.
2. **Run E4** (label quality: origin of the 37 % ignore mass + boundary-error profile). The script
   exists and is self-tested; it has never been run. Its outputs are cited by the SQ4 limitation
   text, so this is required, not optional.
3. **The 8 tiles with nDSM > 100 m** (max 298.33 m; list derivable from the nDSM stats artifacts):
   timebox one hour, inspect each against its DSM/DTM/rgb, classify (DSM spike, residual
   misregistration, real structure), write a short dated findings note either way.
4. **HF sync batch:** the 21 missing per-run log CSVs (idempotent re-run of `--sync_all`; apply the
   list-then-batch fix from handoff §10.2 if trivial), plus the learning-curve tables, the E6
   route tables, `route_cell_provenance.json`, and the current `cross_cell_summary.csv`.
5. **Metric breadth table** over the current 27 cells (extend to 29 when the arms land): per-class
   precision and recall, macro-F1, frequency-weighted IoU, overall accuracy, the completed
   **worst-case metric** exactly as defined in the source it is cited under (currently
   half-implemented), and one derived row per cell collapsing the pooled confusion matrix to
   **binary befæstet/ubefæstet** (sealed = asfalt, fliser, grus, betonflade, brosten, solceller,
   drivhus, green_roof; unsealed = ubefestet — state the mapping in the table header and flag it
   for the author's confirmation). One artifact CSV, regenerable by script.
6. **Boundary IoU** (Cheng et al. 2021 definition) on exactly four cells: `unet_resnet34_rgb`,
   `convnext_upernet_rgb`, `convnext_upernet_rgb_ndsm`, `convnext_upernet_6ch_corrected`.
   Implement, self-test on synthetic shapes, then run as an overnight CPU batch. Timeboxed: if the
   implementation or runtime fights back, stop, report, and mark it dropped-first per plan §4.6.

## Task 3 — build the Part B statistics runner (build and self-test only; DO NOT run on real data)

One script that consumes (a) the per-route score cache (`route_cell_metrics.csv` / the 27-cell E6
outputs) and (b) a pre-declaration file the author will provide (`2026-08-25_pre_declarations.md`
with a machine-readable block: families and pairs, primary/sensitivity route rules, per-route
metric, bootstrap B and unit). It must produce, per declared pair: Wilcoxon signed-rank (exact
where N permits) with the **exact-tie count and effective N reported beside every p-value**;
the exact two-sided sign test as the declared robustness companion; Holm–Bonferroni within each
declared family; per multi-pair family a Friedman omnibus with mean ranks and Kendall's W;
whole-route block-bootstrap CIs for pooled Macro-IoU
of declared cells; effect sizes (median paired route difference with CI; rank-biserial secondary);
McNemar on pooled pixels as context only. **The declaration file's YAML block is authoritative
over this sentence wherever they differ.** Self-test on synthetic data with known answers,
including a forced all-tie route. Then stop. It runs on real data only after the author drops the
locked declaration file and says go.

## End of session

Write `2026-08-24_arms_G_A_build_record.md` in the findings folder: what was built, every gate
result verbatim, the two launch blocks, Task 2 outcomes, Task 3 self-test evidence, deviations,
and the file manifest. Tell the author exactly what to paste and when.
