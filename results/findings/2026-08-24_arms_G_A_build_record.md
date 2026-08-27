# 2026-08-24 — Build record: arms G + A, CPU backlog, Part B runner

**Executes `2026-08-24_work_order_arms_G_A.md` (saved verbatim to this folder as the first action).
Additive only. No GPU job was launched — the author launches all training. No inferential statistic
was computed on real scores.** Great Plan 3.1 §4.8 (arm G) and §4.2 (arm A).

---

## 0. What the author needs to do, and when

| when | what |
|---|---|
| **tonight** | paste **Launch Block 1** (§3). Smoke first, then arm G. ~21.5 h. |
| **Tuesday morning** | read the smoke log against the one-line pass criteria in §3, then paste **Launch Block 2** (~26 h). Both blocks finish well inside the Fri 28/8 kill-by. |
| **before the first arm score is read** | lock `2026-08-25_pre_declarations.md`. The Part B runner refuses to compute anything until it exists with a non-null `locked_utc` (§5). |
| **decide** | the 9 unbacked model checkpoints, 10.2 GB (§4.4) — the only thing in this session I stopped short of doing. |
| **confirm** | the binary sealed/unsealed class mapping (§4.5). |
| **read** | §6 — the lc100 per-class points change two sentences of Plan 3.1 §2. |

---

## 1. What was built

### Configs (additive; the frozen 144 and the 2026-08-15 arms are untouched)

Emitted by a new `emit_arms_G_A()` path in `generate_matrix_configs.py`, guarded behind
`--arms-G-A-2026-08-24` and refusing to overwrite any existing file. It reuses the same
`train_config()` / `infer_config()` writers that produced the frozen matrix, so the new cells are
structurally identical to the frozen ones by construction rather than by inspection.

| cell | n_in | datatypes | trainer | constants |
|---|---:|---|---|---|
| `convnext_upernet_ortorgb` | 3 | `["OrtoRGB"]` | `train.py` | ImageNet, **parsed out of `train/convnext_upernet_rgb_fold0.ini`** |
| `convnext_upernet_rgb_dsm_dtm_corrected` | 5 | `["rgb","DSM","DTM"]` | `train.py` | ImageNet RGB + measured corrected DSM/DTM, **loaded from `corrected_channel_constants.json`** |

Neither constant vector is transcribed by hand. Arm G's is read from the frozen config it is meant
to differ from; arm A's is read from the same artifact the `6ch_corrected` configs load, and
generation asserts it is byte-identical to that file's `means_6ch_corrected[4:]` / `stds_6ch_corrected[4:]`.

**Arm G differs from the frozen `convnext_upernet_rgb_fold0.ini` in exactly two lines**, verified by
diff:

```
datatypes = ["OrtoRGB"]                        vs   datatypes = ["rgb"]
job_name = "convnext_upernet_ortorgb_fold0"    vs   job_name = "convnext_upernet_rgb_fold0"
```

Everything else — bf16 flags, batch 4, lr 0.002, 10 epochs, transforms, the locked 11-element weight
vector, the frozen `fold_0_valid.txt` — is byte-identical. That is what "the swap changes exactly
one thing" means here, and it is now a checkable property rather than a claim.

**Note on `path_to_images`.** It stays `splitted/rgb` in arm G's *training* config, exactly as it
does in every frozen 6ch and 10ch config. `load_all_datasources_for_image` resolves each band as
`<item>.parent.parent / <datatype> / <name>`, so the leaf folder of the item path never affects
which pixels are read — `datatypes` alone does. Inference builds its item list as
`benchmark_folder / datatypes[0] / name` and therefore reads `splitted/OrtoRGB` directly. Both paths
load OrtoRGB pixels; gate G-E below measures that rather than trusting it.

### Files created

```
configs/matrix_configs/train/convnext_upernet_ortorgb_fold{0,1,2}.ini
configs/matrix_configs/train/convnext_upernet_rgb_dsm_dtm_corrected_fold{0,1,2}.ini
configs/matrix_configs/infer/infer_convnext_upernet_ortorgb_fold{0,1,2}.ini
configs/matrix_configs/infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold{0,1,2}.ini
configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_convnext.ini
configs/matrix_configs/MANIFEST_arms_G_A_2026_08_24.md
configs/matrix_configs/run_arms_G_A.cmd
```

---

## 2. Gate results, verbatim

### Gate 1 — `validate_matrix_configs.py`, extended

Extended with the two new tags (`ortorgb` n_in 3, `rgb_dsm_dtm_corrected` n_in 5), a datatype
assertion per tag, and a constants assertion per tag: arm G must carry the ImageNet vectors, arm A
must carry ImageNet RGB plus the measured corrected DSM/DTM read back from the JSON at validation
time. The frozen 72+72 assertion is unchanged and still fires.

```
OK: validated 105 training + 105 inference configs (210 total).
  - frozen 72-run matrix  : 72 train + 72 infer (unchanged)
  - additive 2026-08 arms : 30 train + 30 infer
  - learning curve (3.0 s4): 3 train + 3 infer (whole-route subsets, held-out fold unchanged)
  - n_in matches channel tag (rgb=3 / 6ch=6 / 10ch=10 / rgb_ndsm=4 / 6ch_corrected=6 / ortorgb=3 / rgb_dsm_dtm_corrected=5)
  - ortorgb carries the frozen rgb cell's ImageNet constants (one variable: the source)
  - rgb_dsm_dtm_corrected carries measured corrected DSM/DTM + ImageNet RGB
  - each reads its fold_<f>_valid.txt; train keeps all.txt as the pool
  - training configs carry the locked 11-value weighted-CE vector + cross_entropy
  - pure bf16 (tf32 OFF); to_bf16/cudnn_benchmark/pin_memory true, to_fp16 false
  - inference model_to_load + output_folder match the cell's fold-model
```

### Gate 2 — G-A forward pass at the real geometry (`check_n_in_3_and_5.py`)

n_in=5 had never been exercised by this project. It passes.

```
=== G-A (2026-08-24 arms): ConvNeXt-base+UPerNet forward at 1000x1000 ===
  [PASS     ] n_in=3  1x3x1000x1000 -> expect 1x11x1000x1000
              arm G  ortorgb                (positive control: untouched 3-band stem)
              got (1, 11, 1000, 1000)
  [PASS     ] n_in=5  1x5x1000x1000 -> expect 1x11x1000x1000
              arm A  rgb_dsm_dtm_corrected  (NEW WIDTH -- never exercised before)
              got (1, 11, 1000, 1000)

ALL PASS
```

### Gates 3 and 4 — G-C split dry-run and G-E arm-G dataloader (`gate_arms_G_A.py`)

```
=== G-C: frozen split dry-run (read-only; the split is LOCKED) ===
  [PASS] all.txt active tiles = 19314  --  got 19314
  [PASS] fold 0 held-out count = 6439  --  got 6439
  [PASS] fold 1 held-out count = 6437  --  got 6437
  [PASS] fold 2 held-out count = 6438  --  got 6438
  [PASS] fold 0/1/2 have no duplicate tiles
  [PASS] folds 0/1, 0/2, 1/2 disjoint  --  0 shared
  [PASS] folds partition all.txt exactly  --  union 19314 vs active 19314
  [PASS] no route straddles folds (16 routes)  --  {}
  [PASS] fold_assignment.csv agrees with the fold_*_valid.txt lists (16 routes)  --  {}

=== G-E: arm G (`ortorgb`) dataloader sanity ===
  [PASS] all 19314 active tiles resolve in splitted/OrtoRGB  --  0 missing
  [PASS] config reports n_in = 3        [PASS] datatypes = ['OrtoRGB']
  [PASS] batch shape is (N, 3, 1000, 1000)  --  (12, 3, 1000, 1000)
  [PASS] batch dtype is float32   [PASS] labels are integer class codes
  [PASS] no NaN / inf in the batch

  12 tiles from splitted/OrtoRGB, post-normalisation (ImageNet constants, as the frozen rgb cell):
    band              min      max     mean      std     implied raw/255 mean
    OrtoRGB_R     -1.8953   2.2489   0.0556   0.8054                 0.4977
    OrtoRGB_G     -1.7031   2.4286   0.1887   0.7755                 0.4983
    OrtoRGB_B     -1.5430   2.6400   0.3802   0.7112                 0.4915
  [PASS] post-normalisation range within [-2.5, 3.0]  --  [-1.895, 2.640]
  [PASS] every band: effective std in [0.5, 1.5], implied raw/255 mean in [0.1, 0.9]

ALL GATES PASS
```

The per-band effective std of 0.71–0.81 on this 12-tile sample sits a little below the ~0.91 the
full-pool channel audit reports for the Orto bands under ImageNet constants, which is what a
12-tile sample should do. It is healthy, and it is the whole reason arm G does not need measured
Orto constants.

**No G-D smoke was run for arm G**, per the work order: standard 3-band width, and gates 1–4 cover
it. Arm A's smoke is a GPU job and therefore sits in Launch Block 1 for the author.

---

## 3. The two launch blocks

Fresh Command Prompt, then:

```bat
cd /d c:\thesis\ML_sdfi_fastai2
```

### Block 1 — paste tonight. Smoke, then arm G. **Expected ~21.5 h.**

Smoke first so the log can be read Tuesday morning while arm G is still training, long before arm A
would start.

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_convnext.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_ortorgb_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_ortorgb_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_ortorgb_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_ortorgb_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_ortorgb_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_ortorgb_fold2.ini
```

**Smoke pass criteria, one line:** finite `train_loss` near 0.15–0.25 and finite `valid_loss` with no
divergence, `valid_accuracy` roughly 0.80–0.87 at one epoch (below the 0.8747 trivial floor is normal
at one epoch), and an epoch time near 50 min — anything above ~75 min means the two float32 elevation
rasters are costing more than the 6ch cell did and the block-2 estimate needs revising.

**Timing basis, measured from the existing per-epoch logs rather than assumed:** convnext `rgb`
39:49/epoch, `rgb_ndsm` 40:14, `6ch_corrected` 58:31–60:20. Arm G is a 3-band uint8 cell like `rgb`
→ ~6.6 h/run, 3 folds ≈ 20 h. Inference measured at 13:39 for one 6,439-tile fold → ~45 min for three.

### Block 2 — paste after reading the smoke log. Arm A. **Expected ~26 h.**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_dsm_dtm_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_dsm_dtm_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_rgb_dsm_dtm_corrected_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold2.ini
```

Arm A reads three source rasters per tile against `rgb`'s one and `6ch`'s four, so ~50 min/epoch
→ ~8.3 h/run, 3 folds ≈ 25 h plus ~50 min inference.

**Both blocks stop at inference, deliberately.** Scoring is held until the author says so, because
Plan 3.1's amended gate is *declarations locked before the first arm score is read*. The scoring and
HF-backup commands for both cells are in `configs/matrix_configs/run_arms_G_A.cmd` and in
`MANIFEST_arms_G_A_2026_08_24.md`, ready to run at that point. Both cells are **descriptive, outside
every Holm family**; the Part B runner refuses to put them in one (§5).

---

## 4. Task 2 — the CPU backlog

### 4.1 lc100 per-class points (task 2.1) — done, and it changes Plan 3.1 §2

`extract_lc100_per_class.py` lifts the frozen `unet_resnet34_rgb` fold-0 diagnostic out of its
pooled JSON into the curve-point schema, after two guards: the extracted macro-IoU and accuracy must
reproduce the launch block's anchor (0.3292 / 0.8720 — they do, at 0.329187 / 0.871952), and
`evaluated_pixels` plus every class's `support_pixels` must match all three trained points exactly
(they do, at 3,084,826,556 px). See §6 for what the numbers mean.

### 4.2 E4, label quality (task 2.2) — run for the first time, and it lands a headline

Full write-up in `2026-08-24_E4_label_quality.md`. Three things the thesis did not have this morning:

1. **The 37 % ignore mass is annotation coverage, not annotator uncertainty.** Of everything that
   becomes ignore, **94.20 % is the nodata sentinel** and 5.80 % is explicit `unknown`, measured over
   all 799 parent rasters and 54.1 billion pre-tiling pixels. Cross-checked independently against the
   GeoPackage: 17,791 polygons, 56.43 km² annotated, 29.21 % of the tile footprint. The effective
   supervised dataset is ~12.2 of 19.3 billion pixels.
2. **The ignore mask is severely biased by geography: 11.80 % (82-11) to 89.33 % (83-34), spread 77.5
   points; per fold 52.09 % / 18.93 % / 39.55 %.** Routes therefore differ in *supervision* by a
   factor of 2,700 in scored pixels, on top of the class-diversity problem already recorded. This is
   new material for the statistical-limitations subsection and it sharpens D2.
3. **The boundary-error profile now exists as an artifact**, which is what the post-scriptum's SQ4
   limitation text has been citing without one. Error rate falls from **42.17 % on an annotation
   boundary to 2.42 % at ≥5 m**, a factor of 17; 8.9 % of all error sits within 20 cm of a boundary.
   But **31.23 % of all error sits more than 5 m from any boundary** — that share cannot be edge
   imprecision and is the F5b class-confusion reading. Both mechanisms are present and separable.

### 4.3 The 8 tiles with nDSM > 100 m (task 2.3) — all real structures

Full write-up in `2026-08-24_high_ndsm_tiles.md`. **All eight are genuine tall structures, and the
eight tiles hold only three distinct objects** (a ~298 m mast in 83-26 seen in two tiles of one
parent, a ~114 m mast seen in five tiles across the overlapping 82-20/82-21 pair, and a ~104 m object
in 84-40). The discriminator is the terrain model: DTM standard deviation is 0.027–0.088 m across a
5×5 m window under every peak while the DSM varies by 5–74 m, so the ground is flat and something
tall is standing on it. None is on the 671 misregistration list; all geotransforms agree.

Consequence: **the no-ceiling-clamp decision is vindicated for a better reason than the one recorded**
— a 40 m ceiling would have flattened three real masts. The tracker's guess that these were "almost
certainly a DSM spike or residual misregistration" is withdrawn. This is a landscape property, not a
pipeline defect, and should not be written as a data-quality finding.

### 4.4 HF sync (task 2.4) — 77 files pushed in 3 commits; **9 checkpoints left, needs a decision**

`hf_sync_batch.py` implements the list-then-batch fix: list what the repo holds, diff against what
should be there, upload only the gap, one commit per group. The token is read from file and never
printed. Dry run is the default.

Pushed: **10 findings notes, 45 per-run log files** (this covers the 21 missing CSVs and more), and
**25 analysis tables** — `cross_cell_summary.csv`, `route_cell_metrics.csv`, `route_cell_cms.npz`,
`route_cell_provenance.json`, the learning-curve tables, and tonight's new artifacts. Repo went from
230 to **310** files; post-push verification confirms 0 still missing.

**One thing the first version got wrong, and it generalises.** "Already in the repo" is not the same
as "current": `boundary_iou.csv` was pushed from the 120-tile pilot and would have stayed at that
version, because a missing-only diff never notices that a file *changed*. A `--refresh` flag now
re-uploads `results/tables/` and `results/findings/` unconditionally, and models, logs and split
files are left alone because they never change once written. **Use `--refresh` from now on** —
`python hf_sync_batch.py --refresh --push` — or the backup will silently drift from disk as the
analysis is regenerated.

**Stopped short, on purpose:** the diff also found **9 model checkpoints, 10.2 GB, that are not backed
up at all** —

```
  1455.1 MB  models/convnext_upernet_6ch_corrected_fold{0,1,2}.pth
  1455.0 MB  models/convnext_upernet_rgb_ndsm_fold{0,1,2}.pth
   494.9 MB  models/unet_resnet34_rgb_lc{25,50,75}_fold0.pth
```

The 2026-08-23 wrap-up records "new cells backed up to HF alongside the 72"; for the ConvNeXt arms
and all three learning-curve models that is **not the case**. Plan 3.1 §7's VM-loss mitigation reads
"HF holds the 72+3 models" — it holds 72 plus `unet_resnet34_6ch_corrected`. Task 2.4 asked for logs
and tables, not models, and a 10.2 GB upload is not something to start unasked. Run
`python hf_sync_batch.py --push --include-models` when you want it; everything else is already done
and the command is idempotent.

### 4.5 Metric breadth over all 27 cells (task 2.5) — done

`metric_breadth.py`. Per-class precision/recall/F1/IoU/support, macro precision and recall,
frequency-weighted IoU, overall accuracy, the completed worst-case metric, and the binary collapse.
Cross-check: macro-IoU, macro-F1, overall accuracy and evaluated pixels **reproduce the stored
`pooled_oof_metrics.py` values exactly on all 27 cells**.

**One trap worth recording, because it nearly cost a wrong table.** The first draft reimplemented the
scoring and dropped the ignore-index column as well as its row. The frozen convention keeps the
column: predicting `unknown` on a real-class pixel is a false negative for that class. Exactly one
cell in 27 does this — `unet_resnet34_10ch_unw`, on 14,671 pixels of 19.3 billion — so a private
reimplementation would have matched on 26 cells and been silently wrong on the 27th. The module now
delegates the per-class block to `per_category_metrics.metrics_from_confusion`, the same function
`pooled_oof_metrics.py` calls, and only adds aggregates on top. The self-test carries that case.

The headline exhibit, five metric families over the same cells (full table in
`metric_breadth_by_cell.csv`):

| cell | macro-IoU | FW-IoU | accuracy | macro-F1 | worst-class IoU | binary acc |
|---|---:|---:|---:|---:|---:|---:|
| `convnext_upernet_rgb_ndsm` | 0.3625 | 0.8914 | 0.9337 | 0.4639 | 0.0061 | 0.9626 |
| `convnext_upernet_rgb` | 0.3586 | 0.8906 | 0.9336 | 0.4593 | 0.0083 | 0.9622 |
| `convnext_upernet_10ch` | 0.3447 | **0.8979** | 0.9374 | 0.4348 | 0.0024 | **0.9698** |
| `convnext_upernet_6ch_corrected` | 0.3110 | 0.8756 | 0.9267 | 0.3975 | 0.0013 | 0.9522 |
| `unet_resnet34_rgb` | 0.2695 | 0.8545 | 0.8973 | 0.3454 | 0.0006 | 0.9363 |
| `segformer_b1_10ch_unw` | 0.2513 | 0.8836 | 0.9238 | 0.3229 | 0.0000 | 0.9639 |
| `unet_resnet34_6ch_corrected` | 0.2025 | 0.8378 | 0.8797 | 0.2594 | 0.0000 | 0.9302 |

**The metric choice reverses the ranking, which is the Sub-2/LO6 point made with the project's own
numbers.** `convnext_upernet_10ch` is third on macro-IoU but first on both frequency-weighted IoU and
binary accuracy. `segformer_b1_10ch_unw` is 23rd of 27 on macro-IoU yet beats `convnext_upernet_6ch_corrected`
on FW-IoU and binary accuracy. Macro-IoU spans 0.20–0.36 across the matrix while FW-IoU spans
0.84–0.90 and binary accuracy spans 0.93–0.97 — the last of which is the bridge to the service's ~95 %
headline and the 0.8747 trivial floor. `green_roof` is the worst class in 24 of 27 cells, `drivhus`
in the other 3.

**Two things flagged rather than decided:**

- **The binary mapping needs your confirmation.** Sealed = asfalt, fliser, grus, betonflade, brosten,
  solceller, drivhus, green_roof; unsealed = ubefestet; `unknown` and `unknown2` dropped. It puts
  solar panels, greenhouses and green roofs on the sealed side, which is defensible for an
  imperviousness proxy but is a judgement, not a measurement. It is recorded as such in the artifact.
- **The worst-case metric's citation.** Implemented as min single-class IoU plus the mean of the k
  worst classes, for k = 1, 2, 3 — the operational definition this project's own gap analysis records
  for G3.3. Wang Z. et al. 2023 is not on this machine, so the paper's own formulation must be checked
  on the reference-canon day (Plan 3.1 §5.5) before prose attributes the definition to it. The columns
  are correct as "min-class IoU" and "mean of the k worst" either way.

### 4.6 Boundary IoU (task 2.6)

Implemented to the Cheng et al. 2021 form: `|Gd ∩ Pd| / |Gd ∪ Pd|`, where `Md` is `M` minus its
erosion by a disc of radius `d`, with `d` = 2 % of the tile diagonal = **28 px (2.8 m at 0.1 m GSD)`.
Same citation caveat as the worst-case metric: the formula is implemented from the project's own gap
analysis (G3.2/L4), and the `d` convention must be checked against the PDF before prose attributes
specifics to the paper.

Self-test (six checks, all pass) includes the property the metric exists for:

```
  boundary region of a thick square is a ring (204 px) : OK
  a thin object is entirely its own boundary : OK
  identical -> 1.0, disjoint -> 0.0 : OK
  1 px shift: mask IoU 0.9048 vs boundary IoU 0.7143  (boundary is stricter) : OK
  same 1 px error, 8x larger object: mask IoU 0.9048 -> 0.9876 (dilutes),
                                     boundary IoU 0.7143 -> 0.7143 (holds) : OK
  tile border is not treated as an object boundary : OK
```

That fifth check initially asserted the wrong direction and failed — which is the useful outcome, and
worth keeping in the record. Boundary IoU is deliberately **size-insensitive**: under the same 1 px
boundary error, mask IoU improves as the object grows because the error dilutes against the interior,
while Boundary IoU does not move. That is precisely why it is the right instrument for the thin paved
classes, and it is now a demonstrated property rather than a cited one.

**Scope decision, stated because it deviates from "run as an overnight CPU batch".** Measured cost is
0.80 s per tile per cell, so the full pool would be 19,314 × 4 × 0.80 s ≈ **17 hours of 12-core CPU**,
competing directly with the arm-G training the author is about to launch. Boundary IoU is a ratio of
accumulated pixel counts, so a sample estimates the same quantity. The run is therefore **600 tiles
per fold, 1,800 per cell, fixed seed 20260824, stratified by fold** — the same sampling convention E4
uses — at ~1.6 h. `--tiles 0` runs the full pool if it is ever wanted. The sample size is recorded
beside every number in the artifact.

**Result.** 1,800 tiles per cell, 8.7 min per cell, 35 min total.

```
=== Boundary IoU vs mask IoU (same tiles, d = 28 px) ===
  class            rn34_rgb      cnx_rgb     cnx_ndsm  cnx_6chcorr
  asfalt        0.331/0.504  0.424/0.566  0.451/0.587  0.418/0.530
  fliser        0.157/0.160  0.284/0.270  0.309/0.301  0.271/0.258
  grus          0.129/0.135  0.190/0.216  0.172/0.196  0.159/0.189
  ubefestet     0.604/0.928  0.644/0.954  0.656/0.956  0.613/0.944
  green_roof    0.000/0.000  0.000/0.000  0.000/0.000  0.000/0.000
  drivhus       0.077/0.077  0.253/0.253  0.261/0.261  0.222/0.222
  betonflade    0.004/0.022  0.037/0.200  0.036/0.237  0.010/0.003
  brosten       0.008/0.007  0.102/0.104  0.134/0.149  0.029/0.029
  solceller     0.814/0.594  0.654/0.703  0.594/0.744  0.716/0.599
  (boundary IoU / mask IoU on the same tiles; boundary is stricter by construction)
```

**The metric splits the nine classes into four populations, and the split is the finding.**

1. **Bulk classes pay a large boundary penalty.** ubefestet scores 0.93–0.96 on mask IoU and
   0.60–0.66 on boundary IoU — it loses about a third of its quality at the edges. asfalt loses
   ~0.15 consistently. On these classes the mask metric is genuinely flattered by object interiors,
   which is the standard reading and the reason the metric was added.

2. **The thin classes are *entirely their own boundary*, so the two metrics coincide.** drivhus is
   0.077/0.077, 0.253/0.253, 0.261/0.261, 0.222/0.222 — **identical to three decimals in all four
   cells**. brosten (0.008/0.007, 0.102/0.104, 0.029/0.029) and fliser (0.157/0.160, 0.271/0.258)
   are within noise of identity. At d = 28 px an object thinner than 5.6 m has no interior left after
   erosion, so `Gd = G` and `Pd = P` and Boundary IoU degenerates to mask IoU.

   **This is a measurement, not a null result, and it is the one L4 predicted.** The thin paved
   surfaces really are thin: for brosten, fliser and drivhus there is no "right class, sloppy edge"
   category, because those objects are all edge. Mask IoU was never being flattered on them. It
   follows that their failure cannot be attributed to boundary imprecision and must be class
   confusion — which is exactly the F5b reading, arrived at independently of the JM separability
   matrix and of E4's finding that 31 % of all error sits more than 5 m from any boundary.

3. **betonflade is the exception, and it is a boundary failure.** mask 0.200 → boundary 0.037 on
   `convnext_upernet_rgb` and 0.237 → 0.036 on `rgb_ndsm`, a 5–7× collapse, the largest relative gap
   of any class. Concrete slabs are large enough to have interiors, and the models find the interiors
   while missing the edges. That is a different failure mode from brosten's and should be written as
   one.

4. **solceller inverts: boundary IoU EXCEEDS mask IoU** on `unet_resnet34_rgb` (0.814 vs 0.594) and
   on `6ch_corrected` (0.716 vs 0.599), as does fliser on both ConvNeXt RGB cells. Boundary IoU can
   exceed mask IoU when the errors sit in the *interior* of the ground-truth objects rather than at
   their edges — the model traces the panel outline but hollows it out. That is the salt-and-pepper
   and hollow-object signature, and it is a direct, quantitative hook for two of the pre-registered
   qualitative failure modes (D4: speckle, boundary bleeding).

**One caveat, and a cheap sensitivity if it matters.** d = 28 px is the paper's 2 % of the diagonal,
but at 0.1 m GSD that is 2.8 m, which is wide relative to these objects — it is what makes population
2 degenerate. `--d_px 5` (50 cm) would restore discrimination on the thin classes and is a ~35 minute
re-run. Worth doing only if the thesis wants to say something finer about brosten and fliser edges;
the degeneracy at 28 px is itself reportable and arguably the more interesting number.

Artifacts: `boundary_iou.csv` (36 rows, with the boundary and mask intersection/union counts so any
d-sensitivity is auditable) and `boundary_iou_provenance.json`.

---

## 5. Task 3 — the Part B statistics runner (built and self-tested; nothing run on real data)

`partb_statistics.py`. It consumes `route_cell_metrics.csv` (27 cells × 16 routes) and
`route_cell_cms.npz` (the 27×16×11×11 route confusion-matrix tensor), plus a declaration file, and
produces per declared pair: exact Wilcoxon signed-rank with the exact-tie count and effective N
beside every p-value; the exact two-sided sign test; Holm–Bonferroni within family; a Friedman
omnibus with mean ranks and Kendall's W per multi-pair family; median paired route difference with a
bootstrap CI plus rank-biserial; whole-route block-bootstrap CIs for pooled Macro-IoU; and McNemar as
context only.

**Three refusals are built in, and they are the point.**

1. No `--declaration` → it will not compute a test statistic at all.
2. `locked_utc: null` in the declaration → refuses, naming the reason. Descriptive route medians have
   already been seen (E6 landed 23/8), so the declaration must be dated and locked before any test
   statistic exists.
3. A cell listed in `descriptive_cells` appearing inside a Holm family → refuses. That keeps Plan 3.1
   D1's "the arms stay outside every family" enforceable rather than remembered.

Also: `--dry-run` parses and validates the declaration and prints the plan without computing
anything; `--go` is required for real numbers. No sklearn anywhere (its LDA and lbfgs solvers hard-
abort this environment).

**Get the machine-readable block with `python partb_statistics.py --emit-template`.** It is written
to drop straight into `2026-08-25_pre_declarations.md` and it is authoritative over the runner's
defaults, per the work order.

Self-test output, on synthetic data with known answers:

```
  [OK] Wilcoxon exact, all-positive N=5  --  p = 0.062500 = 2/32, rank-biserial 1.0
  [OK] forced all-tie routes dropped, effective N reported  --  7 pairs -> 2 ties -> effective N 5
  [OK] every pair tied -> undefined rather than a fabricated p
  [OK] effective-N power arithmetic reproduces the declaration
       --  min two-sided p: N=9 -> 0.0039 (clears 0.0050), N=8 -> 0.0078 (cannot)
  [OK] average ranks on tied |d|  --  W+ 4.5, W- 1.5
  [OK] exact sign test  --  5 positives p = 0.062500; 3/4 with 2 ties p = 0.6250
  [OK] Holm-Bonferroni step-down  --  adjusted [0.005, 0.032, 0.117, 0.117, 0.117]
  [OK] Friedman: perfect concordance -> Kendall's W = 1  --  chi2 16.000, df 2
  [OK] Friedman: pure noise -> not significant, W near 0  --  p 0.596, W 0.016
  [OK] chi-square upper tail matches table values to 1e-6
  [OK] bootstrap CI brackets the point estimate  --  [0.4698, 0.5494]
  [OK] whole-route block bootstrap recomputes the POOLED statistic
       --  macro-IoU 0.8877, CI [0.8858, 0.8897]
  [OK] identical routes -> zero-width CI (the bootstrap is resampling routes, not pixels)
  [OK] McNemar exact and chi-square  --  10/0 -> p 0.001953; 5.0M/4.9M -> p 1.16e-221
  [OK] unlocked declaration refused
  [OK] descriptive arm cell inside a Holm family refused
  [OK] independent cross-check vs scipy  --  wilcoxon exact p and friedman chi2/p agree
```

Two of these are worth pointing at. The **forced all-tie route** case the work order asked for is
covered twice: exact zeros are dropped and counted, and a fully tied comparison returns *undefined*
rather than a fabricated p. And the **effective-N arithmetic reproduces the declaration's own
reasoning independently** — at N = 9 the smallest attainable two-sided p is 0.0039, which clears
Holm's first threshold of 0.0050 with almost nothing to spare, and at N = 8 it is 0.0078 and cannot.
That is now a tested property of the runner, not a footnote.

The block bootstrap resamples **whole routes and recomputes the pooled statistic from the resampled
routes' confusion matrices** through the project's own scoring function — not an average of route
scores, which would be a different quantity. The self-test asserts it reproduces the direct pooled
value exactly, and that a set of identical routes yields a zero-width CI.

---

## 6. What the lc100 points do to Plan 3.1 §2 — please read this before writing §2

Per-class IoU on the byte-identical fold-0 held-out set, all four points:

| class | lc25 | lc50 | lc75 | **lc100** |
|---|---:|---:|---:|---:|
| ubefestet | 0.9295 | 0.9435 | 0.9460 | 0.9465 |
| asfalt | 0.5166 | 0.5354 | 0.5507 | 0.5568 |
| solceller | 0.7612 | 0.8603 | 0.8876 | 0.9010 |
| **grus** | 0.1615 | 0.1953 | 0.1950 | **0.2960** |
| fliser | 0.1284 | 0.1194 | 0.1553 | 0.1596 |
| **brosten** | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **green_roof** | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| **betonflade** | 0.0000 | 0.0019 | 0.0106 | **0.0354** |
| drivhus | 0.0284 | 0.0255 | 0.0542 | 0.0674 |
| Macro-IoU | 0.2806 | 0.2979 | 0.3110 | 0.3292 |
| accuracy | 0.8525 | 0.8657 | 0.8689 | 0.8720 |

**What survives, strengthened.** brosten and green_roof are **0.0000 at every point including 100 %**.
The production architecture never emits either class at any training volume, over a fourfold range
of data. Plan 3.1 §2's sharpest sentence holds exactly as written, and it now holds at the anchor too.

**What does not survive as written.** Plan 3.1 §2 says *"Decomposed per class, the entire rise is
refinement of classes that already work"*, and reads grus as "small step, then flat" and betonflade as
"effectively zero throughout". Both readings were taken from lc25–lc75 only:

- **grus rises 0.1950 → 0.2960 between 75 % and 100 %**, a 52 % relative jump after two flat
  intervals. grus is a weak class, and it is not refinement of something that already works.
- **betonflade rises 0.0000 → 0.0019 → 0.0106 → 0.0354.** Still very low, but monotone across all
  four points and 3.3× the lc75 value. "Effectively zero throughout" is no longer accurate at 100 %.

The SQ4 conclusion is unharmed and arguably better specified: **the classes data volume cannot touch
are brosten and green_roof**, and those are precisely the two the other evidence explains — brosten by
pixel-space inseparability (F10, JM asfalt–brosten 0.867) and green_roof by geographic support (2
routes, 90 tiles). grus and betonflade are a different case and do respond to volume. The honest
two-part statement becomes: the aggregate is still data-limited at 100 %, and per class the gains
concentrate on classes that already work **plus grus and, marginally, betonflade — while the two
classes the model never predicts at all stay at exactly zero across a fourfold range of data.**

That is a stronger claim than the one in the plan, because it survives the point that would have
been used to attack it.

---

## 7. Deviations from the work order, all of them

1. **Boundary IoU was sampled, not run over the full pool** (600 tiles/fold, seed 20260824). Reason
   and cost in §4.6; the full-pool run is one flag away. The work order permits stopping and
   reporting; this is less than that.
2. **The 9 missing model checkpoints were not uploaded** (§4.4). Task 2.4 scoped the sync to logs and
   tables; 10.2 GB is a decision, not a detail.
3. **The metric-breadth module imports `per_category_metrics` rather than recomputing**, after the
   first draft's ignore-column bug (§4.5). This is a deviation towards the work order's "no new
   metric code", not away from it.
4. **A `--dry-run` and a `--go` flag were added to the Part B runner** beyond what was specified, and
   the declaration guards refuse rather than warn. Stricter than asked.
5. **Two findings notes were written that the work order did not request** —
   `2026-08-24_E4_label_quality.md` and `2026-08-24_high_ndsm_tiles.md`. E4's own output is a console
   dump; under the one-source rule the write-up needs a persisted reading of it.
6. **No GPU job was launched.** The author did not write "you may launch" in this session.

Nothing was deleted, moved, renamed or truncated. `logs_and_models/` was read-only throughout
(the HF sync reads from it; the only writes this session were under `exploratory_data_analysis/`,
`configs/matrix_configs/` and the two source files named below). `_monitor.py` /
`_monitor_state.json` were not touched. The HF token was never printed.

---

## 8. File manifest

**Created — configs (`c:\thesis\ML_sdfi_fastai2\configs\matrix_configs\`)**

```
train/convnext_upernet_ortorgb_fold0.ini
train/convnext_upernet_ortorgb_fold1.ini
train/convnext_upernet_ortorgb_fold2.ini
train/convnext_upernet_rgb_dsm_dtm_corrected_fold0.ini
train/convnext_upernet_rgb_dsm_dtm_corrected_fold1.ini
train/convnext_upernet_rgb_dsm_dtm_corrected_fold2.ini
infer/infer_convnext_upernet_ortorgb_fold0.ini
infer/infer_convnext_upernet_ortorgb_fold1.ini
infer/infer_convnext_upernet_ortorgb_fold2.ini
infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold0.ini
infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold1.ini
infer/infer_convnext_upernet_rgb_dsm_dtm_corrected_fold2.ini
smoke_rgb_dsm_dtm_corrected_convnext.ini
MANIFEST_arms_G_A_2026_08_24.md
run_arms_G_A.cmd
```

**Created — scripts (`c:\thesis\exploratory_data_analysis\scripts\`)**

```
check_n_in_3_and_5.py          gate 2, forward pass at n_in 3 and 5
gate_arms_G_A.py               gates 3 and 4, split dry-run + arm-G dataloader
extract_lc100_per_class.py     task 2.1
inspect_high_ndsm_tiles.py     task 2.3
hf_sync_batch.py               task 2.4, list-then-batch
metric_breadth.py              task 2.5
boundary_iou.py                task 2.6
partb_statistics.py            task 3
```

**Created — findings (`c:\thesis\exploratory_data_analysis\results\findings\`)**

```
2026-08-24_work_order_arms_G_A.md     (the work order, saved verbatim, first action)
2026-08-24_E4_label_quality.md
2026-08-24_high_ndsm_tiles.md
2026-08-24_arms_G_A_build_record.md   (this file)
```

**Created — tables (`c:\thesis\exploratory_data_analysis\results\tables\`)**

```
gate_arms_G_A_ortorgb_batch_stats.json
learning_curve/lc100_per_class.json
learning_curve/lc100/single_fold_metrics.json
learning_curve/learning_curve_per_class.csv
label_quality.json                       (E4)
boundary_error_profile.csv               (E4)
high_ndsm_tiles.csv
high_ndsm_tiles.json
metric_breadth_by_cell.csv
metric_breadth_per_class.csv
metric_breadth_binary_collapse.csv
metric_breadth_provenance.json
boundary_iou.csv
boundary_iou_provenance.json
```

**Modified — two source files, both additively**

```
src/ML_sdfi_fastai2/analyse/generate_matrix_configs.py
    + emit_arms_G_A(), _load_G_A_constants(), _imagenet_from_frozen_rgb(), the
      --arms-G-A-2026-08-24 dispatch. main() and emit_arms() are byte-unchanged.
src/ML_sdfi_fastai2/analyse/validate_matrix_configs.py
    + the two new channel tags, EXPECTED_DATATYPES, the constants assertions, the
      widened JOB_RE. The frozen-72 assertions are unchanged and still fire.
```

**Pushed to the private HF backup:** 80 distinct files across 5 commits (10 findings, 45 logs, 25
tables; tables and findings re-pushed after the Boundary IoU batch replaced the pilot artifact).
Repo 230 → **310** files, 0 missing of everything checked **except the 9 model checkpoints of §4.4**,
which are the one open item from this session.

---

## 9. Where the four Plan 3.1 §5.4 metric families now stand

For the results chapter's "same cells, five metrics, different stories" exhibit, everything is now
on disk and regenerable by script:

| family | artifact | status |
|---|---|---|
| per-class precision / recall / F1 | `metric_breadth_per_class.csv` | done, 270 rows |
| macro-F1, frequency-weighted IoU, accuracy | `metric_breadth_by_cell.csv` | done, 27 cells |
| worst-case (min-class, k-worst mean) | `metric_breadth_by_cell.csv` | done; citation to verify |
| Boundary IoU | `boundary_iou.csv` | done, 4 declared cells, 1,800 tiles each |
| binary befæstet/ubefæstet | `metric_breadth_binary_collapse.csv` | done; mapping to confirm |

Plan 3.1 §4.6 lists Boundary IoU as first in the drop order. It did not need to be dropped, and it
turned out to carry more than breadth — see §4.6 population 2, which supplies an independent route to
the F5b conclusion.

---

# Addendum, 2026-08-25 — Option 2: full-roster completion

**Executes the 2026-08-25 addendum to the work order (Great Plan 3.1 §11.5). The author's rule:
every arm intervention is replicated across the entire declared four-model roster. Six new cells
built and gated. No GPU job launched; every gate ran on CPU with CUDA hidden, because the A100 is
occupied by the training queue.**

**Superseded, recorded here rather than by editing the source:** `2026-08-19_channel_axis_findings.md`
§5 argued that the Swin and SegFormer `6ch_corrected` arms need not run, on a mechanism-plus-two-
replications basis. That reasoning remains the true record of the 19/8–24/8 decision. Its operative
conclusion is superseded by the author's 25/8 full-roster decision; **Plan 3.1 §11.5 is the
correction of record and the findings file is not edited.**

## A1 — the six cells

| cell | n_in | datatypes | trainer | ImageNet source |
|---|---:|---|---|---|
| `unet_resnet34_ortorgb` | 3 | OrtoRGB | `segformer_train.py` | `unet_resnet34_rgb_fold0.ini` |
| `unet_resnet34_rgb_dsm_dtm_corrected` | 5 | rgb, DSM, DTM | `segformer_train.py` | same |
| `swin_upernet_ortorgb` | 3 | OrtoRGB | `train.py` | `swin_upernet_rgb_fold0.ini` |
| `swin_upernet_rgb_dsm_dtm_corrected` | 5 | rgb, DSM, DTM | `train.py` | same |
| `segformer_b1_ortorgb` | 3 | OrtoRGB | `segformer_train.py` | `segformer_b1_rgb_fold0.ini` |
| `segformer_b1_rgb_dsm_dtm_corrected` | 5 | rgb, DSM, DTM | `segformer_train.py` | same |

36 configs (18 train + 18 infer) plus 3 smoke configs, emitted by a new `emit_arms_roster()` path
behind `--arms-roster-2026-08-25`. Each model's ImageNet vectors are parsed from **its own** frozen
rgb config, so the identical-treatment rule is enforced per model rather than assumed to hold across
the roster.

**The one-variable property holds for all three models**, verified by diff — each `ortorgb` config
differs from its own frozen `rgb` config in exactly two lines, `datatypes` and `job_name`:

```
unet_resnet34 : datatypes = ["OrtoRGB"] vs ["rgb"];  job_name only
swin_upernet  : datatypes = ["OrtoRGB"] vs ["rgb"];  job_name only
segformer_b1  : datatypes = ["OrtoRGB"] vs ["rgb"];  job_name only
```

And every arm-A channel block is byte-identical to the ConvNeXt arm-A cell already in flight:

```
means      = [0.485, 0.456, 0.406, 0.08524121066459263, 0.07131383041867789]   MATCH
stds       = [0.229, 0.224, 0.225, 0.0929699110545978, 0.08554528605670214]    MATCH
datatypes  = ["rgb", "DSM", "DTM"]                                             MATCH
channels   = [[0, 1, 2], [0], [0]]                                             MATCH
```

## A2 — gates, verbatim

### Validator (frozen 72+72 still asserted unchanged)

The `ortorgb` constants assertion was tightened: it no longer compares against an ImageNet literal
but against **that model's own frozen rgb config**, resolved per model prefix.

```
OK: validated 123 training + 123 inference configs (246 total).
  - frozen 72-run matrix  : 72 train + 72 infer (unchanged)
  - additive 2026-08 arms : 48 train + 48 infer
  - learning curve (3.0 s4): 3 train + 3 infer (whole-route subsets, held-out fold unchanged)
  - n_in matches channel tag (rgb=3 / 6ch=6 / 10ch=10 / rgb_ndsm=4 / 6ch_corrected=6 / ortorgb=3 / rgb_dsm_dtm_corrected=5)
  - ortorgb carries ITS OWN model's frozen rgb constants (one variable: the source)
  - rgb_dsm_dtm_corrected carries measured corrected DSM/DTM + ImageNet RGB
  - each reads its fold_<f>_valid.txt; train keeps all.txt as the pool
  - training configs carry the locked 11-value weighted-CE vector + cross_entropy
  - pure bf16 (tf32 OFF); to_bf16/cudnn_benchmark/pin_memory true, to_fp16 false
  - inference model_to_load + output_folder match the cell's fold-model
```

### G-A forward gates — the known width-5 gap, closed

```
torch 2.6.0+cu124  cuda visible: False  (CPU-only gate)

=== G-A (Option 2 roster): forward at 1000x1000 -> expect 1x11x1000x1000 ===
  [PASS     ] resnet34+UNet          (segformer_train.py)  n_in=3  (ortorgb)
  [PASS     ] resnet34+UNet          (segformer_train.py)  n_in=5  (rgb_dsm_dtm_corrected)
  [PASS     ] Swin-base+UPerNet      (train.py)            n_in=3  (ortorgb)
  [PASS     ] Swin-base+UPerNet      (train.py)            n_in=5  (rgb_dsm_dtm_corrected)
  [PASS     ] SegFormer-b1           (segformer_train.py)  n_in=3  (ortorgb)
  [PASS     ] SegFormer-b1           (segformer_train.py)  n_in=5  (rgb_dsm_dtm_corrected)
ALL PASS
```

**GPU discipline, and a bug the gate caught in itself.** `check_n_in_roster.py` sets
`CUDA_VISIBLE_DEVICES` before importing torch and then **asserts** torch cannot see a device. The
first run set it to `""` and the assertion fired: on Windows an empty environment variable is
equivalent to an unset one, so the empty string left the GPU fully visible and the gate would have
contended with the training queue. Changed to `"-1"`. The assertion is the reason this is a footnote
and not an incident.

## A3 — Launch Block 4

Ready as `configs/matrix_configs/run_arms_roster.cmd`, and reproduced in §A3-block below. Order per
the addendum: three 5-channel smokes, then production model first, each cell training ×3 then
inference ×3. **Stops at inference** — scoring only after the declarations lock.

**Wall clock, from this project's own per-epoch logs (fold 0, 10 epochs per run):**

| | measured rgb | measured 6ch | `ortorgb` (3ch) | `rgb_dsm_dtm_corrected` (5ch) |
|---|---:|---:|---:|---:|
| resnet34 | 29:50/ep → 4.97 h | 49:40/ep → 8.28 h | **~5.0 h/run, ~15 h** | est ~7.5 h/run, **~22.5 h** |
| Swin | 44:50/ep → 7.47 h | 53:10/ep → 8.86 h | **~7.5 h/run, ~22.4 h** | est ~8.5 h/run, **~25.5 h** |
| SegFormer | 22:32/ep → 3.76 h | 49:25/ep → 8.24 h | **~3.8 h/run, ~11.3 h** | est ~7.5 h/run, **~22.5 h** |

`ortorgb` is a measurement, not an estimate: it is three uint8 bands from one source file, the same
I/O shape as that model's frozen `rgb` cell, so the rgb figure carries over directly.

`rgb_dsm_dtm_corrected` is **bracketed, not interpolated**. It reads three source rasters (rgb uint8
plus two float32 elevation bands) against `rgb`'s one and `6ch`'s four, so it must land between the
two measured columns, nearer the 6ch end because it carries both float32 rasters and drops only the
uint8 CIR read. The bracket is stated rather than a false-precision point estimate. **The ConvNeXt
arm-A cell in Block 2 lands before Block 4 starts and gives the first real 5-channel epoch time,
which calibrates all three estimates.**

Smokes ~45–51 min each (~2.2 h). Inference ~14 min per fold, measured, so ~4.2 h over 18 folds.

**Block 4 total ≈ 126 h ≈ 5.2 days.**

**Queue arithmetic, which is the thing that decides whether Option 2 fits.** Ahead of Block 4 sit
Block 1 (~21.5 h, running), Block 2 (~26 h), 6ch_corrected Swin (~26.6 h) and 6ch_corrected
SegFormer (~24.7 h) — about 99 h, so **Block 4 starts around 29/8 and finishes around 3–4/9**. That
matches Plan 3.1 §11.5's ~4–5/9 target and clears the **8/9 scoring cut-off by roughly four days**.
The margin is real but it is the only margin: a failed run costs a day, and two cost the slack.

## A4 — post-run handling (not yet due)

Nothing scored. When the author confirms the lock: pooled OOF scoring per cell → additive append to
`cross_cell_summary.csv` (27 → 37 rows at full completion) → E6 with element-by-element validation →
`hf_sync_batch.py --refresh --push`. For the two `6ch_corrected` cells the author launches from the
17/8 blocks, their own scoring lines run inside those blocks, so the job is to **verify** their rows
and E6 output landed rather than re-run them.

## A5 — Part B runner: it did need a code change

The addendum said the runner needs none. **It did, and the reason matters:** the author's actual
declaration block uses a materially different schema from the runner's template — `status: LOCKED`
rather than a `locked_utc` timestamp, `families` as a **mapping** of name to pairs rather than a list
of objects, `per_route_metric` as a mapping, and flat route-rule keys. The runner as built on 24/8
would have refused the real file on lock day. Now fixed, plus the declaration's newer requirements:

- **Sensitivity B**, the δ = 0.001 near-tie rule, on both the Wilcoxon and the sign test, with the
  near-tie count reported beside every p-value. The primary analysis is untouched by default, so δ
  stays a labelled sensitivity rather than a knob.
- **Paired pooled-difference bootstrap** for the 21 descriptive contrasts — one route resample
  applied to *both* cells per replicate, per declaration §4.
- **The fragility rule**: a pair is flagged when primary and either sensitivity disagree in
  significance, or when the sign test disagrees with the Wilcoxon.
- **Friedman fail-to-reject** sets `narrative_downgraded_to_descriptive`, per D1.
- **Skip-with-notice**, asymmetrically and deliberately: a declared **descriptive** cell that is not
  yet scored is skipped with a printed notice and recorded in the run provenance; a declared
  **family** cell that is missing is **fatal**, because dropping a pair silently changes the family
  size and therefore every Holm threshold in it. A partial family is not a valid run.
- **`--validate-schema`**, a pre-lock check that parses a DRAFT and reports what the runner would
  consume, computing nothing.

23 self-tests pass, including the independent scipy cross-check. Two additions worth naming: the
rank-biserial is asserted equal to the declaration's own `(W⁺−W⁻)/(n′(n′+1)/2)` form, and the
paired bootstrap is asserted to return an exactly zero-width interval at zero when both cells are
identical.

### Two findings from running `--validate-schema` against the author's block

**1. Sensitivity A's threshold does not do what the prose says.** `route_min_tiles_sensitivity: 100`
drops **only 83-34**, leaving n = 15. The declaration's §2 prose says it drops "83-34, 10 tiles;
85-48, 150 tiles), n = 14", and the YAML comment repeats it. But 85-48 has **150 tiles, which is not
under 100**. Measured thresholds:

```
  < 100  drops ['83-34']              -> n = 15
  < 150  drops ['83-34']              -> n = 15
  < 151  drops ['83-34', '85-48']     -> n = 14
  < 200  drops ['83-34', '85-48']     -> n = 14
  < 313  drops ['83-34', '85-48']     -> n = 14   (82-21 at 313 is the next route up)
```

The error is inherited, not introduced: tracker §4.2 and Plan 3.1 §5.1 D2 both state it the same
way, so three documents carry it. **It must be resolved before lock**, either by changing the
threshold to something in (150, 313] so n = 14 as intended, or by keeping 100 and correcting the
prose to "drops 83-34, n = 15". Intent points at the former, and 85-48 has an independent reason to
be droppable: the 2026-08-04 provenance report measured 100 % of its tiles as sharing ground with a
route in another fold.

**2. D3's [AUTHOR CONFIRM] resolves in favour of the sentence as written.** The declaration asks
whether `support_pixels > 0` matches `eda_route_cell_metrics.py`. It does. That module scores each
route through `metrics_from_confusion(..., ignore_index=0, report_only=(9,))`, and that function's
presence rule is `macro_present = [c for c in macro_classes if work[c, :].sum() > 0]` — support
greater than zero, with **no minimum-pixel threshold anywhere in the path**. The declaration's
`presence_rule: support_pixels_gt_0` is correct and the bracketed alternative does not apply.

Schema check output, for the record:

```
declaration    : 2026-08-25   status: DRAFT
per-route metric: macro_iou_present_classes   (column present in route_cell_metrics.csv: yes)
alpha 0.05   near-tie delta 0.001   route min tiles 100
routes primary 16, sensitivity A 15 (drops ['83-34'])
bootstrap: B=10000 seed=20260825 ci=0.95

families:
  F1_model_ranking_rgb_weighted            6 pairs   Holm sharpest threshold 0.00833
  F2_channel_within_convnext_weighted      3 pairs   Holm sharpest threshold 0.01667
  F3_weighting_within_convnext_rgb         1 pairs   Holm sharpest threshold 0.05000
  total 10 tests across 3 families

descriptive cells: 13 declared, 3 scored, 10 not yet scored (will be skipped with notice)
descriptive contrasts: 21 declared, 5 runnable now

  NOTE     status is 'DRAFT' -- the runner will refuse until it reads LOCKED
SCHEMA OK -- the runner can consume this block as written.
```

## A3-block — Launch Block 4, fully expanded

```bat
cd /d c:\thesis\ML_sdfi_fastai2
```

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_resnet.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_swin.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_segformer.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_ortorgb_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_ortorgb_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_ortorgb_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_ortorgb_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_ortorgb_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_ortorgb_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_dsm_dtm_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_dsm_dtm_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_dsm_dtm_corrected_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_dsm_dtm_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_dsm_dtm_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_dsm_dtm_corrected_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_ortorgb_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_ortorgb_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_ortorgb_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_ortorgb_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_ortorgb_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_ortorgb_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_dsm_dtm_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_dsm_dtm_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_rgb_dsm_dtm_corrected_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_dsm_dtm_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_dsm_dtm_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_rgb_dsm_dtm_corrected_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_ortorgb_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_ortorgb_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_ortorgb_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_ortorgb_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_ortorgb_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_ortorgb_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_dsm_dtm_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_dsm_dtm_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_rgb_dsm_dtm_corrected_fold2.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_dsm_dtm_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_dsm_dtm_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_rgb_dsm_dtm_corrected_fold2.ini
```

**Smoke pass criteria, one line:** all three finite `train_loss` around 0.15–0.25 and finite
`valid_loss` with no divergence, `valid_accuracy` roughly 0.78–0.87 at one epoch, and epoch times
near 45 min (resnet, SegFormer) and 51 min (Swin) — a time far above those means the two float32
elevation rasters cost more on that trainer path than the bracket assumes and the Block-4 estimate
needs revising before the queue commits five days to it.

## Deviations, 2026-08-25

1. **The Part B runner needed a code change**, contrary to the addendum's expectation (§A5). Fixing
   it was the only way the locked file would have run.
2. **Two problems found in the declaration draft and reported rather than silently worked around**
   (§A5): the Sensitivity A threshold arithmetic, and the D3 presence-rule confirmation.
3. **`--validate-schema` was added** beyond the specified scope, because the author locks today and a
   post-lock schema fix costs a dated amendment.
4. **The author's YAML block was copied to the session scratchpad** to run that check against the
   real text. It was not written into the project — the author owns and places the declaration file.
5. **No GPU job launched; no statistics computed on real data.** The declaration is still DRAFT.

## Files created, 2026-08-25

```
configs/matrix_configs/train/{unet_resnet34,swin_upernet,segformer_b1}_ortorgb_fold{0,1,2}.ini
configs/matrix_configs/train/{unet_resnet34,swin_upernet,segformer_b1}_rgb_dsm_dtm_corrected_fold{0,1,2}.ini
configs/matrix_configs/infer/infer_<each of the 18 above>.ini
configs/matrix_configs/smoke_rgb_dsm_dtm_corrected_{resnet,swin,segformer}.ini
configs/matrix_configs/MANIFEST_arms_roster_2026_08_25.md
configs/matrix_configs/run_arms_roster.cmd
exploratory_data_analysis/scripts/check_n_in_roster.py
```

**Modified, additively:** `generate_matrix_configs.py` (+`emit_arms_roster()`, `_roster_constants()`,
the `--arms-roster-2026-08-25` dispatch; `_imagenet_from_frozen_rgb()` gained an optional reference
path and is backward compatible), `validate_matrix_configs.py` (+`_frozen_rgb_constants()`,
per-model constant assertions), `partb_statistics.py` (declaration schema, near-tie sensitivity,
paired pooled-difference bootstrap, fragility rule, skip-with-notice, `--validate-schema`).
