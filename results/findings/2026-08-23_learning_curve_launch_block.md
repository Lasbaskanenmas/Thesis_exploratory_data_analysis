# Learning curve (Plan 3.0 §4, foundational analysis 4) — launch block

**Prepared and gated. Nothing launched.** resnet34+UNet, weighted CE, frozen spatial protocol.

## Channel config: §4 does not specify one

Plan 3.0 §4 says *"the production model (resnet34+UNet)"* and names **no channel config**. Stating
that plainly rather than inventing a citation.

**`rgb` is the right choice** for two reasons: it is the only resnet cell untouched by all three
elevation defects (placeholder constants, uint8 truncation, and the 671 misregistered DSM/DTM tiles),
so the curve measures data volume rather than channel damage; and it gives a **frozen** 100 % anchor.
That also matches the correction in the request.

## Held-out fold: 0 — measured, not assumed

| candidate | training pool | largest route | held-out classes | `unet_resnet34_rgb` fold score |
|---|---|---|---|---:|
| **fold 0** | **13 routes, 12,875 tiles** | **83-25 = 28.2 %** | **9 / 9** | 0.3292 |
| fold 1 | 11 routes, 12,877 | 84-40 = 40.9 % | 8 / 9 | 0.2045 |
| fold 2 | 8 routes, 12,876 | 84-40 = 40.9 % | 9 / 9 | 0.2701 |

Folds 1 and 2 are both dominated by route 84-40 at 40.9 % of their training pool, which makes a 25 %
point **unreachable** by whole routes. Fold 0 is the only one that supports the ladder, and its
held-out set carries all nine classes so per-class curves are measurable.

**Trade-off, stated honestly:** fold 0 scores 0.3292 against the pooled 0.2695, so the curve sits
above the headline level. That is acceptable — a learning curve is read for *shape*, and every point
here is scored on a byte-identical held-out set.

## The subsets

Held-out fold 0 = routes **83-31, 84-40, 85-48** (6,439 tiles), **identical at every point**.

| point | routes | tiles | % of pool | classes | route list |
|---|---:|---:|---:|---:|---|
| **lc25** | 6 | 3,222 | **25.0 %** | 9/9 | 82-16, 82-19, 82-21, 82-24, 83-34, 85-45 |
| **lc50** | 10 | 6,444 | **50.1 %** | 9/9 | + 82-13, 82-20, 83-26, 84-41 |
| **lc75** | 12 | 9,245 | **71.8 %** | 9/9 | + 82-11, 82-22 |
| **lc100** | 13 | 12,875 | 100 % | 9/9 | + 83-25 — **frozen cell, no run needed** |

**Nesting verified programmatically: lc25 ⊂ lc50 ⊂ lc75 ⊂ lc100.** No held-out route appears in any
subset.

**Why lc75 is 71.8 % and not 75 %.** After lc75 the only route left is 83-25 at 3,630 tiles — 28.2 %
of the pool — so the next whole-route step jumps straight to 100 %. Whole-route subsampling cannot
land on 75 % here; 71.8 % is the closest attainable point.

**Why the subsets are not simply the biggest routes.** Size-ordered prefixes leave `solceller` absent
from training until 84 % of the pool, which would make its curve measure class *presence* rather than
data volume — the opposite of what §4 asks. Subsets are therefore chosen by exhaustive search over
the 2¹³ subsets, ranked by class coverage first and closeness to the target fraction second, subject
to nesting. All four points carry **9/9 classes**.

## Mechanism and gates

Each point gets its own pool list — subset training tiles **+** the untouched fold-0 held-out tiles —
under `configs/matrix_configs/learning_curve/`. The trainer computes `train = path_to_all_txt −
path_to_valid_txt`, and `path_to_valid_txt` is the frozen `fold_0_valid.txt`, unchanged.

- `validate_matrix_configs.py`: **PASS** — 99 + 99 configs; frozen 72 + 72 asserted **unchanged**;
  arms 24 + 24; learning curve 3 + 3, each asserted to read a `learning_curve/` pool matching its own
  tag and fold.
- `score_single_fold.py --validate`: **PASS** — reproduces the frozen `unet_resnet34_rgb` fold-0
  diagnostic to **0.00e+00** on macro-IoU, macro-F1, accuracy and all nine per-class IoUs. The curve
  points are therefore directly comparable to the anchor.
- **Anchor (lc100), no compute:** Macro-IoU **0.3292**, accuracy **0.8720**, 9/9 classes.

## Launch block — cmd.exe, fully expanded

Fresh Command Prompt, then:

```bat
cd /d c:\thesis\ML_sdfi_fastai2
```

**Training** — resnet34 routes through `segformer_train.py` (handoff §2.6). Sequential, ~9 h each:

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_lc25_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_lc50_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_rgb_lc75_fold0.ini
```

**Held-out inference** — the protocol scores held-out predictions, so this is required:

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_lc25_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_lc50_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_rgb_lc75_fold0.ini
```

**Scoring** — single fold, one line each:

```bat
..\envs\ML_sdfi\python.exe ..\exploratory_data_analysis\scripts\score_single_fold.py --fold 0 --pred_folder ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_lc25_fold0/models/example_dataset --out ../exploratory_data_analysis/results/tables/learning_curve/lc25
..\envs\ML_sdfi\python.exe ..\exploratory_data_analysis\scripts\score_single_fold.py --fold 0 --pred_folder ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_lc50_fold0/models/example_dataset --out ../exploratory_data_analysis/results/tables/learning_curve/lc50
..\envs\ML_sdfi\python.exe ..\exploratory_data_analysis\scripts\score_single_fold.py --fold 0 --pred_folder ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_lc75_fold0/models/example_dataset --out ../exploratory_data_analysis/results/tables/learning_curve/lc75
```

**Optional HF backup:**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_lc25_fold0/models/unet_resnet34_rgb_lc25_fold0.pth --path_in_repo models/unet_resnet34_rgb_lc25_fold0.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_lc50_fold0/models/unet_resnet34_rgb_lc50_fold0.pth --path_in_repo models/unet_resnet34_rgb_lc50_fold0.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_rgb_lc75_fold0/models/unet_resnet34_rgb_lc75_fold0.pth --path_in_repo models/unet_resnet34_rgb_lc75_fold0.pth
```

Three training runs, about 28 h total. The 100 % point needs no run.
