# `6ch_corrected` — per-model launch blocks (2026-08-17)

**Four independent blocks. Nothing is chained, nothing has been launched.** Decide model by model.

Each block is self-contained: paste it into a **fresh Command Prompt** (cmd.exe, not PowerShell)
after `cd /d c:\thesis\ML_sdfi_fastai2`. Every line is fully expanded — no `set`, no `%PY%`.

## Pre-flight status (verified 2026-08-17)

- **12 training configs and 12 inference configs exist**, three folds for each of the four models.
  None missing, none invented.
- `validate_matrix_configs.py`: **PASS** — 96 training + 96 inference (192 total); the frozen
  72 + 72 asserted **unchanged**; `6ch_corrected` carries `n_in = 6`, the locked 11-value weighted-CE
  vector, `cross_entropy`, `to_bf16` true, **`tf32` false**.
- Trainer routing read from the generated launcher, not from memory, and matching handoff §2.6.

| model | model dir | trainer | comparison target (frozen weighted `6ch`) |
|---|---|---|---:|
| ConvNeXt+UPerNet | `convnext_upernet` | **`train.py`** | **0.3374** |
| Swin+UPerNet | `swin_upernet` | **`train.py`** | **0.2956** |
| SegFormer-B1 | `segformer_b1` | **`segformer_train.py`** | **0.2687** |
| resnet34+UNet | `unet_resnet34` | **`segformer_train.py`** | **0.2477** |

## What this arm changes, and what it therefore measures

Against the frozen `6ch` cell, two things differ — this is **not** a normalisation-only contrast:

1. **Measured constants** replace the placeholders for the three non-RGB bands. RGB stays at
   ImageNet by design, so the contrast is the auxiliary bands only.

   | band | frozen `6ch` | `6ch_corrected` | effective std before → after |
   |---|---|---|---|
   | cir NIR | mean 0.40779021, std 0.15176421 | mean 0.43791982, std 0.24592853 | 1.6205 → 1.0000 |
   | DSM | mean 0.5, std 1.0 | mean 0.08524121, std 0.09296991 | 0.0930 → 1.0000 |
   | DTM | mean 0.5, std 1.0 | mean 0.07131383, std 0.08554529 | 0.0855 → 1.0000 |

2. **The uint8 truncation of the elevation bands is gone** (`sdfi_transforms.py`, fixed 2026-08-15).
   The frozen 6ch runs trained on integer-metre elevation; these will not.

So a lift here is attributable to "the pipeline's channel handling was broken, and here is the
effect of fixing it" — **not** to normalisation alone. Report it that way.

**Still present in both arms:** the 671 tiles (3.47%) whose DSM/DTM are georeferenced to different
ground. That defect is unchanged between `6ch` and `6ch_corrected`, so it does not confound this
comparison, but it does bound what the corrected arm can recover.

**Sequencing note (2026-08-17).** The `rgb_ndsm` probe came back flat — 0.3625 against 0.3586, with
the elevation-signature classes flat or down. That is inside noise, so supplying object height
directly did not help. It raises the value of this arm: if `6ch_corrected` also comes back flat,
the honest reading is that the elevation *data*, not its encoding, is the limit.

---

## Block 1 — ConvNeXt+UPerNet · trainer `train.py` · target **0.3374**

**Training**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/convnext_upernet_6ch_corrected_fold2.ini
```

**Held-out inference**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_convnext_upernet_6ch_corrected_fold2.ini
```

**Scoring — one unbroken line**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_6ch_corrected
```

**Optional HF backup**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold0/models/convnext_upernet_6ch_corrected_fold0.pth --path_in_repo models/convnext_upernet_6ch_corrected_fold0.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold1/models/convnext_upernet_6ch_corrected_fold1.pth --path_in_repo models/convnext_upernet_6ch_corrected_fold1.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/convnext_upernet/convnext_upernet_6ch_corrected_fold2/models/convnext_upernet_6ch_corrected_fold2.pth --path_in_repo models/convnext_upernet_6ch_corrected_fold2.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/convnext_upernet/oof_convnext_upernet_6ch_corrected/pooled_oof_metrics.json --path_in_repo results/oof/oof_convnext_upernet_6ch_corrected.json
```

---

## Block 2 — Swin+UPerNet · trainer `train.py` · target **0.2956**

**Training**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/train.py --config configs/matrix_configs/train/swin_upernet_6ch_corrected_fold2.ini
```

**Held-out inference**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_swin_upernet_6ch_corrected_fold2.ini
```

**Scoring — one unbroken line**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_6ch_corrected
```

**Optional HF backup**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold0/models/swin_upernet_6ch_corrected_fold0.pth --path_in_repo models/swin_upernet_6ch_corrected_fold0.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold1/models/swin_upernet_6ch_corrected_fold1.pth --path_in_repo models/swin_upernet_6ch_corrected_fold1.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/swin_upernet/swin_upernet_6ch_corrected_fold2/models/swin_upernet_6ch_corrected_fold2.pth --path_in_repo models/swin_upernet_6ch_corrected_fold2.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/swin_upernet/oof_swin_upernet_6ch_corrected/pooled_oof_metrics.json --path_in_repo results/oof/oof_swin_upernet_6ch_corrected.json
```

---

## Block 3 — SegFormer-B1 · trainer `segformer_train.py` · target **0.2687**

**Training**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/segformer_b1_6ch_corrected_fold2.ini
```

**Held-out inference**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_segformer_b1_6ch_corrected_fold2.ini
```

**Scoring — one unbroken line**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_6ch_corrected
```

**Optional HF backup**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold0/models/segformer_b1_6ch_corrected_fold0.pth --path_in_repo models/segformer_b1_6ch_corrected_fold0.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold1/models/segformer_b1_6ch_corrected_fold1.pth --path_in_repo models/segformer_b1_6ch_corrected_fold1.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/segformer_b1/segformer_b1_6ch_corrected_fold2/models/segformer_b1_6ch_corrected_fold2.pth --path_in_repo models/segformer_b1_6ch_corrected_fold2.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/segformer_b1/oof_segformer_b1_6ch_corrected/pooled_oof_metrics.json --path_in_repo results/oof/oof_segformer_b1_6ch_corrected.json
```

---

## Block 4 — resnet34+UNet (production architecture) · trainer `segformer_train.py` · target **0.2477**

**Training**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/segformer_train.py --config configs/matrix_configs/train/unet_resnet34_6ch_corrected_fold2.ini
```

**Held-out inference**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_corrected_fold0.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_corrected_fold1.ini
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/infer.py --config configs/matrix_configs/infer/infer_unet_resnet34_6ch_corrected_fold2.ini
```

**Scoring — one unbroken line**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt ../multi_channel_dataset_creation/example_dataset/data/all.txt --label_folder ../multi_channel_dataset_creation/example_dataset/labels/splitted_labels --codes ../multi_channel_dataset_creation/example_dataset/labels/codes.txt --pred_fold0 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold0/models/example_dataset --pred_fold1 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold1/models/example_dataset --pred_fold2 ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold2/models/example_dataset --out ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_6ch_corrected
```

**Optional HF backup**

```bat
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold0/models/unet_resnet34_6ch_corrected_fold0.pth --path_in_repo models/unet_resnet34_6ch_corrected_fold0.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold1/models/unet_resnet34_6ch_corrected_fold1.pth --path_in_repo models/unet_resnet34_6ch_corrected_fold1.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/unet_resnet34/unet_resnet34_6ch_corrected_fold2/models/unet_resnet34_6ch_corrected_fold2.pth --path_in_repo models/unet_resnet34_6ch_corrected_fold2.pth
..\envs\ML_sdfi\python.exe src/ML_sdfi_fastai2/analyse/backup_to_hf.py --file ../logs_and_models/spatial_matrix/unet_resnet34/oof_unet_resnet34_6ch_corrected/pooled_oof_metrics.json --path_in_repo results/oof/oof_unet_resnet34_6ch_corrected.json
```

---

## Notes

- **`cd /d` matters.** Every path inside the configs is relative to `c:\thesis\ML_sdfi_fastai2`.
- Forward slashes are fine in cmd here — they are argv strings python resolves, and this is the
  exact form that ran the frozen matrix for four weeks.
- **Do not run `configs/matrix_configs/run_arms_2026_08.cmd`** — that file executes all three
  groups, 24 runs, chained.
- Roughly 9.3 h per training run at the observed matrix rate, so about 28 h per block plus
  inference and scoring.
- Nothing here writes to the frozen cells: every output path carries the `_6ch_corrected` suffix.
