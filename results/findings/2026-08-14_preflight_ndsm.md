# Pre-flight findings — nDSM and corrected-normalisation arms

**Answers §2 (P0.1, P0.2, P0.3) of the work order `2026-08-13_work_order_ndsm_and_corrected_normalisation.md`.**
Measurements taken 2026-08-14. Everything below is reproducible with
`exploratory_data_analysis/scripts/eda_preflight_ndsm.py`. Read-only against source data, configs,
checkpoints and logs; all outputs under `exploratory_data_analysis/`.

**Status: STAGE 0 COMPLETE. Nothing has been built. The nDSM folder has NOT been written.**

> **Headline.** P0.1 confirmed the ÷255 — and in doing so exposed **two defects nobody was looking
> for**. The elevation channels are cast to **uint8** before every augmentation, so the models were
> trained on elevation truncated to **integer metres**. And the horizontal-flip transform silently
> runs on the **validation** pass. Neither touches the inference path, so the 24 pooled cells remain
> geometrically sound — but **the nDSM arm cannot be built until the uint8 cast is fixed**, because
> that cast would destroy exactly the signal the arm exists to measure.

---

## P0.1 — the ÷255 is real, and measured

Config `convnext_upernet_6ch_fold0.ini`, real `DataLoader` via `sdfi_dataset.get_dataset()`, forced
to CPU, batch taken from `dls.valid`, tiles identified from `dls.valid.items` so the tensor can be
compared against the source rasters.

### What the model receives (batch of 4, 1000×1000)

| band | cfg mean | cfg std | min | max | mean | std |
|---|---:|---:|---:|---:|---:|---:|
| rgb_R | 0.48500 | 0.22900 | −2.1179 | 2.2489 | −0.2916 | 0.8807 |
| rgb_G | 0.45600 | 0.22400 | −2.0357 | 2.4286 | −0.0700 | 0.8116 |
| rgb_B | 0.40600 | 0.22500 | −1.8044 | 2.6400 | 0.0017 | 0.8367 |
| cir_b0_NIR | 0.40779 | 0.15176 | −2.6870 | 3.9022 | 0.6996 | 1.5363 |
| **DSM** | **0.50000** | **1.00000** | −0.5000 | **−0.3902** | −0.4552 | **0.0371** |
| **DTM** | **0.50000** | **1.00000** | −0.5000 | **−0.3902** | −0.4630 | **0.0393** |

The elevation channels arrive centred near **−0.46** with a standard deviation of **~0.04**, against
RGB's ~0 and ~0.85. SQ1 F8's conclusion stands and is now measured rather than inferred.

### The decisive test

Reconstructing the tensor from the source rasters under two hypotheses:

| band | max&nbsp;\|x−H255\| | max&nbsp;\|x−H1\| | implied divisor | verdict |
|---|---:|---:|---:|---|
| rgb_R | **0.000e+00** | 1.109e+03 | 255.0000 | /255 applied |
| rgb_G | **0.000e+00** | 1.134e+03 | 255.0000 | /255 applied |
| rgb_B | **0.000e+00** | 1.129e+03 | 255.0000 | /255 applied |
| cir_b0_NIR | **0.000e+00** | 1.674e+03 | 255.0000 | /255 applied |
| DSM | 3.922e-03 | 2.820e+01 | 269.69 | /255 applied |
| DTM | 3.922e-03 | 2.820e+01 | 310.00 | /255 applied |

Positive control passes: rgb is uint8 0–255 and lands at mean −0.29…0.00, std 0.81…0.88.

**`eda_common.INT_TO_FLOAT_DIV = 255.0` is CONFIRMED by measurement.** SQ1 F8's hedge ("the
conclusion does not depend on resolving where fastai's `IntToFloatTensor` acts") can be retired.

### The residual that was not supposed to be there

The four uint8 bands reconstruct **exactly** (0.000e+00). The two float32 elevation bands do not —
they carry a residual of **3.922e-03, which is exactly 1/255**, i.e. exactly **1.0 metre**.

Chasing it: `uint8(raw)` matches the delivered tensor to **1.0e-6**, while `raw`, `floor(raw)` and
`round(raw)` all miss by ~1.0.

For one real tile:

| | distinct DSM values | distinct DTM values |
|---|---:|---:|
| source raster / `load_all_datasources_for_image` | **48,698** | **48,031** |
| tensor delivered to the model | **18** | **3** |

---

## P0.1b — two defects, and which paths they reach

### Defect A — elevation is cast to uint8

Every albumentations wrapper in `sdfi_transforms.py` does, before augmenting:

```python
img = np.array(img, dtype=np.uint8).astype(np.uint8).copy()    # :194, :214, :226, :240, :259, :277, ...
```

Lossless for rgb/cir, which are uint8 already. Destructive for DSM/DTM, which hold **float32 metres
at 0.1 m GSD** — they are truncated to integer metres. Negative values (present: 128 tiles have a
negative mean nDSM) would additionally wrap under an unsigned cast.

### Defect B — the horizontal flip runs on validation

`sdfi_transforms.py:273`:

```python
class SegmentationAlbumentationsHorizontalFlip(ItemTransform):
    def __init__(self,split_idx): self.aug = albumentations.HorizontalFlip(p=0.05)
```

It never calls `ItemTransform.__init__`, so `split_idx` is discarded and defaults to `None`, meaning
**both splits**. Every sibling class forwards it correctly — `VerticalFlip` (:254),
`RandomRotate90` (:264), `GaussNoise` (:283). This is a one-line omission in one class.

### Measured consequence, 200 tiles down each path

| path | active `after_item` transforms | flipped | DSM integer-quantised |
|---|---|---:|---:|
| **train** | AddMaskCodes, Transpose, ShiftScaleRotate, GaussNoise, **HorizontalFlip**, VerticalFlip, ToTensor | 14/200 | **200/200** |
| **valid** | AddMaskCodes, **HorizontalFlip**, ToTensor | **11/200** | **200/200** |
| **inference** | AddMaskCodes, ToTensor | **0/200** | 3/200 |

Flipping on *train* is intended augmentation. Flipping on *valid* is Defect B. The 3/200 on
inference are degenerate tiles whose DSM is genuinely integer-valued (flat or nodata-dominated), not
a systematic cast.

### What this does and does not invalidate

- **The 463,536 predictions and the 24 pooled cells are geometrically sound.** `test_dl` carries none
  of the item transforms, so inference never flipped a tile and used **full float32 elevation**.
- **The 6ch and 10ch models were trained on integer-metre elevation and then served float32.** That
  is a train/serve skew on precisely the channels under investigation.
- **`valid_loss` / `valid_accuracy` in the training logs were computed with ~5% of tiles randomly
  mirrored.** Minor, but it is noise in the curves analysed in P0.3, and it should be stated.

**This strengthens handoff finding F4 rather than merely explaining it.** There are now two
independent mechanisms crippling the auxiliary elevation channels — placeholder normalisation
constants *and* uint8 truncation. "The auxiliary channels do not earn their keep" is not a supportable
reading of the matrix.

---

## P0.2 — DSM/DTM redundancy, exactly, over the full pool

`tile_elevation_stats.csv` carries per-tile `dsm_std`, `dtm_std` and `ndsm_std`, so the pixel-level
correlation is recoverable in closed form over all 19,314 tiles. **No sampling.**

### Validity gate

| channel | total sd | between-tile sd | within-tile sd | SQ1 F8 sd | match |
|---|---:|---:|---:|---:|---|
| DSM | 23.707 | 23.295 | 4.403 | 23.707 | ✔ |
| DTM | 21.814 | 21.773 | 1.334 | 21.814 | ✔ |
| nDSM | 8.977 | 7.950 | 4.169 | 8.977 | ✔ |

All three SQ1 F8 standard deviations reproduced exactly. 128 tiles have a negative mean nDSM, so the
CSV's nDSM column is **unclamped** and the identity `Cov = (Var(DSM)+Var(DTM)−Var(nDSM))/2` is valid.

### Three quantities, kept apart

| quantity | r | r² |
|---|---:|---:|
| **pixel-level Pearson r — closed form, full pool** | **0.9256** | **0.8567** |
| pixel-level r — back-solved from SQ1 F8's three sds | 0.9255 | 0.8566 |
| pixel-level r — direct, 190 tiles × 16 routes, 190,000,000 pixel pairs | 0.9147 | 0.8366 |
| tile-mean r — correlation of the 19,314 per-tile means | 0.9400 | 0.8836 |

The three pixel-level estimates agree; the raster sample differs by 0.011, as expected from a 1%
sample.

### What the "~97%" actually was

**It was never a computed correlation.** It was a loose verbal restatement of SQ1 F8's **between-tile
variance shares** — DSM 96.6%, DTM 99.6% — which describe how much of each channel is terrain
elevation rather than within-tile structure. That is a statement about one channel at a time, not
about how much the two channels share with each other. The tile-mean correlation (0.940) is not its
source either.

**The number for the thesis is r = 0.926, r² = 0.857.**

---

## P0.3 — epoch budget

24 weighted jobs (4 models × 3 folds × {rgb, 6ch}). Each epoch is logged twice in `<job>.csv`, once
with an `lr_0` column and with a second header re-emitted mid-file, which is why a 10-epoch run has
21 lines; rows are deduplicated on `epoch`.

- **Epoch count: exactly 10 in all 24 jobs. UNIFORM.** The new arms must use 10.
- **Final learning rate 2.00e-08 in all 24** — `fit_one_cycle` fully annealed, so no run was truncated.
- **Still improving on `valid_loss`: 1 / 24** (`unet_resnet34_6ch_fold0`).
- **Still improving on `valid_accuracy`: 2 / 24.**
- **The final epoch is the best `valid_loss` in only 1 / 24.**

**Falsification test #4 from handoff §6.1 is answered: NO.** The RGB ≥ 10ch ≥ 6ch ordering is not an
epoch-budget artefact. If anything the runs trained too long — `train_loss` falls ~7× while
`valid_loss` rises.

### P0.3b — is `<job>.pth` the final epoch, or selected on the held-out fold?

This matters because each fold's "validation" split **is** the held-out fold pooled into the OOF
matrix.

Wiring: `SaveModelCallback(every_epoch=True, monitor='valid_loss')` writes `<job>_<n>.pth` each epoch
and performs **no best-model restore**; `learn.save(job_name)` then writes `<job>.pth`
(`train.py:665`, `segformer_train.py:375`). Timestamps agree — epoch 9 at 11:00:07, final at
11:00:08.

Verified on the **weights** (byte comparison is unreliable across the two save paths), across all
three architecture families and both channel configs:

| job | best valid_loss epoch | `<job>.pth` matches | verdict |
|---|---:|---|---|
| convnext_upernet_rgb_fold0 | 0 | epoch **9** | clean |
| convnext_upernet_6ch_fold0 | 0 | epoch **9** | clean |
| segformer_b1_rgb_fold0 | 4 | epoch **9** | clean |
| segformer_b1_6ch_fold0 | 1 | epoch **9** | clean |
| swin_upernet_rgb_fold0 | 2 | epoch **9** | clean |
| swin_upernet_6ch_fold0 | 1 | epoch **9** | clean |
| unet_resnet34_rgb_fold0 | 6 | epoch **9** | clean |
| unet_resnet34_6ch_fold0 | 9 | epoch **9** | clean |

**`<job>.pth` is the FINAL epoch in every job checked. Model selection never saw the held-out fold.
The 24 pooled cells are NOT contaminated.** The overfitting is a limitation paragraph, not a validity
problem. No per-epoch checkpoint has been deleted or moved.

### P0.3c — are the two Swin weighting inversions an overfitting artefact?

Divergence = mean over the cell's 3 folds of `final_valid_loss − best_valid_loss`.

| rank | cell | divergence | valid−train gap |
|---:|---|---:|---:|
| 1 | convnext_upernet_6ch | 0.4462 | 0.8357 |
| **2** | **swin_upernet_6ch** ← inversion | **0.3354** | 0.7472 |
| 3 | convnext_upernet_rgb | 0.3099 | 0.6876 |
| **4** | **swin_upernet_rgb** ← inversion | **0.2933** | 0.6961 |
| … | … | … | … |
| 24 | unet_resnet34_10ch_unw | 0.0482 | 0.4013 |

Paired against their own unweighted twins:

| pair | weighted divergence | unweighted divergence | |
|---|---:|---:|---|
| swin_upernet_rgb | 0.2933 | 0.2113 | weighted diverges **more** |
| swin_upernet_6ch | 0.3354 | 0.1558 | weighted diverges **more** |

**Both inversion cells rank in the worst 4 of 24 on divergence, and in both pairs the weighted arm
diverges more than its unweighted twin.** This is consistent with F3's two exceptions being an
**overfitting artefact** rather than a real loss-by-architecture interaction. It is suggestive, not
proof — a seed replication would settle it. F3 should be written up with this caveat attached rather
than presenting Swin as a genuine exception.

---

## Consequences for Stage 1

1. **BLOCKER — the uint8 cast must be fixed before the nDSM arm is worth running.** Per SQ1 F8 the
   class-mean nDSM values are betonflade 0.19 m, solceller 0.26 m, brosten 0.78 m, asfalt 2.48,
   ubefestet 3.13, drivhus 4.11, green_roof 5.28. Truncating to integer metres collapses the first
   three to **0** and leaves the rest with three or four distinct levels. The arm would measure
   almost nothing. The same applies to `6ch_corrected`: correcting the constants while leaving the
   cast in place fixes one of the two defects and would understate the effect.
2. **The fix is narrow.** Preserve float32 through the augmentation wrappers, or exempt the float
   bands from the uint8 round-trip. It touches `sdfi_transforms.py` only.
3. **Defect B (validation flipping) should be fixed in the same change** — one missing
   `ItemTransform.__init__` call at `sdfi_transforms.py:273`.
4. **Comparability must be stated.** Fixing the cast means the new arms differ from the frozen
   6ch/10ch cells in **two** ways, normalisation and quantisation. For `6ch_corrected` that is the
   intended contrast ("the pipeline's channel handling was broken, here is the effect of fixing it"),
   but it cannot be reported as a normalisation-only result. `rgb_ndsm` remains a clean isolation of
   object height because it contains no CIR.
5. Epoch budget for both new arms: **10 epochs**, matching the cells they are compared against.

## Reproduction

```
cd c:\thesis\exploratory_data_analysis\scripts
..\..\envs\ML_sdfi\python.exe eda_preflight_ndsm.py --selftest
..\..\envs\ML_sdfi\python.exe eda_preflight_ndsm.py --p01 --p01b --p02 --p03
```

Artifacts: `results/tables/preflight_batch_channel_stats.csv`,
`preflight_pipeline_paths.json`, `preflight_dsm_dtm_correlation.json`,
`preflight_epoch_budget.csv`.
