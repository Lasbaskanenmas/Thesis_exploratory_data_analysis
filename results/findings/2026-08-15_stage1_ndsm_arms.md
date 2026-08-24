# Stage 1 — nDSM and corrected-normalisation arms: build record

**Executes Stage 1 of the 2026-08-13 work order under the 2026-08-15 operating rules.**
Additive only; `logs_and_models/` was read-only throughout. Nothing was deleted, moved, renamed or
truncated. **No GPU job was launched.** Pre-flight measurements are in
`2026-08-14_preflight_ndsm.md`; this document records what was built on top of them.

---

## 1. The two transform defects are fixed and re-verified

Pre-fix source preserved two ways: commit **`1a2640809620b50ea307c1676866c927b2927d3f`** on
`thesis/spatial-matrix-2026` (pushed to github.com/Lasbaskanenmas/Thesis_ML_sdfi_fastai2), and a
git-independent copy in `exploratory_data_analysis/provenance/`
(sha256 `1DD27519…B523F9A9`, mtime **2026-05-27**, a month before the matrix ran, so those bytes are
exactly what trained the 72 models).

**Defect A — uint8 cast on the elevation bands.** Fixed by routing float-safe augmentations through
float32 and giving the float-*unsafe* ones a band split. The split was necessary, not cosmetic:
measured on albumentations 1.3.0, `GaussNoise`, `RandomBrightness` and `ColorJitter` assume float
images live in `[0,1]` and would have clipped elevation metres to 1.0 — worse than the original bug.
They now run on the uint8-representable bands only, chosen data-driven per band rather than from a
hard-coded list, and elevation passes through untouched. The geometric ops (`Transpose`, both flips,
`RandomRotate90`, `ShiftScaleRotate`, the crops, `LongestMaxSize`) preserve float32 exactly and take
the array as-is.

**Defect B — `SegmentationAlbumentationsHorizontalFlip` dropped its `split_idx`.** One missing
`ItemTransform.__init__` call, added.

**Verified by the same measurement that found the defects:**

| check | before fix | after fix |
|---|---|---|
| DSM/DTM `max\|x − H255\|` | 3.922e-03 (= exactly 1/255) | **0.000e+00** |
| implied divisor, DSM / DTM | 269.69 / 310.00 | **255.0000 / 255.0000** |
| "reconstruction exact" | False | **True** |
| valid-set `after_item` | AddMaskCodes, **HorizontalFlip**, ToTensor | AddMaskCodes, ToTensor |
| valid tiles flipped | 11 / 200 | **0 / 200** |
| train tiles quantised **by the pipeline** | 200 / 200 | **0 / 200** |

The residual 14/200 "quantised" tiles on the train path were checked individually: every one has a
source DSM with `std = 0.00000`, i.e. an entirely constant raster. Zero tiles are quantised by the
pipeline.

## 2. A third defect, found while building nDSM: channel misregistration

`load_all_datasources_for_image` stacks channel folders **by filename** and never compares
georeferencing. Audit over all 19,314 tiles and all six folders (`channel_georeferencing_audit.csv`):

| channel | misaligned vs the `rgb` tile |
|---|---:|
| rgb, cir | 0 |
| OrtoRGB / OrtoCIR | 1 / 4 |
| **DSM** | **100** (0.52 %) |
| **DTM** | **580** (3.00 %) |
| **DSM vs DTM (the pair nDSM subtracts)** | **671** (3.47 %) |

669 of the 671 are **≥ 100 m apart**, and a tile is 100 m across, so there is no overlap at all; at
least one has an outright corrupt origin. Labels follow the `rgb` grid (663/671), and of the
mismatched pairs DSM matches `rgb` in 571 cases, DTM in 91, neither in 9 — the 100 DSM misalignments
are exactly those 91 + 9, so **all 18,643 computed nDSM tiles are correctly aligned to `rgb` and
hence to the labels**. Concentrated by route: 82-20 alone accounts for 333.

**Consequence for the frozen matrix:** every 6ch and 10ch run fed the network an elevation channel
from the wrong place on those ~3.5 % of tiles. RGB is untouched.

## 3. The 10ch − 6ch contrast survives all three defects

All three defects apply **identically** to 6ch and 10ch: the same placeholder constants, the same
uint8 cast, the same misregistered tiles. The bands that distinguish the two configs are `OrtoRGB`
and `OrtoCIR`, which are uint8 imagery — untouched by the cast, and essentially perfectly aligned
(1 and 4 misaligned tiles). **So `10ch − 6ch` still isolates the oblique source with the defects
held constant, and remains usable**, pending the route-level Wilcoxon. F1 and F2 are likewise
unaffected. What is not supportable is any claim of the form "the auxiliary channels do not earn
their keep".

## 4. The nDSM channel

`nDSM = max(DSM − DTM, 0)`, floor clamp only, clamped **before** any statistic was measured.

- **18,643 tiles computed**, **671 zero-filled** (the misregistered pairs — no co-located terrain
  model exists, listed in `ndsm_unresolved_tiles.csv`), **19,314 total**.
- Zero-filling rather than dropping keeps the evaluation pool identical to the 24 frozen cells, so
  the arm's pooled out-of-fold matrix stays directly comparable, and route composition is unchanged.
  Those tiles carry no height signal — a documented 3.47 % limitation, and strictly better than the
  wrong-place elevation the frozen runs were given for the same tiles.
- **2.74 GB** on disk, against a 40 GB budget. Zero stray `.tmp` files (atomic rename; a partial
  write can never appear at a final path). The generator is idempotent — existing tiles are skipped.

**All §3 assertions pass**, over the full 19,314: file count and names exact, 0 missing / 0 extra /
0 stray; every tile 1000×1000 float32 EPSG:25832 with a geotransform equal to its **`rgb`** tile
(changed from "its DSM source" — the audit showed DSM is itself misplaced on 100 tiles, so it is not
a sound reference); global minimum **0.000000**; median of per-tile medians **0.000000 m**; and
5/5 spot-checks against a manual numpy difference exact to **0.000e+00**.

### Clamped full-pool statistics

| | |
|---|---:|
| pooled mean | 2.569503 m |
| pooled sd | 5.676798 m |
| between-tile / within-tile sd | 3.9316 / 4.0949 |
| **between share** | **48.0 %** |
| global min / max | 0.0000 / 298.3268 m |

Clamping improves the informative within-tile share from SQ1 F8's unclamped 78.4 % between-share to
**48.0 %**, i.e. more than half of nDSM's variance is now structure inside the tile, against DSM's
3.4 %.

**One deviation from the work order to record.** §3 justified omitting a ceiling clamp partly as
"standardisation handles a **7σ** tail". The measured tail is **52σ** (298.33 m against mean 2.57,
sd 5.68), because clamping shrinks the pooled sd. The *extent* premise still holds and the decision
stands: median per-tile max is 11.71 m, p99 is 40.86 m, and only **8 tiles (0.04 %)** exceed 100 m.
No ceiling clamp was applied. Separately, 1,852 tiles have max nDSM = 0, of which only the 671 are
zero-filled — the other 1,181 are genuinely flat ground.

## 5. Measured normalisation constants

Derived from `channel_per_class_stats.csv` by pooling classes with the law of total variance; the
class pixel counts sum to exactly 19,314,000,000, so this is the whole pool. Constants live in
post-÷255 space, which P0.1 established by measurement.

| band | matrix constant | corrected | effective std before → after |
|---|---|---|---|
| cir NIR | mean 0.40779021, std 0.15176421 | mean 0.43791982, std 0.24592853 | 1.6205 → **1.0000** |
| DSM | mean 0.5, std 1.0 | mean 0.08524121, std 0.09296991 | 0.0930 → **1.0000** |
| DTM | mean 0.5, std 1.0 | mean 0.07131383, std 0.08554529 | 0.0855 → **1.0000** |
| nDSM (new) | — | mean 0.01007648, std 0.02226195 | — → 1.0000 |

RGB is deliberately left at ImageNet, per the work order's "non-RGB constants", so `6ch_corrected`
differs from `6ch` in the auxiliary bands only. The configs **load** these numbers from the measured
JSONs rather than transcribing them, so they cannot drift.

## 6. Configs and gates

**24 training + 24 inference configs**, weighted only, new filenames only. The frozen 144 configs,
`MANIFEST.md` and `run_spatial_matrix.cmd` were **not** touched — `generate_matrix_configs.py`'s
`main()` now refuses to run without an explicit flag, precisely so it cannot regenerate them.

New artifacts: `configs/matrix_configs/MANIFEST_arms_2026_08.md`,
`configs/matrix_configs/run_arms_2026_08.cmd` (in the configs directory, not the repo root, so it
sits beside the frozen launcher rather than replacing it), and
`configs/matrix_configs/smoke_rgb_ndsm_convnext.ini`.

| gate | result |
|---|---|
| **G-A** n_in=4 forward, **1×4×1000×1000 → 1×11×1000×1000**, all four models | **ALL PASS** |
| **G-B** config validation | **PASS** — 96+96; frozen 72+72 asserted unchanged; new arms n_in 4 and 6 |
| **G-C** split dry-run | **PASS** — 6,439 / 6,437 / 6,438, clean partition, no route leakage, nDSM 19,314/19,314 |
| **G-D** 1-epoch bf16 smoke | **prepared, not launched** |

G-A deliberately uses the real 1000×1000 geometry. `check_n_in_10.py` used a synthetic 256×256,
which is where the incorrect "256×256 tiles" line in the 2026-07-28 handoff originated.

## 7. Not built, and why

**Group 3 of §6 — the resnet34 learning curve** (25/50/75 % of training routes, one fold). It needs
new route-subset training lists, which belong under `logs_and_models/route_class_audit/`. That path
was read-only for this task, so the artifact could not be created. It remains a separate piece of
work and nothing else depends on it.
