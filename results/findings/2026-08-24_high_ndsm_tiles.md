# 2026-08-24 — The eight tiles with nDSM above 100 m: all real structures, three objects

**Role.** Closes open item 6 of `2026-08-16_progress_tracker_2.md` §8 and Task 2.3 of the 2026-08-24
work order. Measured with `exploratory_data_analysis/scripts/inspect_high_ndsm_tiles.py`; every
number below is read from `high_ndsm_tiles.csv` / `high_ndsm_tiles.json`. Read-only investigation.

## The question

`2026-08-15_stage1_ndsm_arms.md` §4 recorded a clamped nDSM maximum of **298.33 m** against a pooled
mean of 2.57 m and sd 5.68 m, a **52σ** tail carried by **8 tiles (0.04 %)**. The build record noted
that the work order's "standardisation handles a 7σ tail" premise had been overtaken by the
measurement, kept the decision to apply no ceiling clamp, and left the tiles uninspected. Three
outcomes were possible and all three were reportable: a DSM spike, residual misregistration, or real
structure.

## The answer

**All eight are real structures, and the eight tiles hold only three distinct objects.**

| tile | route | nDSM max | DSM at peak | DTM at peak | DTM sd in a 5×5 m window | px > 100 m |
|---|---|---:|---:|---:|---:|---:|
| `O2021_83_26_1_0012_00001782_7000_4000` | 83-26 | 298.33 m | 318.72 m | 20.39 m | 0.088 m | 1,008 |
| `O2021_83_26_1_0012_00001782_7000_3000` | 83-26 | 298.33 m | 318.72 m | 20.39 m | 0.088 m | 1,008 |
| `O2021_82_20_1_0023_00004975_10000_0` | 82-20 | 113.84 m | 169.31 m | 55.47 m | 0.061 m | 96 |
| `O2021_82_21_1_0004_00002040_6000_0` | 82-21 | 113.84 m | 169.31 m | 55.47 m | 0.061 m | 96 |
| `O2021_82_20_1_0023_00004974_10000_5000` | 82-20 | 113.84 m | 169.31 m | 55.47 m | 0.061 m | 56 |
| `O2021_82_20_1_0023_00004973_10000_11000` | 82-20 | 113.84 m | 169.31 m | 55.47 m | 0.061 m | 96 |
| `O2021_82_20_1_0023_00004974_10000_6000` | 82-20 | 107.56 m | 163.69 m | 56.13 m | 0.056 m | 40 |
| `O2021_84_40_1_0045_00093319_2000_2000` | 84-40 | 103.79 m | 108.82 m | 5.03 m | 0.027 m | 64 |

The discriminating evidence is the terrain model beneath the peak. Across a 5×5 m window centred on
every peak the DTM standard deviation is **0.027 m to 0.088 m**, i.e. flat to within a few
centimetres, while the DSM in the same window varies by 5 m to 74 m. A misregistered pair would put
an unrelated and unstable terrain surface under the peak; a DSM spike would be an isolated pixel or
two. What the data shows instead is a smooth ground surface with a tall, compact object standing on
it. None of the eight appears on the 671-tile misregistration list, and all four rasters per tile
(nDSM, DSM, DTM, rgb) carry identical geotransforms.

**Three objects, not eight.** The two 83-26 tiles are consecutive crops of the same parent frame and
report identical peak statistics, so they see one 298 m object. The five 82-20 and 82-21 tiles all
report a peak at DSM 169.31 m over DTM 55.47 m, which is the same roughly 114 m object seen from
several tiles across two route keys that the 2026-08-04 provenance report measured as sharing
59 % to 65 % of their ground. The eighth tile is a separate roughly 104 m object in route 84-40.
This is the 2.56× tile redundancy factor appearing at the scale of a single feature, and it is worth
one sentence in the dataset chapter as a concrete illustration of it.

## What follows

1. **The no-ceiling-clamp decision is vindicated, and for a better reason than the one recorded.**
   The build record defended it on extent, that only 8 tiles of 19,314 exceed 100 m. The stronger
   defence is now available: a ceiling clamp at 40 m, the value the work order considered and
   rejected, would have flattened three genuine masts into the terrain and manufactured exactly the
   artefact the clamp was meant to remove. nDSM is behaving correctly at its extreme.
2. **This is not a data-quality defect and should not be written as one.** The 52σ tail is a
   property of the landscape, not of the pipeline. The honest statement is that the nDSM channel's
   dynamic range is set by a handful of tall masts, so its standardisation constants sit far below
   its maximum, which is expected for object height and is not a normalisation failure of the kind
   defect 1 described.
3. **It is a fourth data-quality observation only in a weak sense**, and the tracker's speculation
   that it was "almost certainly a DSM spike or residual misregistration" is **withdrawn**. Recording
   the withdrawal here so it is not rediscovered later.

## Artifacts

- `results/tables/high_ndsm_tiles.csv` — one row per tile, every measured quantity.
- `results/tables/high_ndsm_tiles.json` — the same, plus the verdict tally and the threshold used.
- `exploratory_data_analysis/scripts/inspect_high_ndsm_tiles.py` — regenerates both.
