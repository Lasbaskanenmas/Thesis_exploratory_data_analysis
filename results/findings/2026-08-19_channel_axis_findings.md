# The channel axis — resolved

**Every figure below is read from `pooled_oof_metrics.json` on disk, not from any summary.** Cells are
directly comparable: all report 9/9 classes, 19,314 tiles, 12,194,633,781 evaluated pixels, fold sizes
6,439 / 6,437 / 6,438, same frozen split, same weighted CE, pure bf16.

**Verdict up front.** The `RGB ≥ multi-channel` ordering in the frozen matrix is **real, and it survives
correcting the encoding**. The mechanism is not that elevation is uninformative — it is that
**absolute elevation is a spatial confound under route blocking**, and the encoding bug was *muting*
it. Fixing the encoding let the network use the confound, and generalisation got worse.

---

## 1. The ledger

### ConvNeXt+UPerNet

| cell | Macro-IoU | macro-F1 | overall acc |
|---|---:|---:|---:|
| `rgb_ndsm` | **0.3625** | 0.4639 | 0.9337 |
| `rgb` (frozen) | 0.3586 | 0.4593 | 0.9336 |
| `6ch` broken (frozen) | 0.3374 | 0.4258 | 0.9275 |
| **`6ch_corrected`** | **0.3110** | 0.3975 | 0.9267 |

### resnet34+UNet (the production architecture)

| cell | Macro-IoU | macro-F1 | overall acc |
|---|---:|---:|---:|
| `rgb` (frozen) | 0.2695 | 0.3454 | 0.8973 |
| `6ch` broken (frozen) | 0.2477 | 0.3134 | 0.8977 |
| **`6ch_corrected`** | **0.2025** | 0.2594 | **0.8797** |

**Correcting the constants made both models worse, at both ends of the capacity range:**

| model | 6ch broken → corrected | vs its own `rgb` |
|---|---:|---:|
| ConvNeXt | −0.0264 (−7.8 %) | −0.0476 |
| resnet34 | −0.0452 (−18.2 %) | −0.0670 |

## 2. The reductio: the corrected production model is barely above a constant predictor

`unet_resnet34_6ch_corrected` reports **overall accuracy 0.8797**. The trivial "always answer
ubefestet" predictor, scored through the same metric code, reports **0.8747363**
(`performance_bounds.csv`, `B1_constant_majority`).

**The gap is 0.0050 — half a percentage point.** A production-architecture model, given every channel
KDS pays to acquire, correctly normalised, trained under an honest spatial protocol, lands half a
point above answering the same word for every pixel. Its Macro-IoU of 0.2025 against the trivial
predictor's 0.0972 is the only thing separating them, which is precisely the SQ1 F1 argument for
per-category metrics, now sharpened.

## 3. The confound exhibit: per-fold behaviour

A channel that genuinely helps improves every held-out fold. A channel that encodes *where the tile
is* helps the folds it was fitted on and hurts the one held out. The per-fold diagnostics show the
second pattern.

### ConvNeXt macro-IoU per held-out fold

| cell | fold 0 | fold 1 | fold 2 |
|---|---:|---:|---:|
| `rgb` | 0.3302 | 0.3307 | 0.3750 |
| `6ch` broken | 0.3159 | 0.3185 | 0.3351 |
| **`6ch_corrected`** | **0.2669** | **0.3549** | **0.3378** |

Correcting the constants **raised folds 1 and 2** (0.3185 → 0.3549, 0.3351 → 0.3378) and **collapsed
fold 0** (0.3159 → 0.2669). The pooled number falls because one fold falls hard. That is the
signature of a feature that fits the training distribution and fails to transfer — not of a feature
that is simply weak.

### betonflade per held-out fold — the sharpest exhibit

| cell | pooled | fold 0 | fold 1 | fold 2 |
|---|---:|---:|---:|---:|
| convnext `rgb` | 0.1526 | 0.1955 | 0.0601 | 0.0110 |
| convnext `6ch` broken | 0.1363 | 0.1485 | 0.0831 | 0.0247 |
| **convnext `6ch_corrected`** | **0.0084** | **0.0000** | 0.0389 | **0.0002** |
| resnet `rgb` | 0.0301 | 0.0354 | 0.0148 | 0.0000 |
| resnet `6ch` broken | 0.0001 | 0.0000 | 0.0000 | 0.0003 |
| **resnet `6ch_corrected`** | **0.0002** | 0.0000 | 0.0000 | 0.0011 |

Support is identical across all cells (67,320,009 / 20,068,763 / 1,871,601 px), so this is not a
support artefact.

**Stated precisely, because the two models differ.** For ConvNeXt this is a genuine new collapse:
betonflade goes 0.1363 → **0.0084** pooled, and 0.1485 → **0.0000** on fold 0. For resnet34 betonflade
had *already* collapsed in the broken 6ch cell (0.0001), so the corrected arm does not newly break it
— it fails to rescue it (0.0002). The exhibit is that in **both** corrected cells betonflade is
effectively zero on every fold, and in ConvNeXt's case correcting the channels is what destroyed it.

betonflade is the class this should hit hardest: per SQ1 F8 its mean nDSM is **0.19 m**, the lowest of
the nine — a flat concrete slab is defined by being at ground level, so a model keying on absolute
elevation has nothing to hold on to, and a model keying on *relative* height has almost no signal.

`solceller` shows the same direction and is not marginal: convnext 0.7967 → **0.6446**,
resnet 0.5882 → **0.3597**.

## 4. Why this is a confound and not weak data

Three measurements converge:

1. **Absolute elevation is mostly a location code.** SQ1 F8: DSM carries **96.6 %** of its variance
   between tiles rather than within them, DTM **99.6 %**. Between-tile elevation is a proxy for where
   in Denmark the tile sits. Route blocking deliberately puts held-out routes in *different places*,
   so that component cannot transfer by construction.
2. **The encoding bug was suppressing it.** In the frozen runs those channels reached the network at
   an effective std of **0.093 (DSM)** and **0.086 (DTM)** against RGB's 0.868, *and* were truncated
   to integer metres. The network could largely ignore them. Correcting both — unit variance, full
   float32 — is what let it start using them, and using them hurt.
3. **Removing the absolute component removes the harm.** nDSM is object height, and after clamping
   carries only **48.0 %** of its variance between tiles versus DSM's 96.6 %
   (`ndsm_clamped_stats.json`). `rgb_ndsm` scores **0.3625** against `rgb`'s 0.3586 — **+0.0039**,
   inside noise. Strip the location code out of elevation and the multi-channel penalty disappears,
   but no real gain appears either.

**So the honest statement to KDS is not "the auxiliary channels are useless".** It is: *elevation as
currently supplied encodes location more than structure, and under a spatially honest protocol that
makes it actively harmful; supplying object height instead is safe but did not help on this data.*
The earlier reading — that the frozen `RGB ≥ 6ch` ordering was probably a normalisation artefact —
is now **falsified**. Correcting the normalisation moved the result the other way.

**What is still untouched:** the 671 tiles (3.47 %) whose DSM/DTM are georeferenced to different
ground. That defect is identical in `6ch` and `6ch_corrected`, so it does not explain the drop
between them, but it bounds what any elevation arm on this data can achieve.

**What this does not overturn:** the `10ch − 6ch` contrast still isolates the oblique source with all
defects held constant, and remains usable.

## 5. Why Swin and SegFormer corrected arms were not run

**Deliberate, not an omission.** Both configs exist, are validated, and are ready in
`2026-08-17_6ch_corrected_launch_blocks.md`.

The two arms that were run sit at **opposite ends of the capacity range** — ConvNeXt is the best cell
in the entire matrix (0.3374 broken) and resnet34 is the weakest and the production architecture
(0.2477 broken). Both dropped, in the same direction, with the same per-fold signature and the same
betonflade behaviour. The mechanism in §4 is measured independently of either run, from the variance
decomposition and the nDSM control.

Running Swin (0.2956) and SegFormer (0.2687) would place two more points *between* two replications
that already agree, at roughly **56 GPU-hours** (2 models × 3 folds × ~9.3 h). Against a real
deadline that buys interpolation, not a new fact, and the compute is better spent on the route-level
statistics and the learning curve. Should a reviewer press, the blocks are ready and the argument
does not depend on the answer.

## 6. Artifacts

- `cross_cell_summary.csv` — extended additively with `convnext_upernet_rgb_ndsm`,
  `convnext_upernet_6ch_corrected`, `unet_resnet34_6ch_corrected`. ~~`route_*` columns are empty for
  these rows because module E6 has only been run for `convnext_upernet_rgb`; they are not invented.~~
  **Superseded 2026-08-23:** the file now carries **all 27 cells** (24 frozen + 3 arms) with every
  `route_*` column populated. Module E6 was run over the full matrix and validated element by
  element against all 27 frozen `global_confusion_matrix` arrays. The pooled figures quoted above
  are unchanged — they were re-read from the same JSONs, not recomputed.

  **Flagged for §5, not resolved here.** At route level `convnext_upernet_6ch_corrected` has a
  *higher* median macro-IoU (0.4982) than `convnext_upernet_rgb` (0.4902), the opposite direction to
  the pooled result (0.3110 vs 0.3586). This is not a contradiction: route-level macro-IoU averages
  only the classes **present in that route**, so a class that collapses to zero almost everywhere —
  betonflade, per §3 — is dropped from the routes that lack it instead of scoring 0. The pooled
  metric counts it in full. Which quantity the §5 Wilcoxon should test is therefore a live question
  that the route matrices now make answerable; the pooled conclusion in §1–§4 stands on its own
  metric and is not amended by this.
- `performance_bounds.csv` — the trivial and oracle bounds used in §2.
- `ndsm_clamped_stats.json` — the 48.0 % between-tile share used in §4.
- Pooled scores: `logs_and_models/spatial_matrix/<model_dir>/oof_<cell>/pooled_oof_metrics.json`.
