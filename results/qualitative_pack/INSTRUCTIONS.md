# Qualitative error analysis — scoring instructions (locked declaration D4)

**Everything in this pack was generated from `2026-08-25_pre_declarations.md`, Status LOCKED, seed
20260825. The sample was drawn and written to `sample_tiles.csv` before a single panel was
rendered. Re-running the builder reproduces the sample to an identical SHA-256, so the pack is
recoverable from the seed alone.**

**Do not open `SEALED_item_key.csv` or `SEALED_item_key_pass2.csv` until scoring is complete.** They
hold the item → (tile, cell) mapping. Opening either one ends the blinding.

---

## What you are looking at

Each panel is one **item** = one tile scored under one of the two declared cells. Three sub-panels,
left to right, 1000 × 1000 px each at 0.1 m:

| panel | what it is |
|---|---|
| **IMAGERY** | the `rgb` tile, unmodified |
| **GROUND TRUTH** | the same tile with the annotation painted over it at 45 % opacity |
| **PREDICTION** | the same tile with that cell's held-out prediction painted the same way |

`legend.png` is the colour key. Keep it open in a second window.

**Class 0 (`unknown`) is deliberately left unpainted.** So in the GROUND TRUTH panel, anywhere that
looks like plain imagery carries **no annotation at all**. That is not a rendering gap: E4 measured
that 94.20 % of the ignore mass is unannotated ground rather than annotator uncertainty, and roughly
37 % of the label pixels in the pool are ignore. The prediction panel, by contrast, is painted
everywhere, because the model predicts everywhere.

### One ambiguity D4 does not resolve — please decide before you start

Predictions exist in regions where no ground truth does. D4's mode 3 is defined as "a false-positive
island of a class **absent or negligible in the tile's ground truth**", which is well defined where
annotation exists and undefined where it does not.

**Suggested reading, which you should confirm or overrule:** judge every mode only against the
annotated footprint — the painted part of the GROUND TRUTH panel — and ignore predictions falling
in unannotated ground entirely. That keeps the qualitative leg consistent with the quantitative one,
which drops ignore pixels from every metric.

This is an interpretation, not a protocol change, so nothing has been altered. If you would rather
score the whole tile, say so and it is recorded as a dated amendment under D4 item 9 before any
result is used.

---

## The nine modes — verbatim from D4, closed, no additions

Score **present / absent** per item, as `1` or `0`:

| column | mode |
|---|---|
| `m1_speckle` | salt-and-pepper speckle (isolated single-class pixels/specks) |
| `m2_boundary_bleed` | boundary bleeding (class runs across a visible surface edge) |
| `m3_spurious_patch` | spurious minority-class patch (false-positive island of a class absent or negligible in the tile's ground truth) |
| `m4_paved_swap` | paved-class swap (asfalt/fliser/grus/brosten/betonflade confusion over paved ground) |
| `m5_road_continuation` | road/driveway continuation error (carriageway class extended into entrances, or cut) |
| `m6_veg_occlusion` | vegetation-occlusion error (surface under canopy mislabelled) |
| `m7_misregistration` | misregistration echo (prediction pattern visibly offset from image structure) |
| `m8_large_object` | large-object failure (solar farm, greenhouse, green roof missed, fragmented or hallucinated) |
| `m9_other` | other — put the description in `note` |

Then **`dominant_mode`**: the single mode you judge most consequential for this item. Write the
column name (`m4_paved_swap`) or just the number (`4`). One per item, always filled, even when only
one mode fired. If nothing at all is wrong, leave every mode `0` and put `none` in `dominant_mode`.

**No new modes may be added after lock.** Anything that does not fit goes in `m9_other` with a
one-line note.

---

## Files

```
results/qualitative_pack/
  panels/            item_0001.png ... item_0192.png     pass 1, 192 items
  panels_pass2/      item_R001.png ... item_R024.png     pass 2, 24 items
  scoring_sheet.csv          192 rows, one per item      <- you fill this
  scoring_sheet_pass2.csv     24 rows                    <- you fill this, >= 24 h later
  sample_tiles.csv            the 96 drawn tiles, persisted before rendering
  legend.png                  class colour key
  pack_provenance.json        seed usage order, strata, mode definitions, blinding statement
  build_log.txt               the build transcript
  SEALED_item_key.csv         DO NOT OPEN until scoring is complete
  SEALED_item_key_pass2.csv   DO NOT OPEN until scoring is complete
```

626 MB total. Panels are 3020 × 1046 px PNG.

---

## How to work through it efficiently

Plan 3.1 §5.3 budgets two half-day blocks for pass 1. At 192 items that is **about one minute per
item**, which is the right pace — this is a triage instrument, not a forensic one. First impression,
mark, move on.

1. **Open `panels/` in an image viewer with keyboard next/previous**, sorted by filename. Windows
   Photos works; IrfanView or similar is better because space-bar advances and it will not re-scale
   between images. Put `legend.png` on a second monitor or a second window.
2. **Open `scoring_sheet.csv` beside it.** It is already in item order with the ids pre-filled, so
   you never have to type an id — just move down the rows in step with the viewer.
3. **Score in the given order. Do not sort.** The order is the declared shuffle and it is what keeps
   the two cells of a tile from being compared against each other.
4. **Split the 192 into two blocks of 96** across two sittings, per §5.3, rather than one long run.
   Scoring fatigue is the bias D4's fixed order and hidden identity are there to bound, and a break
   in the middle is the cheapest further protection.
5. **Do not go back and revise earlier items** after you have developed a sharper eye later in the
   run. Consistency drift is exactly what the pass-2 re-score is designed to measure; smoothing it
   by revision destroys the measurement.
6. **Pass 2 comes at least 24 hours after pass 1**, from `panels_pass2/` into
   `scoring_sheet_pass2.csv`. Those 24 items are 12 tiles you have already scored, re-shuffled under
   new ids. You are not expected to remember them, and you should not try to.

**Two adjacent same-tile pairs exist in the pass-1 order** — items where the two cells of one tile
happen to land next to each other. That is what a plain shuffle produces and D4 declares a plain
shuffle, so it was reported rather than engineered away. If you notice a repeat, score it as you see
it; do not try to recall what you gave the neighbour.

When both sheets are filled, tell me and I will unseal the keys, build the incidence table
(mode × cell), compute the self-consistency figures, and pick the exemplar panels by D4's fixed rule
— first tile in sampled order exhibiting the mode in the ConvNeXt cell, falling back to the
production cell, never hand-picked.

---

## What the sample looks like (no outcome information)

96 tiles, 32 per fold, 16 weak-class-present and 16 complement in each, drawn without replacement.
The weak-class stratum is any tile whose ground truth contains at least one pixel of `green_roof`,
`drivhus`, `betonflade` or `brosten`.

| fold | held-out tiles | weak-present available | complement available |
|---|---:|---:|---:|
| 0 | 6,439 | 1,719 | 4,720 |
| 1 | 6,437 | **239** | 6,198 |
| 2 | 6,438 | 713 | 5,725 |

Worth knowing when the incidence table is read: the weak classes are heavily concentrated in folds 0
and 2. Fold 1 offered only 239 candidate tiles, so its 16 weak-stratum draws come from a much thinner
pool, and the stratum is a larger fraction of what exists there. This does not bias the draw — it was
uniform within the stratum — but it does mean fold 1's weak-class panels are less representative of
"a typical weak-class tile" than the other two folds'.
