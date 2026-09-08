# 2026-09-07 — Dated amendment to D4: the Great Plan 3.1 §4.6 drop, 96 → 64 tiles

**Required by D4 item 9: "any departure from this protocol is recorded in a dated amendment before
the results are used." This is that record, written before a single item was scored.**

## Authority

Great Plan 3.1 §4.6 fixes the drop order in advance, for exactly this situation: *"then the
qualitative sample shrinks 96 → 64 tiles with the double-scored share kept"*. It is a **pre-agreed**
fallback, decided on 2026-08-24 when nothing about the outcome was known, and invoked by the author
on 2026-09-07. It is not a decision taken under pressure with results in view.

The trigger is the schedule. §5.3 scheduled the blind scoring for 31/8–1/9; on 7/9 both sheets were
still empty, and the 9/9 results freeze allows pass 1, a 24-hour gap and pass 2 only if pass 1 is
short enough to finish now. §4.6 lists the qualitative leg itself under **"never dropped"**, so the
sample shrinks rather than the leg.

## What changed

| | D4 as locked | after the §4.6 drop |
|---|---:|---:|
| tiles | 96 | **64** |
| pass-1 items | 192 | **128** |
| per fold | 32 (16 + 16) | **22 / 22 / 20** (11+11, 11+11, 10+10) |
| weak-present total | 48 | **32** |
| complement total | 48 | **32** |
| double-scored tiles | 12 (12.5 %) | **8 (12.5 %)** |
| pass-2 items | 24 | **16** |

Everything else in D4 is untouched: the same two cells, the same nine-mode closed taxonomy, the same
dominant-mode tag, the same blinding, the same fixed exemplar rule, the same operational-consequence
mapping, the same seed.

## How the 64 were chosen, and why it is a filter rather than a redraw

**The 64 are a subset of the already-persisted 96, taken as a prefix of the original per-(fold,
stratum) draw order.** No new randomness enters the sample. `sample_tiles.csv`, the 192 rendered
pass-1 panels and both original sealed keys remain on disk, untouched, as the parent record.

Redrawing 64 tiles under the seed would have been the obvious move and would have been wrong: it
replaces a sample that was drawn and persisted before any panel was rendered with one drawn after
the pack existed. The prefix rule keeps the pre-registration property intact and makes the reduction
reproducible from `sample_tiles.csv` alone, with no seed required.

**The allocation arithmetic, stated because 64 does not divide evenly into D4's structure.** D4 fixes
three folds of 32, half weak-present and half complement. 64 cannot be three equal folds. The
allocation is 22 / 22 / 20 across folds 0 / 1 / 2, with the weak and complement halves held exactly
equal *within* every fold, so the totals stay 32 and 32. The reduction falls on the highest fold
index by a stated deterministic rule, chosen so that it is arbitrary-but-fixed rather than selected
for any property of the data.

## The double-scored share, and a correction made during the build

§4.6 says the *share* is kept. D4 states 12 of 96 as "12.5 %", so at 64 tiles the share is **8 tiles
and 16 items**, not 12.

The first build produced only 14 items. Seven of the original twelve double-scored tiles survived the
truncation, and the eighth had to be topped up from the retained set — but that tile had no pass-2
panel, because pass 2 was only ever rendered for the original twelve. Reusing just the survivors
would have quietly delivered 7 tiles and a 10.9 % share while the provenance claimed 12.5 %.

**Fix:** pass 2 was rebuilt for all eight chosen tiles — both cells per tile, reshuffled under the
declared seed 20260825, into a fresh `S###` id space, with all sixteen panels rendered. The original
24-item pass 2 stays on disk, unused. No scoring work was invalidated, because none had been done.

The eighth tile is the first tile in retained draw order that was not already in the original twelve.

## What was not re-rendered

Nothing in pass 1. All 128 in-scope panels already existed and are hardlinked into `panels_64/`, so
the author has a clean folder of exactly the items to score at zero additional disk. Only the sixteen
rebuilt pass-2 panels were rendered.

## Effect on what the leg can show

Honest statement of the cost, so it is not discovered later:

- **Incidence counts lose a third of their base.** A mode seen in 8 of 96 items becomes roughly 5 of
  64. Proportions stay estimable; rare modes get noisier and should be reported with counts beside
  every proportion, not proportions alone.
- **Self-consistency is measured on 16 items rather than 24.** Cohen's κ was already going to be
  thin at 24 and is thinner at 16. The declaration's instruction stands and matters more now:
  report raw percent agreement, and κ only where prevalence permits a stable estimate. Where it does
  not, say so rather than quoting an unstable κ.
- **Fold 2 contributes 20 tiles against the other two folds' 22.** Small, but it should be stated
  wherever per-fold breakdowns appear.
- **Stratification is unharmed.** The weak-present and complement halves stay exactly balanced within
  every fold and across the sample.

## Files

Additive; nothing from the parent pack was modified or deleted.

```
sample_tiles_64.csv                 the retained 64, with fold and stratum
scoring_sheet_64.csv                128 rows, ids pre-filled, in the existing shuffled order
scoring_sheet_pass2_16.csv          16 rows, fresh S### ids
SEALED_item_key_pass2_64.csv        DO NOT OPEN until scoring is complete
panels_64/                          128 hardlinks to the existing pass-1 panels
panels_pass2_64/                    16 rebuilt pass-2 panels
drop_46_provenance_64.json          allocation, method, blinding statement, this amendment's terms
```

## Blinding

Constructing the subset required mapping item ids to tiles, so the build read `SEALED_item_key.csv`.
That does not affect the author's blinding: the author has not opened it, and **no item → (tile,
cell) mapping appears in any output, console line or document produced by this amendment.** The
sheets carry item ids only.
