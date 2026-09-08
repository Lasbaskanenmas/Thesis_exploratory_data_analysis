# 2026-09-08 — The §4.6 drop was invoked, then withdrawn. D4's n = 96 stands.

**Supersedes and replaces `2026-09-07_D4_amendment_section_4_6_drop.md`, which is removed. That
amendment never took effect: not a single item was scored under it. This document is the record of
both the invocation and the withdrawal, kept so the reasoning survives rather than the file simply
disappearing.**

**Net effect on the protocol: none. The locked D4 sample of 96 tiles, 192 pass-1 items, 12
double-scored tiles and 24 pass-2 items is what will be scored.**

## What happened

**2026-09-07.** With the qualitative scoring six days behind §6's schedule and the 9/9 freeze
approaching, the session proposed invoking Great Plan 3.1 §4.6's pre-agreed drop — *"the qualitative
sample shrinks 96 → 64 tiles with the double-scored share kept"* — and the author authorised it. The
reduced pack was built as a filter over the existing sample: 64 tiles taken as a prefix of the
parent per-(fold, stratum) draw order, 128 pass-1 items, 8 double-scored tiles, 16 rebuilt pass-2
panels.

**2026-09-08.** The author challenged the justification before beginning to score. The challenge was
correct and the drop was withdrawn.

## Why it was withdrawn

**The arithmetic does not support it.** At the intended one-minute-per-item pace the drop saves
about one hour, and costs a third of the incidence base:

| | tiles | pass-1 items | scoring time | double-scored |
|---|---:|---:|---:|---:|
| D4 as locked | 96 | 192 | ~3.2 h | 12 tiles / 24 items |
| §4.6 drop | 64 | 128 | ~2.1 h | 8 tiles / 16 items |
| **difference** | −32 | −64 | **−1.1 h** | −4 tiles / −8 items |

Trading a third of a headline instrument, and taking self-consistency from 24 items to 16 when
Cohen's κ was already thin at 24, is not a proportionate response to recovering one hour.

**§4.6 is an ordered list and its first three items were never dropped.** The order is Boundary IoU,
then the decomposition-arm paragraph, then the 8-tile paragraph, then the qualitative sample. All
three ahead of the sample were **delivered in full** — Boundary IoU became a finding in its own
right, the decomposition arm ran across the whole roster, and the 8-tile inspection produced its
note. Nothing ahead of the qualitative sample in the queue was sacrificed, so reaching item four
skipped three places in a list whose purpose is sequence.

**The constraint §4.6 was written against no longer exists.** On 24/8 the author's attention was
competing with a live GPU queue: launches, gates, monitoring, and scoring cells as they landed. That
is finished. All 37 cells are scored, Part B is closed, metric breadth and the paired-difference
figure are done, and no remaining task depends on the GPU or on anything except reading panels and
writing. The trigger in §3 rule 4 (slippage beyond two days) was met on paper, but the remedy has to
be proportionate to the constraint that actually binds, and the binding constraint is now hours of
writing, not contention for the author's attention.

**Recorded plainly:** the session proposed the drop on a weak justification — its own words at the
time were that 96 "also fits, but with less room if anything slips" — and should have said so rather
than offering it. The author caught it. The correction is the author's, not the session's.

## What was reverted, and what it cost

Nothing was lost, because the reduced pack was built as a **filter** rather than a redraw.
`sample_tiles.csv`, `scoring_sheet.csv`, `scoring_sheet_pass2.csv`, both original sealed keys and
all 216 rendered panels were untouched throughout and are the active pack again.

The 64-scoped artifacts are quarantined, not deleted, under
`results/qualitative_pack/withdrawn_2026-09-08_section_4_6_drop/`:

```
sample_tiles_64.csv, scoring_sheet_64.csv, scoring_sheet_pass2_16.csv,
SEALED_item_key_pass2_64.csv, drop_46_provenance_64.json,
panels_64/ (128 hardlinks), panels_pass2_64/ (16 rendered panels)
```

They are kept so this record is checkable and so nothing is destroyed, and quarantined so the wrong
sheet cannot be opened by mistake. **The only real cost of the episode is 16 pass-2 panels rendered
and now unused, about thirty seconds of compute.**

## What is unaffected

The scoping rule adopted the same day is independent of sample size and stands unchanged:
`2026-09-08_D4_amendment_mode_scoping_rule.md` — every mode is scored only where the GROUND TRUTH
panel is painted, with `not_assessable` available for items carrying too little annotated ground.
Its thin-tile figures have been restated for the full 96.

## Standing position

**D4 is scored as locked.** 96 tiles, 192 pass-1 items from `panels/` and `scoring_sheet.csv`, then
12 tiles and 24 pass-2 items from `panels_pass2/` and `scoring_sheet_pass2.csv` at least 24 hours
later.

§4.6 remains available and unspent. If the timeline does bite later, the honest order is to take its
first three items first — and all three are already delivered, so in practice the qualitative sample
is the only lever left. That should be a decision taken against a real shortfall, with the hour it
buys weighed against the third of the sample it costs, and not before.
