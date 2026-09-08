# 2026-09-08 — Dated amendment to D4: the scoring region is the annotated footprint

**Recorded under D4 item 9, before any item was scored. Author decision, 2026-09-08: Option A.**

## The ambiguity this closes

D4 defines mode 3 as *"a false-positive island of a class absent or negligible in the tile's ground
truth"*. That is well defined where annotation exists. But class 0 is deliberately left unpainted in
the GROUND TRUTH panel, and E4 measured the ignore share at **11.80 % to 89.33 % by route** — so on
many tiles most of the panel carries no ground truth, while the model predicts everywhere. D4 does
not say whether a prediction sitting on unannotated ground can trigger a mode.

**The ambiguity is not confined to mode 3.** It is identical for **m4** (paved-class swap), **m6**
(vegetation-occlusion error) and the "missed" half of **m8**, all of which need ground truth to
establish that an error occurred. It is absent for **m1** (speckle), **m2** (boundary bleeding),
**m7** (misregistration echo) and the "hallucinated" half of **m8**, which are judged against the
imagery. One rule is therefore declared for all nine modes rather than a patch for mode 3.

## The rule

**Every mode is scored only where the GROUND TRUTH panel is painted. The annotated footprint is the
scoring region for the whole taxonomy.**

Predictions falling on unannotated ground are not scored and do not trigger any mode.

## Why this and not the alternative

It keeps the two evaluation legs measuring the same ground. The quantitative leg drops every
ignore-labelled pixel from every metric, so a qualitative leg that scored unannotated ground would
be reporting failure modes on pixels no number in the thesis is computed over.

It also preserves what mode 3 is for. Mode 3 exists as the **D3 blindness catcher**: the per-route
metric averages only the classes present in a route, so a false positive on a class with no route
support vanishes from the route score. Those false positives sit on **annotated** ground — that is
what makes them invisible to the route metric while still counting in the pooled metric. A patch
over unannotated ground is invisible to the pooled metric too, so scoring it would not close the gap
mode 3 was written to close.

The rejected alternative, scoring the whole tile, would have meant judging predictions against the
scorer's own reading of the aerial image rather than against ground truth. That is a different
instrument from the one D4 declares, and it would have made the incidence table a mixture of two.

## Consequence accepted, and stated so it is not discovered later

**Modes that do not need ground truth are under-counted relative to a whole-tile reading.** Speckle,
boundary bleeding and misregistration echo are all visible in unannotated ground and will not be
recorded there. This is a deliberate bound on what the leg measures, not an oversight, and the
incidence table must be read as *"failure modes on annotated ground"* rather than *"failure modes"*.

Anything striking seen in unannotated ground goes to **`m9_other` with a one-line note**, which D4
already provides for (*"anything else goes to mode 9's one-line note"*). That captures the
observation without inventing a mode and without contaminating the incidence counts. Those notes can
support a sentence in the discussion; they are not measured incidence and must not be reported as
such.

## Unscorable items — a direct consequence of the rule

Rule A makes it possible for an item to have too little annotated ground to assess. Measured across
the **96 sampled tiles** (D4 as locked; the §4.6 drop to 64 was invoked on 7/9 and withdrawn on 8/9
before any scoring — see `2026-09-08_D4_section_4_6_drop_invoked_and_withdrawn.md`):

| annotated pixels of 1,000,000 | tiles |
|---|---:|
| **27 px (0.003 %)** | **1** |
| 9,053 px (0.91 %) | 1 |
| under 10,000 px (1 %) | 2 |
| under 50,000 px (5 %) | 7 |
| under 200,000 px (20 %) | 22 |
| median | 474,541 px (47.5 %) |

**One tile carries 27 annotated pixels — roughly a 5 × 5 pixel patch, 50 cm across.** Both of its
items are unassessable under this rule, and a second tile at 9,053 px is close behind. Twenty-two of
the ninety-six sit under 20 % annotated, so the provision below will see real use. Scoring them as nine zeros and `dominant_mode = none` would
record "no errors found" when the truth is "nothing could be assessed", which would bias every
incidence proportion downward.

**Provision:** `dominant_mode` may take the value **`not_assessable`**, with all nine mode columns
left `0`. This is not a tenth mode and adds nothing to the closed taxonomy — it marks an item as
outside the assessable set. No pixel threshold is imposed; the scorer decides from the panel, since
the scorer can see how much painted ground there is and a fixed cut-off would be arbitrary.

**Reporting duty:** items marked `not_assessable` are excluded from the denominator, and every
incidence figure is reported as *"x of N assessable"* with N and the excluded count both stated. The
count of unassessable items is itself reportable — it is a direct measure of how much of the sampled
product carries no ground truth to check it against, which is E4's coverage finding seen from the
qualitative side.

## Scope

Nothing else in D4 changes. Same two cells, same nine-mode closed taxonomy, same dominant-mode tag,
same blinding, same seed, same fixed exemplar rule, same operational-consequence mapping. This
amendment sits alongside `2026-09-07_D4_amendment_section_4_6_drop.md`; the two are independent and
both precede scoring.
