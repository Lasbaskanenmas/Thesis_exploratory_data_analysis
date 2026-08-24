# Provenance — pre-fix copy of `sdfi_transforms.py`

## Why this exists

On 2026-08-15 two defects were fixed in
`c:\thesis\ML_sdfi_fastai2\src\ML_sdfi_fastai2\transforms\sdfi_transforms.py`:

1. a `uint8` cast applied to every channel before augmentation, which truncated the float32
   DSM/DTM elevation bands to integer metres;
2. `SegmentationAlbumentationsHorizontalFlip.__init__` never calling `ItemTransform.__init__`, so
   its `split_idx` was discarded and a 5% horizontal flip ran on the validation split.

Both are documented with measurements in
`exploratory_data_analysis/results/findings/2026-08-14_preflight_ndsm.md`.

**This file is the verbatim copy of the version that produced the frozen 72-run spatial matrix.** It
exists so that code is recoverable without depending on a git binary being installed on this machine
(none is). It is a *second* line of defence, not the primary one.

## Primary recoverability (author-confirmed, 2026-08-15)

- Committed at **`1a2640809620b50ea307c1676866c927b2927d3f`**
- Branch **`thesis/spatial-matrix-2026`**
- Pushed to **github.com/Lasbaskanenmas/Thesis_ML_sdfi_fastai2**
- (`refs/heads/main` was at `45f11abb77e810f8b5ba3686d113cf07c182a0e2`)

Read directly from `.git/HEAD` and `.git/refs/heads/`; `git` itself is not installed here, so the
working-tree cleanliness was confirmed by the author rather than by `git status`.

## The copy

| | |
|---|---|
| file | `sdfi_transforms.py.pre-fix-2026-08-15` |
| source | `c:\thesis\ML_sdfi_fastai2\src\ML_sdfi_fastai2\transforms\sdfi_transforms.py` |
| sha256 | `1DD27519BD795065DC3AC750EAC7F62320CD7A899C59BED571EACF58B523F9A9` |
| bytes | 38,582 |
| lines | 651 |
| **source mtime** | **2026-05-27 04:16:30** |

**The mtime predates the matrix run (2026-06-27 → 2026-07-25) by a month.** The file was not touched
between the matrix launching and this copy being taken, so these bytes are exactly what trained the
72 models and produced the 24 pooled cells.

## Restoring

Copy the file back over the source path and drop the `.pre-fix-2026-08-15` suffix, or
`git checkout 1a26408 -- src/ML_sdfi_fastai2/transforms/sdfi_transforms.py`. Nothing here is
required for normal operation; it is an archive.
