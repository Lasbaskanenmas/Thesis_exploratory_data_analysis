# Exploratory Data Analysis — Befæstelsesdata

Characterization of the Befæstelsesdata semantic-segmentation task as a statistical-learning
problem. This is the empirical answer to **SQ1** of Great Plan 3.0:

> What are the properties of the Befæstelsesdata classification task as a statistical-learning
> problem, and what do these properties imply for which learning and evaluation methods are
> appropriate and what performance is attainable?

SQ1 does double duty. It is a result in its own right, and it is what *earns* the evaluation
instrument in SQ2. Without it, "why spatial cross validation, why per category metrics" rests on
citing Roberts, Karasiak and Wang rather than on demonstrating the need on KDS's own data.

---

## The rule this repository is built on

**Additive only.** Nothing outside this folder is ever written, moved or deleted. The scripts read
`logs_and_models/` and `multi_channel_dataset_creation/example_dataset/` strictly as input and fail
loud rather than repairing anything in place. In particular they never regenerate the frozen fold
assignment, never touch the 72 trained models or the 794 per-epoch checkpoints, and never overwrite
the 24 `pooled_oof_metrics.json` files (module E6 validates itself *against* those files, so they
must stay intact).

---

## Layout

```
exploratory_data_analysis/
  scripts/            eda_*.py, one per module, plus eda_common.py   [tracked]
  results/tables/     derived CSV / JSON / small NPZ                 [tracked]
  results/figures/    PNG at 150 dpi, PDF at 300 dpi for thesis use  [tracked]
  results/findings/   markdown write-ups, one per module + synthesis [tracked]
  cache/              large regenerable intermediates                [gitignored]
  run_eda.cmd         the runner
```

### Size guard

Nothing committed here approaches GitHub's 100 MB per-file limit, and no Git LFS is required. The
largest tracked artifact is `results/tables/tile_inventory.csv` at roughly 5 MB. **Refuse to commit
any file over 50 MB** — if one appears, it belongs in `cache/`. Check before committing:

```powershell
Get-ChildItem -Recurse -File . |
  Where-Object { $_.Length -gt 50MB -and $_.FullName -notmatch '\\cache\\|\\\.git\\' } |
  Select-Object FullName, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
```

Every tracked table is a derived artifact, fully regenerable from the scripts plus the untouched
source data. This repository is a record of findings, not a data store.

---

## Modules

| ID | Script | What it establishes |
|---|---|---|
| E0 | `eda_tile_inventory.py` | The per-tile join table (19,314 rows) every other module groups by. No per-tile record existed anywhere before this. |
| E1 | `eda_separability.py` | Class co-occurrence, blob sizes, spectral signatures, and a context-free linear probe floor. |
| E2 | `eda_channel_stats.py` | Full-pool channel statistics, the normalization audit, nDSM, channel redundancy, rgb-vs-OrtoRGB provenance. Gates finding F4. |
| E3 | `eda_spatial_dependence.py` | Semivariogram, Moran's I, the autocorrelation range, and the naive-split leakage demonstration. The empirical warrant for spatial CV. |
| E4 | `eda_label_quality.py` | Decomposition of the 36.9 percent `unknown` mass, ignore-mask bias, selection bias, and boundary ambiguity. Gates finding F5b. |
| E5 | `eda_bounds.py` | Trivial and attainable performance bounds, so 0.24 to 0.36 Macro-IoU can be read against something. |
| E6 | `eda_route_cell_metrics.py` | 24 cells x 16 routes confusion matrices, cached. Feeds both the geographic-variation question and the Part B route-level statistics. |

---

## Reproducing

Requires the project environment at `c:\thesis\envs\ML_sdfi` (Python 3.10; numpy, scipy, pandas,
matplotlib, rasterio, geopandas, sklearn, cv2 all present). `C:\thesis\ML_sdfi_fastai2\src` is a
permanent `sys.path` entry via editable install, so the scripts import the existing analysis
machinery directly and duplicate none of it.

Run everything in dependency order:

```
run_eda.cmd
```

Or a single module:

```
c:\thesis\envs\ML_sdfi\python.exe scripts\eda_tile_inventory.py
```

Every script supports `--help`, and those with non-trivial logic support `--selftest`, which runs
on synthetic data and touches no real files.

**Runtime.** E0 is 15 to 30 minutes. E3, E5 and E1 are seconds to minutes. E2 is a full-pool scan of
all six channel folders, roughly 460 GB of reads, 2 to 5 hours. E6 reads the 463,536 out-of-fold
predictions, 4 to 8 hours. Both long scans are CPU and IO only and run unattended. Nothing in this
folder requires a GPU.

---

## Validation

Each module is checked against artifacts that already exist, so a silent error cannot pass:

- **E0** must reproduce `class_pixel_audit.json` per-class totals exactly (green_roof 1,430,413 px
  across 90 tiles; ubefestet 10,667,088,919 px), reproduce `route_class_audit.csv` per-route totals
  exactly, and produce fold counts 6,439 / 6,437 / 6,438 summing to 19,314.
- **E2** recomputes the rgb band means on a 2,000-tile subsample and compares against the full-pool
  figure.
- **E5** must reproduce the trivial baseline by hand from `class_pixel_audit.json`.
- **E6** must reproduce each cell's `global_confusion_matrix` element-wise by summing that cell's 16
  route matrices. This single assertion validates the route-to-fold-to-prediction-folder routing,
  the tile partition and the accumulation together.
