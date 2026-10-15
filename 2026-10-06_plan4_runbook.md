# Plan 4 runbook — the author's checklist, 2026-10-06 to 2026-10-13

Every command in order, so the queue can be driven without a live agent session. The decisions
behind each step are in the plan file and in `2026-10-06_plan4_declarations.md`; this file is only
the how. Weekdays checked against the calendar: 06/10 is a Tuesday.

## Before launch (Tue 06/10, or Wed 07/10 morning)

1. **Lock the declaration.** Read `2026-10-06_plan4_declarations.md` sections 0 to 5, amend
   anything, then set the header to `Status: LOCKED, 2026-10-06 <time>` and the YAML line to
   `status: LOCKED`. The queue refuses to start until it does.
2. **Gates** (CPU, ~10 min), must print `ALL GATES PASS`:
   ```
   C:\thesis\envs\ML_sdfi\python.exe C:\thesis\exploratory_data_analysis\scripts\gate_plan4.py
   ```
3. **Dry run of the pick** on the existing cells (CPU, seconds; needs the lock). It writes
   `results\tables\plan4\pick_dryrun_verdict.json`, which is not the pick:
   ```
   C:\thesis\envs\ML_sdfi\python.exe C:\thesis\exploratory_data_analysis\scripts\plan4_pick_production.py --dry-run
   ```

## Launch (Wed 07/10 ~09:00)

4. In a **standalone cmd window**, not the VSCode terminal:
   ```
   C:\thesis\ML_sdfi_fastai2\run_plan4_queue.cmd
   ```
   Watch job 1 (`convnext_upernet_ortorgb_ndsm` fold 0) clear its first epoch on W&B, project
   `thesis spatial matrix`, before leaving it.

## While it runs (Thu 08/10 to Sun 11/10)

- Progress: `C:\thesis\exploratory_data_analysis\logs\plan4_queue.log`. Each cell lands roughly
  every 21 h. Scoring logs: `logs\plan4_score_<cell>.log`, each ending in `SCORED <cell>`.
- After any crash: `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`, and only if
  something lingers, `taskkill /F /IM python.exe /T`. Do not do that while a scoring window is
  still working.
- Declared rule P4: a failed arm is re-run once only if it can finish before the production
  start deadline. Otherwise it is excluded and reported.

## After the five arms (Sun 11/10 ~16:00)

5. The queue runs the pick itself and prints the verdict, also saved in `logs\plan4_pick.log`
   and `results\tables\plan4\pick_verdict.json`.
6. **Launch production. It must START by Mon 12/10 12:00:**
   ```
   C:\thesis\ML_sdfi_fastai2\run_plan4_production.cmd
   ```
   It reads the winner, builds and gates the full-pool config, trains (~10 h) and backs up.

## Consolidate and back up (Mon 12/10, after production lands)

7. Snapshot, then one full E6 sweep over all 42 cells (~2.5 h CPU):
   ```
   robocopy C:\thesis\exploratory_data_analysis\results\tables C:\thesis\exploratory_data_analysis\results\tables\pre_e6_2026-10-12 route_cell_cms.npz route_cell_metrics.csv cross_cell_summary.csv
   C:\thesis\envs\ML_sdfi\python.exe C:\thesis\exploratory_data_analysis\scripts\eda_route_cell_metrics.py --procs 16
   ```
   It must end in `VALIDATION PASSED`, with EXACT MATCH for every cell.
8. Per-fold table:
   ```
   C:\thesis\envs\ML_sdfi\python.exe C:\thesis\exploratory_data_analysis\scripts\assemble_per_fold_macro_iou.py --force
   ```
9. HF, dry run first, then push:
   ```
   C:\thesis\envs\ML_sdfi\python.exe C:\thesis\exploratory_data_analysis\scripts\hf_sync_batch.py --refresh
   C:\thesis\envs\ML_sdfi\python.exe C:\thesis\exploratory_data_analysis\scripts\hf_sync_batch.py --refresh --push --include-models
   ```

## Tue 13/10

Buffer, final backup check, VM empty by end of day.
