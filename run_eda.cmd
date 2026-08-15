@echo off
REM ==============================================================================================
REM  Befaestelsesdata EDA -- full run, in dependency order.
REM
REM  Everything here is CPU and IO only. No GPU, no training, and nothing outside
REM  exploratory_data_analysis\ is written. Source data and logs_and_models\ are read-only.
REM
REM  Runtime: E0 ~20 min, E3/E5 minutes, E2 ~30 min, E4 ~30 min, E1 ~15 min, E6 several hours.
REM  Run a single module instead by calling its script directly; each supports --help.
REM ==============================================================================================
setlocal
set PY=c:\thesis\envs\ML_sdfi\python.exe
set S=%~dp0scripts
cd /d %~dp0

echo(
echo ============ E0  tile inventory (prerequisite for everything) ============
%PY% "%S%\eda_tile_inventory.py" --procs 32 || goto :failed

echo(
echo ============ E3  spatial dependence ============
%PY% "%S%\eda_spatial_dependence.py" || goto :failed

echo(
echo ============ E5  performance bounds ============
%PY% "%S%\eda_bounds.py" || goto :failed

echo(
echo ============ E2  channel audit (full-pool scan, ~460 GB of reads) ============
%PY% "%S%\eda_channel_stats.py" --procs 24 || goto :failed

echo(
echo ============ E1  separability and the linear probe ============
%PY% "%S%\eda_separability.py" --procs 16 || goto :failed

echo(
echo ============ E4  label quality ============
%PY% "%S%\eda_label_quality.py" --procs 16 || goto :failed

echo(
echo ============ E6  per-route confusion matrices ============
echo  Validating on one cell first -- this must report EXACT MATCH before the full sweep.
%PY% "%S%\eda_route_cell_metrics.py" --cells convnext_upernet_rgb --procs 16 || goto :failed
echo(
echo  Single-cell validation passed. Running all 24 cells (several hours).
%PY% "%S%\eda_route_cell_metrics.py" --procs 16 || goto :failed

echo(
echo ============ figures ============
%PY% "%S%\eda_figures.py" --thesis || goto :failed

echo(
echo ============ DONE ============
echo Tables  : results\tables
echo Figures : results\figures
echo Findings: results\findings
goto :eof

:failed
echo(
echo *** FAILED at the step above. Nothing outside this folder was modified. ***
exit /b 1
