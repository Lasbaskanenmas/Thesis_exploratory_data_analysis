@echo off
REM ==============================================================================================
REM  Great Plan 4 -- score ONE finished cell. CPU only. Called by run_plan4_queue.cmd as soon as a
REM  cell's third fold has been inferred; also safe to run by hand:
REM      plan4_score_cell.cmd convnext_upernet_cir
REM
REM  1. pooled out-of-fold score    -> spatial_matrix\convnext_upernet\oof_<cell>\  (new folder only)
REM  2. cross_cell_summary.csv      -> appends the new row (existing rows pass through unchanged)
REM  3. clean rescore               -> exploratory_data_analysis\results\tables\plan4\clean_rescore\
REM  4. clean per-route matrices    -> results\tables\plan4\clean_route_cms.npz (+ exact cross-check)
REM  5. HF backup of the pooled JSON (non-fatal)
REM  Log: exploratory_data_analysis\logs\plan4_score_<cell>.log
REM ==============================================================================================
setlocal
cd /d C:\thesis\ML_sdfi_fastai2
set CELL=%~1
if "%CELL%"=="" (echo usage: plan4_score_cell.cmd ^<cell^> & exit /b 2)
set PY=..\envs\ML_sdfi\python.exe
set EDA=..\exploratory_data_analysis
set LOG=%EDA%\logs\plan4_score_%CELL%.log
set M=../logs_and_models/spatial_matrix/convnext_upernet
set DS=../multi_channel_dataset_creation/example_dataset

echo %date% %time% scoring %CELL% > "%LOG%"
%PY% src/ML_sdfi_fastai2/analyse/pooled_oof_metrics.py --fold_assignment ../logs_and_models/route_class_audit/fold_assignment.csv --all_txt %DS%/data/all.txt --label_folder %DS%/labels/splitted_labels --codes %DS%/labels/codes.txt --pred_fold0 %M%/%CELL%_fold0/models/example_dataset --pred_fold1 %M%/%CELL%_fold1/models/example_dataset --pred_fold2 %M%/%CELL%_fold2/models/example_dataset --out %M%/oof_%CELL% >> "%LOG%" 2>&1 || goto :fail
%PY% %EDA%\scripts\extend_cross_cell_summary.py --write >> "%LOG%" 2>&1
%PY% %EDA%\scripts\plan4_clean_rescore.py --cells %CELL% >> "%LOG%" 2>&1 || goto :fail
%PY% %EDA%\scripts\plan4_pick_production.py --build %CELL% >> "%LOG%" 2>&1 || goto :fail
%PY% src/ML_sdfi_fastai2/analyse/backup_to_hf.py --repo_id Lasbaskanenmas/befaestelsesdata-spatial-matrix --token_file ..\hftoken_write.txt --file %M%/oof_%CELL%/pooled_oof_metrics.json --path_in_repo results/oof/oof_%CELL%.json >> "%LOG%" 2>&1
echo %date% %time% SCORED %CELL% >> "%LOG%"
exit /b 0

:fail
echo %date% %time% SCORING FAILED %CELL% -- see above >> "%LOG%"
exit /b 1
