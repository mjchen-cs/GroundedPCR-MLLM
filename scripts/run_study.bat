@echo off
REM ============================================================================
REM  Reproduce the I-SPY2 MLLM study end to end (Windows cmd / PowerShell via "cmd /c").
REM
REM    scripts\run_study.bat <stage>
REM    stages: features | b1 | export | pilot | query | evaluate | all   (see README)
REM
REM  1) Edit the four paths below (no trailing backslash).
REM  2) API keys come from environment variables only. In PowerShell:
REM       $env:ZHIPU_API_KEY="..."; $env:SILICONFLOW_API_KEY="..."
REM     In cmd:   set ZHIPU_API_KEY=...
REM ============================================================================
chcp 65001 >nul
setlocal EnableDelayedExpansion
set STAGE=%1
if "%STAGE%"=="" set STAGE=all

REM ---- edit these -------------------------------------------------------------
if "%DATA_DIR%"=="" set DATA_DIR=D:\data\BreastDCEDL_ISPY2
if "%BREASTDCEDL_REPO%"=="" set BREASTDCEDL_REPO=D:\data\BreastDCEDL
if "%OUT_DIR%"=="" set OUT_DIR=%~dp0..\results
if "%PAPER_DIR%"=="" set PAPER_DIR=%OUT_DIR%\paper
REM ---------------------------------------------------------------------------
if "%ZHIPU_MODEL%"=="" set ZHIPU_MODEL=glm-4.5v
if "%SF_MODEL%"=="" set SF_MODEL=Qwen/Qwen3-VL-32B-Instruct
if "%REPEATS%"=="" set REPEATS=2
if "%WORKERS%"=="" set WORKERS=8
if "%PINN_DEVICE%"=="" set PINN_DEVICE=cpu
if "%FIG1_PID%"=="" set FIG1_PID=ACRIN-6698-102212

cd /d "%~dp0.." || exit /b 1
set META=%OUT_DIR%\metadata\ISPY2\metadata.csv
set R=%OUT_DIR%\ispy2_roi
set M=%OUT_DIR%\mllm
set B=--feature_csv "%R%\dynamics_feature_summary.csv" --volume_csv "%R%\volume_summary.csv" --tcc_csv "%R%\tcc_long.csv" --metadata_csv "%META%" --pinn_device %PINN_DEVICE%

if "%STAGE%"=="features" goto features
if "%STAGE%"=="b1" goto b1
if "%STAGE%"=="export" goto export
if "%STAGE%"=="pilot" goto pilot
if "%STAGE%"=="query" goto query
if "%STAGE%"=="evaluate" goto evaluate
if "%STAGE%"=="all" goto features
echo unknown stage %STAGE% & exit /b 2

:features
if not exist "%META%" (
  python scripts\00_prepare_breastdcedl.py --breastdcedl_metadata "%BREASTDCEDL_REPO%\BreastDCEDL_metadata.csv" --ispy2_clinical "%BREASTDCEDL_REPO%\ISPY2\BreastDCEDL_spy2_tcia_metadata.csv" --output_dir "%OUT_DIR%\metadata" || exit /b 1
)
if not exist "%R%\dynamics_feature_summary.csv" (
  python scripts\01_extract_roi_tcc.py --full_root "%DATA_DIR%" --mask_dir "%DATA_DIR%" --metadata_csv "%META%" --output_dir "%R%" --spacing_source metadata --slice_axis 0 --no-save_qc_plots --n_jobs 0 || exit /b 1
)
if not "%STAGE%"=="all" exit /b 0

:b1
for %%P in (clinical_mp clinical_mp_volume clinical_mp_volume_het main main_mp main_het main_mp_het) do (
  if not exist "%OUT_DIR%\cv\%%P\xgboost_metrics.json" (
    python scripts\03_train_xgboost.py %B% --feature_mode %%P --n_bootstrap 0 --output_dir "%OUT_DIR%\cv\%%P" || exit /b 1
  )
)
python scripts\11_compare_cv.py --runs_dir "%OUT_DIR%\cv" --candidates main main_mp main_het main_mp_het --paper_dir "%PAPER_DIR%" || exit /b 1
set /p BEST=<"%OUT_DIR%\cv\best_preset.txt"
echo B1 preset selected on development CV: %BEST%
if not exist "%OUT_DIR%\b1\heldout_test_shap.csv" (
  python scripts\03_train_xgboost.py %B% --feature_mode %BEST% --run_final_test --output_dir "%OUT_DIR%\b1" || exit /b 1
  python scripts\07_shap_analysis.py --model_dir "%OUT_DIR%\b1\final_model" --model_inputs_csv "%OUT_DIR%\b1\final_development_model_inputs.csv" --output_dir "%OUT_DIR%\b1\shap" || exit /b 1
)
if not exist "%OUT_DIR%\sens\spw1\xgboost_metrics.json" python scripts\03_train_xgboost.py %B% --feature_mode %BEST% --scale_pos_weight 1 --n_bootstrap 0 --output_dir "%OUT_DIR%\sens\spw1" || exit /b 1
if not exist "%OUT_DIR%\sens\depth3\xgboost_metrics.json" python scripts\03_train_xgboost.py %B% --feature_mode %BEST% --max_depth 3 --n_bootstrap 0 --output_dir "%OUT_DIR%\sens\depth3" || exit /b 1
if not exist "%OUT_DIR%\sens\arm\xgboost_metrics.json" python scripts\03_train_xgboost.py %B% --feature_mode main_mp_arm --n_bootstrap 0 --output_dir "%OUT_DIR%\sens\arm" || exit /b 1
if not "%STAGE%"=="all" exit /b 0

:export
if not exist "%M%\cases.jsonl" (
  python scripts\12_export_mllm_inputs.py --data_root "%DATA_DIR%" --metadata_csv "%META%" --xgb_dir "%OUT_DIR%\b1" --output_dir "%M%" --slice_axis 0 || exit /b 1
)
if not "%STAGE%"=="all" exit /b 0
goto query

:pilot
python scripts\13_run_mllm.py --mllm_dir "%M%" --provider zhipu --model %ZHIPU_MODEL% --disable_thinking --split development --limit 20 --repeats 1 --workers %WORKERS% || exit /b 1
python scripts\13_run_mllm.py --mllm_dir "%M%" --provider siliconflow --model %SF_MODEL% --split development --limit 20 --repeats 1 --workers %WORKERS% || exit /b 1
echo Pilot done. Check %M%\runs\*\*.csv (parse_ok, tokens) before running "query".
exit /b 0

:query
for %%S in (development heldout_test) do (
  python scripts\13_run_mllm.py --mllm_dir "%M%" --provider zhipu --model %ZHIPU_MODEL% --disable_thinking --split %%S --repeats %REPEATS% --workers %WORKERS% || exit /b 1
  python scripts\13_run_mllm.py --mllm_dir "%M%" --provider siliconflow --model %SF_MODEL% --split %%S --repeats %REPEATS% --workers %WORKERS% || exit /b 1
)
if not "%STAGE%"=="all" exit /b 0

:evaluate
set /p BEST=<"%OUT_DIR%\cv\best_preset.txt"
python scripts\14_evaluate_mllm.py --mllm_dir "%M%" --paper_dir "%PAPER_DIR%" --b1_preset %BEST% || exit /b 1
python scripts\15_plot_results.py "%M%\evaluation" "%M%\cases.jsonl" "%M%\runs" "%PAPER_DIR%\figures\fig_results" || exit /b 1
for /d %%D in ("%M%\runs\zhipu__*") do set ZDIR=%%D
python scripts\16_plot_workflow.py "%M%\cases.jsonl" "%M%\images\%FIG1_PID%.png" "%R%\tcc_long.csv" "!ZDIR!\M3_heldout_test.csv" %FIG1_PID% "%PAPER_DIR%\figures\fig1_workflow" --eval_dir "%M%\evaluation" --n_repeats %REPEATS% || exit /b 1
python scripts\17_extra_analyses.py --mllm_dir "%M%" --b1_dir "%OUT_DIR%\b1" --roi_dir "%R%" --out_dir "%PAPER_DIR%" || exit /b 1
set REPRO=
if exist "%OUT_DIR%\sens\repro\xgboost_predictions.csv" set REPRO="Re-run of B1 in a second environment=%OUT_DIR%\sens\repro"
python scripts\18_b1_sensitivity.py --b1_dir "%OUT_DIR%\b1" --out_dir "%PAPER_DIR%" --runs "XGBoost without class weighting=%OUT_DIR%\sens\spw1" "XGBoost, tree depth 3=%OUT_DIR%\sens\depth3" "XGBoost + treatment arm (24 inputs)=%OUT_DIR%\sens\arm" %REPRO% || exit /b 1
echo Done. Tables, macros and figures are in %PAPER_DIR%
endlocal
