# Output schema

This document lists every output file produced by the pipeline scripts.

## Step 1 — `scripts/01_extract_roi_tcc.py`

All paths are relative to `--output_dir`.

| file                              | description                                                                  |
|-----------------------------------|------------------------------------------------------------------------------|
| `volume_summary.csv`              | per-case tumour, core15, and safe-rim voxel counts and mm^3 volumes          |
| `tcc_long.csv`                    | per-case per-ROI per-timepoint mean signal and S/S0                          |
| `dynamics_feature_summary.csv`    | per-case scalar dynamics features (TTP, wash-in, wash-out, normalised AUC)   |
| `qc_report.csv`                   | per-case QC flags, geometry status, warnings, and final processing status   |
| `case_json/<pid>.json`            | per-case structured JSON, including geometry summary and warnings           |
| `qc_plots/<pid>.png` (optional)   | per-case QC overlay + TCC plot, written when `--save_qc_plots` is on        |
| `run.log`                         | run log                                                                      |

Schema templates: see [`schemas/volume_summary_schema.csv`](../schemas/volume_summary_schema.csv)
and [`schemas/tcc_long_schema.csv`](../schemas/tcc_long_schema.csv).

## Step 2 — `scripts/02_extract_tofts_pinn_features.py`

| file                              | description                                                                  |
|-----------------------------------|------------------------------------------------------------------------------|
| `aligned_tcc_curves.csv`          | per-case per-ROI fixed-grid mean signal curves                              |
| `concentration_curves.csv`        | per-case per-ROI concentration curves C_t(t)                                 |
| `tofts_pinn_features.csv`         | per-case Tofts parameters (K_trans, k_ep, derived ratios) for dev + test    |
| `pinn_test_features.csv`          | held-out test subset of the above                                           |
| `tofts_pinn_training_summary.json`| configuration + fit-quality summary of the final-stage PINN refit            |
| `tofts_pinn_extract.log`          | run log                                                                      |

Schema template: see [`schemas/tofts_pinn_features_schema.csv`](../schemas/tofts_pinn_features_schema.csv).

## Step 3 — `scripts/03_train_xgboost.py`

| file                              | description                                                                  |
|-----------------------------------|------------------------------------------------------------------------------|
| `xgboost_cv_summary.json`         | CV configuration, per-fold and aggregated OOF metrics                       |
| `xgboost_cv_fold_metrics.csv`     | per-fold metrics in tabular form                                             |
| `xgboost_cv_oof_predictions.csv`  | OOF predictions (development split only)                                     |
| `xgboost_metrics.json`            | combined CV + held-out test metrics, plus selected hyperparameters           |
| `xgboost_predictions.csv`         | OOF + held-out test predictions concatenated                                 |
| `tofts_pinn_features.csv`         | final-stage Tofts feature table (dev + test), if PINN preset is selected    |
| `pinn_test_features.csv`          | held-out test subset of the above                                           |
| `tofts_pinn_training_summary.json`| final-stage PINN fit summary                                                 |
| `aligned_tcc_curves.csv`          | curves saved alongside step-2 artefacts for transparency                     |
| `concentration_curves.csv`        | same                                                                         |
| `train_xgboost.log`               | run log                                                                      |

Schema templates: [`schemas/xgboost_cv_summary_schema.json`](../schemas/xgboost_cv_summary_schema.json)
and [`schemas/xgboost_metrics_schema.json`](../schemas/xgboost_metrics_schema.json).

### A note on per-case prediction files

`xgboost_cv_oof_predictions.csv`, `xgboost_predictions.csv`, and
`pinn_test_features.csv` are per-case prediction tables. They are produced
locally by the pipeline but **must not be committed** to a public fork: the
shipped `.gitignore` excludes them by pattern. The repository carries only
schema templates, never real per-case predictions.

## Step 4 — `scripts/04_evaluate_metrics.py`

| file               | description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `metrics.json`     | metrics computed on the supplied predictions CSV (optionally filtered by split) |


## Additions in the revised release

Step 3 (`03_train_xgboost.py`) additionally writes:

| file | description |
|---|---|
| `xgboost_cv_fold_metrics.csv` | per-fold metrics at the single OOF threshold, best n_estimators, scale_pos_weight |
| `tofts_pinn_oof_features.csv` | fold-wise (out-of-fold) PINN features of the validation cases |
| `final_development_model_inputs.csv` | imputed model inputs of the final development fit (for SHAP) |
| `final_model/` | locked model: `pinn.pt`, `preprocessor.pkl`, `xgboost.json`, `model_card.json` |
| `xgboost_metrics.json` | now includes `cv_fold_metrics_mean_sd`, `cv_oof_metrics_pooled`, `heldout_test_bootstrap_ci` |

| step | file | description |
|---|---|---|
| 5 | `predictions_<cohort>.csv`, `features_<cohort>.csv`, `metrics_<cohort>.json` | external validation (with bootstrap CIs) |
| 6 | `table_ii_performance.csv`, `table_iv_subtypes.csv`, `calibration_bins.csv`, `brier_scores.csv`, `decision_curves.csv`, `fig4_calibration_dca.png` | Table II / IV, Fig. 4 |
| 7 | `shap_values.csv`, `shap_importance.csv`, `fig5a_shap_summary.png` | Fig. 5a |
| 8 | `<output_csv>` (+ `.json` report) | 2048-D frozen visual embeddings |
| 9 | `table_iii_ablation.csv` (+ one step-3 output folder per configuration) | Table III |
| 10 | `sensitivity_results.csv`, `fig5b_sensitivity.png` | Fig. 5b |
