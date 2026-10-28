# Paper ↔ code map (MLLM manuscript)

Section numbers refer to the submitted manuscript. Output paths are relative to `OUT_DIR` (`results/`) or `PAPER_DIR`.

## Methods

| Paper | What | Code | Output |
|---|---|---|---|
| III-A | I-SPY2 subset of BreastDCEDL, BreastDCEDL train/test flag (784 / 198), MammaPrint, metadata spacing | `scripts/00_prepare_breastdcedl.py`; `src/io_utils.py` (TCIA layout) | `metadata/ISPY2/metadata.csv` |
| III-B, Eq. 1–3 | EDT, core (inner 15 %), safe rim, < 10 core voxels excluded | `src/roi_utils.py`; `scripts/01_extract_roi_tcc.py` | `ispy2_roi/volume_summary.csv`, `tcc_long.csv` |
| III-C, Eq. 4–5 | time grid, Δt = 90 s fallback, validity mask, SPGR conversion | `src/tofts_pinn_utils.py` (`_spgr_signal_to_concentration`, `_build_fixed_time_grid`) | `b1/concentration_curves.csv` |
| III-D, Eq. 6–7 | Tofts-PINN (shared MLP, Softplus, Parker AIF, masked MSE + range penalty), fold-wise training | `src/tofts_pinn_utils.py` (`SharedToftsMLP`, `tofts_forward_torch`, `train_shared_tofts_pinn`) | `b1/tofts_pinn_*` |
| III-E | B1 XGBoost, imputation and early stopping inside folds, 7 feature sets, selection rule | `src/xgboost_utils.py` (`FEATURE_PRESETS`, `run_stratified_kfold_cv`), `scripts/03_train_xgboost.py`, `scripts/11_compare_cv.py` | `cv/`, `b1/` |
| III-E | heterogeneity descriptors (PE histogram, SER, washout, peritumoral shells) | `src/heterogeneity_features.py` | `ispy2_roi/dynamics_feature_summary.csv` (`het_*`) |
| III-E | TreeSHAP out-of-fold and held-out | `src/xgboost_utils.py::tree_shap_frame`; `scripts/07_shap_analysis.py` | `b1/xgboost_cv_oof_shap.csv`, `b1/heldout_test_shap.csv` |
| III-F | models, temperature, repeats, cache | `src/mllm_client.py`; `scripts/13_run_mllm.py` | `mllm/runs/`, `mllm/cache/` |
| III-F | 2×2 image panel | `scripts/12_export_mllm_inputs.py::make_panel` | `mllm/images/<pid>.png` |
| III-F | prompt ladder M1, M2, M3f, M3, M4; output schema; anchoring rate | `src/mllm_prompts.py` (`build_messages`, `pick_examples`); `scripts/14_evaluate_mllm.py` | — |
| III-G, Eq. 8 | stacked fusion | `scripts/14_evaluate_mllm.py` (fusion block) | `mllm/evaluation/prediction_metrics.csv` |
| III-H, Eq. 9 | unsupported numbers, approximations, SHAP contradictions, consistency, Jaccard | `scripts/14_evaluate_mllm.py` (`faithfulness_row`, `_supported`, `_approximates`) | `mllm/evaluation/faithfulness_*.csv` |
| III-I | secondary analyses (i)–(vi) | `scripts/17_extra_analyses.py` | `analysis/*.csv` |
| III-I (vii) | B1 sensitivity, Platt recalibration | `scripts/18_b1_sensitivity.py`; preset `main_mp_arm` | `analysis/b1_sensitivity.csv` |
| III-J | bootstrap CIs (2000), paired DeLong | `scripts/14_evaluate_mllm.py` (`boot_ci`, `delong_test`) | — |

## Tables

| Table | Content | Written by | File (in `PAPER_DIR`) |
|---|---|---|---|
| I | cohort characteristics | 17 | `tables/table_cohort_rows.tex` |
| II | feature sets for B1 (development CV) | 11 | `tables/table_cv_rows.tex` |
| III | discrimination and calibration | 14 | `tables/table_prediction_rows.tex` |
| IV | AUROC within subtypes | 17 | `tables/table_subtype_rows.tex` |
| V | types of unsupported numbers | 17 | `tables/table_errors_rows.tex` |
| VI | verification of image claims | 17 | `tables/table_visual_rows.tex` |
| VII | B1 sensitivity | 18 | `tables/table_b1_sensitivity_rows.tex` |
| VIII, IX | faithfulness, development / held-out | 14 | `tables/table_faithfulness_rows.tex`, `table_faithfulness_test_rows.tex` |
| text | all `\res...` numbers | 14 | `results_macros.tex` |

## Figures

| Figure | Content | Written by | File |
|---|---|---|---|
| 1 | workflow on one held-out case | 16 | `figures/fig1_workflow.pdf` |
| 2 | B1 feature pipeline (schematic) | drawn by hand | `Fig/IEEE-JBHI.png` in the manuscript folder |
| 3 | example input panel | 12 | `mllm/images/<pid>.png` |
| 4 | AUROC, anchoring, unsupported numbers, contradictions | 15 | `figures/fig_results.pdf` |
| 5 | calibration, probability granularity, decision curves | 17 | `figures/fig_calib_dca.pdf` |
| 6 | B1 SHAP and evidence citation | 17 | `figures/fig_shap_citation.pdf` |
