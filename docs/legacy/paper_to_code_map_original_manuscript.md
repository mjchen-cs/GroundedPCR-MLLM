# Manuscript-to-code map

| Manuscript | What | Code |
|---|---|---|
| Sec. II-A | cohort exclusion (< 10 core voxels) | `src/feature_table.py::apply_core_voxel_exclusion`, `--min_core_voxels` (steps 3, 5) |
| Sec. II-A, II-F | held-out split `test in {1,2}` | `src/tofts_pinn_utils.py::HELDOUT_TEST_VALUES`, `scripts/03_train_xgboost.py::_split_dev_test` |
| Sec. II-B, Eq. 1 | spacing-aware EDT | `src/roi_utils.py::compute_edt_distances` (`ndi.distance_transform_edt(..., sampling=spacing)`) |
| Sec. II-B, Eq. 2 | core15 (α = 0.15) | `src/roi_utils.py::build_core_mask_by_fraction`, `--core_fraction` |
| Sec. II-B, Eq. 3 | safe rim (β = 0.10, γ = 0.20) | `src/roi_utils.py::build_safe_rim_mask_by_fraction` |
| Sec. II-C | ROI mean curves S(t) | `scripts/01_extract_roi_tcc.py` → `tcc_long.csv` |
| Sec. II-C | time axis (metadata or Δt = 90 s), fixed grid, validity mask V(t) | `src/tofts_pinn_utils.py::_resolve_time_axis_from_metadata`, `_build_fixed_time_grid`, `_interpolate_with_nan_edges` |
| Sec. II-C, Eq. 4-7 | SPGR signal → concentration | `src/tofts_pinn_utils.py::_spgr_signal_to_concentration` |
| Sec. II-C, Eq. 8 | linearised approximation | `src/tofts_pinn_utils.py::_linearized_signal_to_concentration` (`--concentration_mode linearized`) |
| Sec. II-D | shared MLP, input 2N_t+1, hidden 32 | `src/tofts_pinn_utils.py::SharedToftsMLP` |
| Sec. II-D, Eq. 9 | Softplus + offset | `SharedToftsMLP.forward` |
| Sec. II-D, Eq. 10 | Tofts convolution, trapezoidal | `src/tofts_pinn_utils.py::tofts_forward_torch` |
| Sec. II-D | Parker population AIF / CSV AIF | `parker_population_aif`, `_load_fixed_aif_from_csv` |
| Sec. II-D, Eq. 11-12 | masked MSE + weak range regulariser | `masked_reconstruction_mse`, `weak_parameter_range_regularizer`, `train_shared_tofts_pinn` |
| Sec. II-E | 12-feature vector | `src/xgboost_utils.py::MAIN_FEATURE_COLUMNS` |
| Sec. II-E | median imputation fitted on training data only | `src/xgboost_utils.py::TabularPreprocessor` |
| Sec. II-E, Eq. 13 | XGBoost configuration, `scale_pos_weight`, `max_delta_step`, early stopping, final n_estimators | `build_xgb_classifier`, `fit_xgb_with_inner_early_stopping`, `run_stratified_kfold_cv`, `fit_final_model_and_evaluate` |
| Sec. II-F | fold-wise PINN refits, OOF features | `run_stratified_kfold_cv` (`tofts_pinn_oof_features.csv`) |
| Sec. II-F, II-H | F1-optimal OOF threshold | `src/metrics_utils.py::select_best_threshold` |
| Sec. II-F | single final refit of PINN + XGBoost on development | `scripts/03_train_xgboost.py` (`--run_final_test`), saved by `src/locked_model.py::save_locked_model` |
| Sec. II-G | frozen RadImageNet-ResNet50, pseudo-RGB patches, 2048-D | `scripts/08_extract_visual_embeddings.py` |
| Sec. II-G | fold-wise PCA / PLS (8-D) | `TabularPreprocessor` (`--visual_reduction pca|pls`) |
| Sec. II-H | development mean ± SD across folds | `xgboost_cv_fold_metrics.csv`, `fold_metrics_mean_sd` in `xgboost_cv_summary.json` |
| Sec. II-H | 10,000 bootstrap 95% CIs | `src/metrics_utils.py::bootstrap_metric_ci` (steps 3, 5, 6, 10) |
| Sec. II-H | subtype stratification | `src/metrics_utils.py::assign_molecular_subtype`, `scripts/06_cohort_statistics.py` |
| Sec. II-H | calibration / Brier, DCA | `calibration_table`, `brier_score`, `decision_curve`; `scripts/06_cohort_statistics.py` |
| Table I | cohort characteristics | derived from `metadata.csv`, `volume_summary.csv` (tumour volume), `n_times` |
| Table II | internal + external performance | steps 3, 5 → `scripts/06_cohort_statistics.py` → `table_ii_performance.csv` |
| Table III | feature-group ablation | `FEATURE_PRESETS` / `TABLE_III_PRESETS`; `scripts/09_run_ablations.py` → `table_iii_ablation.csv` |
| Table IV | subtype-stratified AUROC / AUPRC | `scripts/06_cohort_statistics.py` → `table_iv_subtypes.csv` |
| Fig. 3b | masked reconstruction MSE per ROI | `core15_recon_mse`, `safe_rim_recon_mse` in `tofts_pinn_features.csv` |
| Fig. 4 | calibration + DCA | `scripts/06_cohort_statistics.py` → `fig4_calibration_dca.png` |
| Fig. 5a | SHAP summary | `scripts/07_shap_analysis.py` → `fig5a_shap_summary.png` |
| Fig. 5b | sensitivity (temporal sampling, ROI, AIF timing, noise) | `scripts/10_sensitivity_analysis.py` → `fig5b_sensitivity.png` |

Not covered by this repository: rigid co-registration of DCE phases (Sec. II-B;
must be done upstream, before step 1) and nnU-Net training /
inference / Dice evaluation (Sec. III-B).
