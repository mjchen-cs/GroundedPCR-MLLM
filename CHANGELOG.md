# Changelog

## GroundedPCR-MLLM release (MLLM manuscript, I-SPY2)
- New prompt configuration **M3f** (physics features without B1 output) and the **anchoring rate**.
- `scripts/14_evaluate_mllm.py`: thousands separators are removed and ranges such as "10-15%" are no longer read as negative numbers; rounded or unit-converted volumes are counted as approximations rather than unsupported numbers.
- New steps:
  - `15_plot_results.py`: main results figure;
  - `16_plot_workflow.py`: Fig. 1 drawn from a real case;
  - `17_extra_analyses.py`: cohort table, subtype AUROC, calibration and decision curves, evidence citation, error taxonomy, image-claim verification, repeat stability;
  - `18_b1_sensitivity.py`: B1 recalibration, logistic regression, no class weighting, depth 3, + treatment arm (preset `main_mp_arm`);
  - `19_make_release_bundle.py`: release bundle of derived results.
- `run_study.sh` / `run_study.bat` replace the machine-specific Windows batch files; paths and models come from environment variables.
- `check_api.py --list_models`; progress output in steps 12 and 13; default number of repeats is 2.
- `tests/test_mllm_pipeline.sh`: offline end-to-end test with the mock model.
- Documentation:
  - `README.md` with full reproduction steps;
  - `docs/REPRODUCTION_GUIDE.md` (step-by-step guide);
  - `docs/paper_to_code_map.md` for the MLLM manuscript; the documentation of the original manuscript moved to `docs/legacy/`.
- `CITATION.cff` and `.env.example`; `.gitignore` excludes data, results and API caches.

## Revised release (aligned with the JBHI manuscript)

### Bug fixes
- `src/roi_utils.py`: `getattr(np, "trapezoid", np.trapz)` raised `AttributeError`
  on NumPy >= 2.0 (the default argument is evaluated eagerly), so step 1 failed
  for every case. Resolved lazily now.
- `scripts/03_train_xgboost.py`: step-1 `dynamics_feature_summary.csv` already
  contains `pCR` / `test`; merging with metadata produced `pCR_x` / `pCR_y` and
  `KeyError: 'pCR'`. Labels, split flag and clinical covariates are now taken
  from `metadata.csv` only (`src/feature_table.py`).
- Cases absent from `metadata.csv` are no longer scored.

### Behaviour now matching the manuscript
- **Median imputation** (Sec. II-E): fitted on the training fold / full
  development set and *applied* to validation / held-out / external data
  (previously re-fitted on each split). `src/xgboost_utils.py::TabularPreprocessor`.
- **Class imbalance** (Sec. II-E): `scale_pos_weight` defaults to
  `n_negative / n_positive` of the training data (previously fixed at 1.0).
- **`max_delta_step`** (Sec. II-E): now passed to XGBoost (default 1.0).
- **Early stopping** (Sec. II-E): enabled inside CV (100 rounds) and monitored
  on an inner stratified 15% split of the training fold, never on the
  validation fold; final `n_estimators` = mean fold-wise best iteration
  (previously no early stopping, so always 1500 trees).
- **Fold-wise metrics** (Sec. II-H, Table II): computed at the single
  OOF-derived threshold and summarised as mean ± SD (previously each fold used
  a threshold optimised on its own validation data).
- **Exclusion rule** (Sec. II-A): cases with < 10 core voxels are excluded
  (`--min_core_voxels`; previously only a warning on total tumour voxels).

### New functionality
- Locked final model saved to `<output_dir>/final_model/` (`src/locked_model.py`).
- `scripts/05_external_validation.py`: I-SPY 1 / Duke inference with the locked model.
- `scripts/06_cohort_statistics.py`: Table II, Table IV, calibration / Brier, DCA (Fig. 4).
- `scripts/07_shap_analysis.py`: TreeSHAP summary (Fig. 5a).
- `scripts/08_extract_visual_embeddings.py`: frozen RadImageNet-ResNet50 embeddings.
- `scripts/09_run_ablations.py` + new presets: Table III.
- `scripts/10_sensitivity_analysis.py`: Fig. 5b.
- `--core_fraction` (step 1), `--aif_delay_seconds`, `time_shift_seconds` hook.
- `tests/smoke_test.sh`: synthetic end-to-end run of steps 1-10.

### BreastDCEDL compatibility
- `scripts/00_prepare_breastdcedl.py`: per-cohort metadata from `BreastDCEDL_metadata.csv`.
- Step 1 `--spacing_source metadata --slice_axis 0`: BreastDCEDL NIfTI headers
  carry 1 mm isotropic spacing; true spacing comes from `xy_spacing` /
  `slice_thick` / `voxel_z`. Checked on the repository sample cases: tumour
  volumes now agree with BreastDCEDL `tum_vol` (e.g. 30.2 vs 31.1 cm³; header
  spacing gave 41.2 cm³).
- Flat `dce/` layout and I-SPY 1 `_acq<t>` / `_spy1_vis1_mask` names supported.
- Step 8 `--slice_axis`.
- Step 1 `--n_jobs` (process-parallel cases; identical outputs to the serial run).
- TCIA package layout `<root>/<pid>/dce|mask/`.

### Performance-improvement study (exploratory, development CV only)
- `src/heterogeneity_features.py` + step 1 `--heterogeneity_features` (default on): voxel-level
  percent-enhancement histogram, SER / washout fractions, core-vs-rim contrast, peritumoral
  0-3 mm and 3-6 mm shells (`het_*` columns).
- Step 0 `--ispy2_clinical`: MammaPrint (`clinical_mp`) and treatment-arm (`arm_*`) columns.
- Presets `clinical_mp`, `clinical_mp_volume`, `clinical_mp_volume_het`, `main_mp`, `main_het`,
  `main_mp_het`, `main_mp_het_arm`; `scripts/11_compare_cv.py`;
  (never touches the held-out test set).
- Early-stopping metric default changed from AUC to logloss (AUC on ~100 inner cases stopped
  some folds after 3-5 trees).
- Fixed `tofts_pinn_features.csv` export missing K_trans / k_ep (merge name clash).

### Physics-grounded MLLM study (I-SPY2)
- `src/mllm_client.py`: OpenAI-compatible client for Zhipu and SiliconFlow (+ offline `mock`), retries,
  atomic on-disk reply cache; keys from `ZHIPU_API_KEY` / `SILICONFLOW_API_KEY` only.
- `src/mllm_prompts.py`: fixed prompt ladder M1-M4 and JSON output schema; few-shot references drawn
  from other folds only.
- `src/xgboost_utils.py`, `scripts/03_train_xgboost.py`: out-of-fold and held-out TreeSHAP and raw
  model inputs are written (`xgboost_cv_oof_shap.csv`, `heldout_test_shap.csv`, `*_inputs.csv`).
- `scripts/11_compare_cv.py`: `--candidates`, `--paper_dir` (LaTeX rows), `best_preset.txt`.
- `scripts/12_export_mllm_inputs.py`, `13_run_mllm.py`, `14_evaluate_mllm.py`,
  `run_study.bat`: panel export, queries, DeLong / bootstrap / stacked fusion, faithfulness
  metrics and auto-generated LaTeX tables and macros for the manuscript.

### Removed
- Empty stray file `src/1`; README references to non-existent `configs/`.

### Defaults that the manuscript should state explicitly
The manuscript names these settings but not their values; please make the text
match whatever was actually used:
`early_stopping_rounds = 100`, inner early-stopping split = 15%,
`max_delta_step = 1.0`, `scale_pos_weight = n_neg/n_pos`, XGBoost learning
rate 0.02, subsample 0.85, colsample_bytree 0.75, `min_child_weight = 2`,
γ = 0.1, λ = 6, α = 0.5; PINN: 2 hidden layers (ReLU), Adam 1e-3, weight
decay 1e-5, batch 32, ≤ 250 epochs, patience 25, gradient clip 5; default
TR = 4 ms, flip angle 15°, T10 = 1.6 s, r1 = 4.5 L mmol⁻¹ s⁻¹.
