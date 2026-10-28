# Feature definitions

The 12-feature main preset reported in the manuscript is implemented in
[`src/xgboost_utils.py`](../src/xgboost_utils.py) as `MAIN_FEATURE_COLUMNS`
and is selected via `--feature_mode main` in
[`scripts/03_train_xgboost.py`](../scripts/03_train_xgboost.py).

## Tofts pharmacokinetic features (PINN-derived)

All four come from the shared MLP Tofts-PINN, which is constrained to predict
non-negative `K_trans` and `k_ep` and is fit by minimising masked-reconstruction
MSE plus weak parameter-range regularisation.

| feature           | source ROI | parameter                                                       |
|-------------------|------------|------------------------------------------------------------------|
| `core15_Ktrans`   | core15     | volume transfer constant (min^-1)                                |
| `core15_kep`      | core15     | reflux rate constant (min^-1)                                    |
| `safe_rim_Ktrans` | safe-rim   | volume transfer constant (min^-1)                                |
| `safe_rim_kep`    | safe-rim   | reflux rate constant (min^-1)                                    |

`v_e` is recoverable as `K_trans / k_ep` and is exported to
`tofts_pinn_features.csv` under names like `core15_ve`, but it is not part
of the 12-feature main preset (it is present in the
`pinn_tofts_interpretable` augmentation in the source code, which the public
release exposes only via the schema templates).

## Volume features

All volume features come from the spacing-aware ROI partitioning step.

| feature                            | meaning                                                                    |
|------------------------------------|----------------------------------------------------------------------------|
| `tumor_volume_mm3`                 | total tumour volume in mm^3 after binarisation + optional largest-CC filter |
| `core15_volume_mm3`                | volume in mm^3 of the innermost 15% of tumour voxels by EDT                |
| `safe_rim_volume_mm3`              | volume in mm^3 of the safe-rim ROI (outer rim with uncertain-boundary trim) |
| `safe_rim_to_core15_volume_ratio`  | `safe_rim_volume_mm3 / core15_volume_mm3` (safe divide)                    |

## Clinical features

Clinical fields are read from `metadata.csv` and lightly normalised. The
binary fields accept both numeric and textual values (`1` / `0`, `pos` /
`neg`, `yes` / `no`, etc.).

| feature                | type   | meaning                                              |
|------------------------|--------|------------------------------------------------------|
| `clinical_age`         | float  | baseline age in years                                |
| `clinical_hr`          | 0/1    | hormone receptor positive                            |
| `clinical_her2`        | 0/1    | HER2 positive                                        |
| `clinical_triple_neg`  | 0/1    | triple-negative (HR-, HER2-)                         |

## Ablation presets (Table III)

`scripts/03_train_xgboost.py --feature_mode <preset>`; `scripts/09_run_ablations.py`
runs all of them.

| preset                      | columns                                                    | Table III row |
|-----------------------------|------------------------------------------------------------|---------------|
| `ktrans_only`               | core15_Ktrans, safe_rim_Ktrans                             | K_trans only |
| `kep_only`                  | core15_kep, safe_rim_kep                                   | k_ep only |
| `core15_physics_only`       | core15_Ktrans, core15_kep                                  | Core15 physics only |
| `safe_rim_physics_only`     | safe_rim_Ktrans, safe_rim_kep                              | Safe rim physics only |
| `tofts_only`                | 4 Tofts features                                           | All physics |
| `volume_only`               | 4 volume features                                          | Morphological volumes only |
| `tofts_basic_volume`        | 4 Tofts + tumor_volume_mm3                                 | All physics + Basic Volume |
| `tofts_full_volume`         | 4 Tofts + 4 volume                                         | All physics + Full Volume |
| `tofts_deltas_ratios`       | 4 Tofts + rim−core differences and log ratios of K_trans, k_ep | All physics + Deltas & Ratios |
| `main`                      | the 12 features above                                      | Proposed PI-ML |
| `main_plus_simple_dynamics` | 12 + time-to-peak, wash-in, wash-out, normalised AUC per ROI | PI-ML + 8 Simple Dynamics |
| `main` + `--visual_reduction pca/pls` | 12 + 8 fold-wise PCA / PLS components of the 2048-D embedding | PI-ML + Visual Branch |
| `clinical_only`             | 4 clinical features                                        | (reference) |

Derived Tofts contrasts are recomputed after every fold-wise PINN refit.
