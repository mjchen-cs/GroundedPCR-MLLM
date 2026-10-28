# Validation and leakage control

The pipeline implements a **two-tier evaluation protocol** that is designed
to be leakage-safe for both the Tofts-PINN and the XGBoost classifier.

## Tier 1: held-out test split

The held-out test set is fixed by the `test` column of `metadata.csv`:

- `test == 0` (or missing) — development split
- `test == 1` or `test == 2` — held out

Held-out cases are excluded from the development dataframe before any
model selection takes place. They are **never seen** by:

- the 5-fold CV procedure,
- any fold-wise PINN training inside CV,
- the threshold search on OOF predictions,
- or the final-stage PINN refit when computing the released
  `tofts_pinn_features.csv`.

The held-out split is consulted only at the very end, when
`scripts/03_train_xgboost.py` is invoked with `--run_final_test`.

The constant `HELDOUT_TEST_VALUES = {1, 2}` is defined in
[`src/tofts_pinn_utils.py`](../src/tofts_pinn_utils.py) and re-used by
the XGBoost script, so the development / test boundary is identical
across the two stages.

## Tier 2: stratified 5-fold CV on the development split

Within the development split:

1. A `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` is built
   on the binary `pCR` label.
2. For each of the 5 folds:
   - **Fold-wise PINN refit.** The Tofts-PINN is retrained from scratch on
     the fold's training PIDs only. It is then used to infer K_trans /
     k_ep for the fold's training and validation PIDs. Validation cases
     therefore never contribute to the parameters of the PINN that scores
     them.
   - **Preprocessing.** The median imputer (and, for the exploratory visual
     branch, the StandardScaler + PCA/PLS) is fitted on the fold's training
     cases and applied unchanged to the validation cases.
   - **XGBoost fit.** `scale_pos_weight = n_neg / n_pos` of the training
     fold. Early stopping (100 rounds) is monitored on an inner stratified
     15% split of the *training* fold; the validation fold is never used for
     model selection.
   - The fold's validation predictions are accumulated as out-of-fold
     (OOF) predictions on the development split.
3. After all folds, the OOF predictions are pooled and a single decision
   threshold is searched on them (F1-maximising threshold over a
   `[0.05, 0.95]` grid in steps of 0.01). Fold-wise metrics are then computed
   at that single threshold and summarised as mean ± SD.

## Final-stage refits

When `--run_final_test` is passed:

1. The Tofts-PINN is refit one final time on the **full** development
   split. It is then applied to both development and held-out PIDs to
   produce the released `tofts_pinn_features.csv`. The held-out test
   PIDs do not contribute to this PINN's parameters.
2. The imputer and XGBoost classifier are refit on the full development
   split using the final-stage Tofts features (no early stopping;
   `n_estimators` = mean fold-wise best iteration). The held-out test set is
   scored under the threshold chosen in step 3 above.
3. The locked model (PINN weights, curve scale, time grid, AIF, imputer,
   booster, threshold) is written to `final_model/`. External cohorts
   (`05_external_validation.py`) and perturbed inputs
   (`10_sensitivity_analysis.py`) are scored with it without any refitting;
   their curves are aligned to the training time grid and missing values are
   filled with the development medians.

## What this protocol guarantees

| component                                    | sees development PIDs? | sees held-out PIDs? |
|----------------------------------------------|------------------------|---------------------|
| Per-fold PINN (one of 5)                     | only its fold's train  | never               |
| Per-fold XGBoost (one of 5)                  | only its fold's train  | never               |
| Final-stage PINN                             | yes (all of dev)       | never               |
| Final-stage XGBoost                          | yes (all of dev)       | never               |
| Per-fold imputer / visual PCA-PLS            | only its fold's train  | never               |
| Per-fold early-stopping monitor              | inner split of train   | never               |
| Final-stage imputer                          | yes (all of dev)       | never               |
| Decision-threshold search on OOF predictions | yes (all of dev)       | never               |
| External / sensitivity inference            | n/a (locked model)     | scored only         |

## What this protocol does NOT guarantee

- It does not protect against **dataset-level leakage**. If the same
  patient appears under different PIDs in development and held-out, that
  is a data-curation concern, not something the script can detect.
- It does not protect against **label noise** in `pCR`. Garbage in /
  garbage out applies.
- The fold-wise PINN refit is stochastic: changing `--random_state` will
  perturb both the fold assignment and PINN initialisation. Reported
  numbers in the manuscript correspond to `--random_state 42` and the
  default hyperparameters declared in
  [`scripts/03_train_xgboost.py`](../scripts/03_train_xgboost.py).
