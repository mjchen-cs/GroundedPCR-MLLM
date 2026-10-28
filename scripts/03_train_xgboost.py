#!/usr/bin/env python3
"""
Step 3: Train and evaluate the XGBoost pCR classifier.

Implements the leakage-safe protocol of Sec. II-E / II-F:

1. Merge dynamics + volume + clinical tables (clinical covariates, labels and
   the split flag come from metadata.csv); exclude cases with fewer than
   ``--min_core_voxels`` core voxels.
2. Held-out split: ``test in {1, 2}`` never enters model selection.
3. Stratified 5-fold CV on the development split. Per fold:
   - Tofts-PINN refit on the fold's training PIDs only;
   - median imputer (and optional visual PCA/PLS) fitted on the training fold
     and applied to the validation fold;
   - XGBoost with scale_pos_weight = n_neg / n_pos of the training fold and
     early stopping monitored on an inner split of the training fold.
4. Single decision threshold = F1-maximising threshold on pooled OOF
   probabilities; fold-wise metrics are reported at that threshold
   (mean +/- SD).
5. ``--run_final_test``: refit PINN + imputer + XGBoost on the full
   development split (n_estimators = mean fold-wise best iteration), score the
   held-out test set (with bootstrap 95% CIs), and save the locked model to
   ``<output_dir>/final_model`` for external validation / sensitivity runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from feature_table import (
    add_subtype_column,
    apply_core_voxel_exclusion,
    load_and_merge_features,
    visual_columns,
)
from locked_model import FINAL_MODEL_DIRNAME, save_locked_model
from metrics_utils import bootstrap_metric_ci
from tofts_pinn_utils import (
    HELDOUT_TEST_VALUES,
    add_shared_tofts_pinn_args,
    build_final_tofts_feature_table,
    pinn_config_from_args,
    prepare_curve_records,
    save_curve_artifacts,
    train_pinn_and_infer_for_pid_sets,
)
from xgboost_utils import (
    FEATURE_PRESETS,
    PINN_COLUMNS,
    feature_columns_for_preset,
    fit_final_model_and_evaluate,
    preset_needs_pinn,
    refresh_tofts_derived_columns,
    run_stratified_kfold_cv,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: 5-fold CV + (optional) held-out test evaluation of an "
            "XGBoost pCR classifier on the Tofts + volume + clinical features."
        )
    )

    # Inputs.
    parser.add_argument("--feature_csv", type=Path, required=True,
                        help="Path to dynamics_feature_summary.csv from step 1.")
    parser.add_argument("--volume_csv", type=Path, required=True,
                        help="Path to volume_summary.csv from step 1.")
    parser.add_argument("--metadata_csv", type=Path, required=True,
                        help="Path to metadata.csv (clinical fields, labels, splits).")
    parser.add_argument("--tcc_csv", type=Path, required=True,
                        help="Path to tcc_long.csv from step 1 (used for fold-wise PINN refit).")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Output directory for CV summary, metrics, and predictions.")

    # Column names.
    parser.add_argument("--pid_col", type=str, default="pid", help="Case ID column.")
    parser.add_argument("--label_col", type=str, default="pCR", help="Binary label column.")
    parser.add_argument("--test_col", type=str, default="test", help="Held-out test indicator column.")

    # Feature preset.
    parser.add_argument("--feature_mode", type=str, default="main",
                        choices=sorted(FEATURE_PRESETS.keys()),
                        help="Feature preset. 'main' is the 12-feature manuscript preset.")

    # CV / threshold.
    parser.add_argument("--n_splits", type=int, default=5, help="Number of stratified CV folds.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    parser.add_argument("--threshold_metric", type=str, default="f1",
                        choices=["f1", "balanced_accuracy"],
                        help="Metric used to select the decision threshold from OOF predictions.")
    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.01)

    # XGBoost hyperparameters.
    parser.add_argument("--n_estimators", type=int, default=1500)
    parser.add_argument("--max_depth", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.02)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample_bytree", type=float, default=0.75)
    parser.add_argument("--colsample_bylevel", type=float, default=0.90)
    parser.add_argument("--colsample_bynode", type=float, default=0.90)
    parser.add_argument("--min_child_weight", type=float, default=2.0)
    parser.add_argument("--reg_lambda", type=float, default=6.0)
    parser.add_argument("--reg_alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--max_delta_step", type=float, default=1.0,
                        help="Caps the magnitude of each logistic update (Sec. II-E).")
    parser.add_argument("--scale_pos_weight", type=str, default="auto",
                        help="'auto' = n_negative / n_positive of the training data (up-weights pCR); or a number.")
    parser.add_argument("--early_stopping_rounds", type=int, default=100,
                        help="Early-stopping patience inside CV (0 disables).")
    parser.add_argument("--es_val_fraction", type=float, default=0.15,
                        help="Inner stratified fraction of each training fold used to monitor early stopping.")
    parser.add_argument("--eval_metric", type=str, default="logloss",
                        help="Early-stopping metric on the inner split (logloss is more stable than AUC on ~100 cases).")
    parser.add_argument("--tree_method", type=str, default="hist")
    parser.add_argument("--n_jobs", type=int, default=1)

    # Cohort definition.
    parser.add_argument("--min_core_voxels", type=int, default=10,
                        help="Exclude cases with fewer valid voxels in the core ROI (Sec. II-A). 0 disables.")

    # Exploratory visual branch (Sec. II-G, Table III).
    parser.add_argument("--visual_embedding_csv", type=Path, default=None,
                        help="Per-case embeddings (pid + vis_* columns) from scripts/08_extract_visual_embeddings.py.")
    parser.add_argument("--visual_reduction", type=str, default=None, choices=["pca", "pls"],
                        help="Fold-wise reduction of the visual embeddings appended to the selected preset.")
    parser.add_argument("--visual_n_components", type=int, default=8)

    # Statistics.
    parser.add_argument("--n_bootstrap", type=int, default=10000,
                        help="Bootstrap resamples for held-out 95%% CIs (0 disables).")

    # Held-out test.
    parser.add_argument("--run_final_test", action="store_true",
                        help="If set, refit on the full development split and score the held-out test set.")
    parser.add_argument("--no_save_final_model", action="store_true",
                        help="Do not write the locked final model to <output_dir>/final_model.")

    # Tofts-PINN configuration (shared across scripts 02 and 03).
    add_shared_tofts_pinn_args(parser)

    return parser


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_xgboost")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    fh = logging.FileHandler(output_dir / "train_xgboost.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def _split_dev_test(df: pd.DataFrame, label_col: str, test_col: str, logger: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_col in df.columns:
        test_mask = pd.to_numeric(df[test_col], errors="coerce").isin(list(HELDOUT_TEST_VALUES))
    else:
        test_mask = pd.Series(False, index=df.index)
    dev_df = df.loc[~test_mask & pd.to_numeric(df[label_col], errors="coerce").notna()].copy()
    test_df = df.loc[test_mask].copy()
    logger.info("Split sizes | development=%d | held-out test=%d | test rule: %s in %s",
                len(dev_df), len(test_df), test_col, sorted(HELDOUT_TEST_VALUES))
    return dev_df, test_df


def _parse_scale_pos_weight(value: str) -> float | None:
    if value is None or str(value).strip().lower() == "auto":
        return None
    return float(value)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(args.output_dir)

    feature_cols = feature_columns_for_preset(args.feature_mode)
    logger.info("Feature preset: %s | %d columns: %s", args.feature_mode, len(feature_cols), feature_cols)
    if args.visual_reduction and args.visual_embedding_csv is None:
        raise ValueError("--visual_reduction requires --visual_embedding_csv.")

    df = load_and_merge_features(
        feature_csv=args.feature_csv,
        volume_csv=args.volume_csv,
        metadata_csv=args.metadata_csv,
        pid_col=args.pid_col,
        label_col=args.label_col,
        test_col=args.test_col,
        visual_embedding_csv=args.visual_embedding_csv if args.visual_reduction else None,
        logger=logger,
    )
    vis_cols = visual_columns(df) if args.visual_reduction else []
    if args.visual_reduction:
        logger.info("Visual branch: %d embedding columns -> %s(%d), fitted fold-wise.",
                    len(vis_cols), args.visual_reduction.upper(), args.visual_n_components)

    df, excluded = apply_core_voxel_exclusion(df, args.min_core_voxels, pid_col=args.pid_col, logger=logger)

    for col in PINN_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    dev_df, test_df = _split_dev_test(df, args.label_col, args.test_col, logger)
    if len(dev_df) < args.n_splits * 2:
        raise ValueError(f"Development set is too small after filtering (n={len(dev_df)}).")

    needs_pinn = preset_needs_pinn(feature_cols)
    pinn_bundle = None
    pinn_config = None
    if needs_pinn:
        logger.info("Preparing curve records for fold-wise Tofts-PINN refit.")
        pinn_config = pinn_config_from_args(args)
        pinn_bundle = prepare_curve_records(
            tcc_csv=args.tcc_csv, metadata_csv=args.metadata_csv, pid_col=args.pid_col,
            test_col=args.test_col, label_col=args.label_col, config=pinn_config, logger=logger,
        )
        save_curve_artifacts(bundle=pinn_bundle, output_dir=args.output_dir)
        eligible = set(pinn_bundle["eligible_pids"])
        before_dev, before_test = len(dev_df), len(test_df)
        dev_df = dev_df.loc[dev_df[args.pid_col].astype(str).isin(eligible)].copy()
        test_df = test_df.loc[test_df[args.pid_col].astype(str).isin(eligible)].copy()
        logger.info("After PINN-eligibility filtering | development: %d -> %d | held-out test: %d -> %d",
                    before_dev, len(dev_df), before_test, len(test_df))

    xgb_kwargs = dict(
        n_estimators=args.n_estimators, max_depth=args.max_depth, learning_rate=args.learning_rate,
        subsample=args.subsample, colsample_bytree=args.colsample_bytree,
        colsample_bylevel=args.colsample_bylevel, colsample_bynode=args.colsample_bynode,
        min_child_weight=args.min_child_weight, reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha,
        gamma=args.gamma, max_delta_step=args.max_delta_step,
        scale_pos_weight=_parse_scale_pos_weight(args.scale_pos_weight),
        eval_metric=args.eval_metric, tree_method=args.tree_method, n_jobs=args.n_jobs,
    )

    cv_result = run_stratified_kfold_cv(
        dev_df=dev_df, feature_cols=feature_cols, pid_col=args.pid_col, label_col=args.label_col,
        n_splits=args.n_splits, random_state=args.random_state, xgb_kwargs=xgb_kwargs,
        early_stopping_rounds=(args.early_stopping_rounds or None), es_val_fraction=args.es_val_fraction,
        threshold_metric=args.threshold_metric, threshold_min=args.threshold_min,
        threshold_max=args.threshold_max, threshold_step=args.threshold_step,
        pinn_bundle=pinn_bundle, pinn_config=pinn_config,
        visual_cols=vis_cols, visual_reduction=args.visual_reduction,
        visual_n_components=args.visual_n_components, logger=logger,
    )

    oof_df = cv_result["oof_df"].merge(
        add_subtype_column(dev_df)[[args.pid_col, "clinical_hr", "clinical_her2", "clinical_triple_neg", "molecular_subtype"]],
        on=args.pid_col, how="left",
    )
    cv_result["fold_metrics_df"].to_csv(args.output_dir / "xgboost_cv_fold_metrics.csv", index=False)
    oof_df.to_csv(args.output_dir / "xgboost_cv_oof_predictions.csv", index=False)
    # Fold-wise (out-of-fold) TreeSHAP and raw model inputs: used to ground the MLLM on development cases.
    cv_result["oof_shap_df"].to_csv(args.output_dir / "xgboost_cv_oof_shap.csv", index=False)
    cv_result["oof_raw_inputs_df"].to_csv(args.output_dir / "xgboost_cv_oof_inputs.csv", index=False)
    if not cv_result["oof_pinn_features_df"].empty:
        cv_result["oof_pinn_features_df"].to_csv(args.output_dir / "tofts_pinn_oof_features.csv", index=False)
    with open(args.output_dir / "xgboost_cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_result["cv_summary"], f, indent=2, ensure_ascii=False, default=_json_default)
    msd = cv_result["cv_summary"]["fold_metrics_mean_sd"]
    logger.info(
        "CV finished | fold AUROC=%.4f +/- %.4f | fold AUPRC=%.4f +/- %.4f | pooled OOF AUROC=%.4f | threshold=%.2f | final_n_estimators=%d",
        msd["roc_auc"]["mean"], msd["roc_auc"]["sd"], msd["average_precision"]["mean"], msd["average_precision"]["sd"],
        float(cv_result["cv_summary"]["oof_metrics"].get("roc_auc", float("nan"))),
        float(cv_result["decision_threshold_used"]), int(cv_result["final_n_estimators_from_cv"]),
    )

    test_metrics: dict[str, Any] | None = None
    test_ci: dict[str, Any] | None = None
    test_pred_df = pd.DataFrame()
    final_dev_df, final_test_df = dev_df, test_df
    final_pinn = None

    if needs_pinn and args.run_final_test:
        logger.info("Final-stage Tofts-PINN refit on full development split.")
        train_pids = dev_df[args.pid_col].astype(str).tolist()
        infer_pids = train_pids + test_df[args.pid_col].astype(str).tolist()
        final_pinn = train_pinn_and_infer_for_pid_sets(
            curve_records=pinn_bundle["curve_records"], train_pids=train_pids, infer_pids=infer_pids,
            time_grid_seconds=pinn_bundle["time_grid_seconds"], aif_cp=pinn_bundle["aif_cp"],
            random_state=int(args.random_state), config=pinn_config, logger=logger,
        )
        base_case_df = pd.concat([dev_df, test_df], ignore_index=True)
        # drop placeholder PINN columns so the merge does not create *_x / *_y duplicates
        base_case_df = base_case_df.drop(columns=[c for c in base_case_df.columns
                                                  if c in set(final_pinn["feature_wide_df"].columns) - {args.pid_col}])
        final_feature_df = build_final_tofts_feature_table(
            base_case_df=base_case_df, pinn_wide_df=final_pinn["feature_wide_df"],
            pid_col=args.pid_col, test_col=args.test_col, label_col=args.label_col,
        )
        final_feature_df.to_csv(args.output_dir / "tofts_pinn_features.csv", index=False)
        final_feature_df.loc[final_feature_df.get("split_group") == "heldout_test"].to_csv(
            args.output_dir / "pinn_test_features.csv", index=False)
        with open(args.output_dir / "tofts_pinn_training_summary.json", "w", encoding="utf-8") as f:
            json.dump({**final_pinn["fit_summary"], "pinn_feature_generation_mode": "final_full_development_refit_only",
                       "whether_test_seen_in_pinn_training": False, "development_pid_count": int(len(dev_df)),
                       "heldout_test_pid_count": int(len(test_df)), "random_state": int(args.random_state)},
                      f, indent=2, ensure_ascii=False, default=_json_default)

        wide = final_pinn["feature_wide_df"].copy()
        lookup = wide.set_index(wide["pid"].astype(str))
        final_dev_df, final_test_df = dev_df.copy(), test_df.copy()
        for col in list(PINN_COLUMNS) + ["core15_recon_mse", "safe_rim_recon_mse"]:
            if col in lookup.columns:
                final_dev_df[col] = final_dev_df[args.pid_col].astype(str).map(lookup[col])
                final_test_df[col] = final_test_df[args.pid_col].astype(str).map(lookup[col])
        final_dev_df = refresh_tofts_derived_columns(final_dev_df)
        final_test_df = refresh_tofts_derived_columns(final_test_df)

    if args.run_final_test:
        threshold = float(cv_result["decision_threshold_used"])
        final = fit_final_model_and_evaluate(
            dev_df=final_dev_df, test_df=final_test_df, feature_cols=feature_cols,
            pid_col=args.pid_col, label_col=args.label_col, decision_threshold=threshold,
            n_estimators=int(cv_result["final_n_estimators_from_cv"] or args.n_estimators),
            random_state=args.random_state,
            xgb_kwargs={k: v for k, v in xgb_kwargs.items() if k != "n_estimators"},
            visual_cols=vis_cols, visual_reduction=args.visual_reduction,
            visual_n_components=args.visual_n_components, logger=logger,
        )
        test_pred_df = final["test_pred_df"]
        test_metrics = final["test_metrics"]
        if not final["test_shap_df"].empty:
            final["test_shap_df"].to_csv(args.output_dir / "heldout_test_shap.csv", index=False)
            final["test_raw_inputs_df"].to_csv(args.output_dir / "heldout_test_inputs.csv", index=False)
        if not test_pred_df.empty:
            test_pred_df = test_pred_df.merge(
                add_subtype_column(final_test_df)[[args.pid_col, "clinical_hr", "clinical_her2", "clinical_triple_neg", "molecular_subtype"]],
                on=args.pid_col, how="left",
            )
            labeled = test_pred_df["y_true"].notna()
            if args.n_bootstrap and labeled.sum() > 1:
                test_ci = bootstrap_metric_ci(test_pred_df.loc[labeled, "y_true"], test_pred_df.loc[labeled, "y_prob"],
                                              threshold, n_bootstrap=args.n_bootstrap, random_state=args.random_state)

        # Final development feature matrix (model inputs) for SHAP / reporting.
        X_dev = final["X_dev"].copy()
        X_dev.insert(0, args.pid_col, final_dev_df[args.pid_col].astype(str).to_numpy())
        X_dev[args.label_col] = final_dev_df[args.label_col].astype(int).to_numpy()
        X_dev.to_csv(args.output_dir / "final_development_model_inputs.csv", index=False)

        if not args.no_save_final_model:
            pinn_payload = None
            if final_pinn is not None:
                pinn_payload = {"model": final_pinn["model"], "curve_scale": final_pinn["curve_scale"],
                                "time_grid_seconds": pinn_bundle["time_grid_seconds"],
                                "aif_cp": pinn_bundle["aif_cp"], "config": pinn_config}
            save_locked_model(
                args.output_dir / FINAL_MODEL_DIRNAME, feature_cols=feature_cols,
                preprocessor=final["preprocessor"], xgb_model=final["model"], decision_threshold=threshold,
                n_estimators=int(cv_result["final_n_estimators_from_cv"] or args.n_estimators),
                scale_pos_weight=final["scale_pos_weight"], pinn_payload=pinn_payload,
                extra={"feature_mode": args.feature_mode, "min_core_voxels": int(args.min_core_voxels),
                       "random_state": int(args.random_state), "xgb_hyperparameters": xgb_kwargs},
            )
            logger.info("Locked final model written to %s", args.output_dir / FINAL_MODEL_DIRNAME)

    predictions_frames = [oof_df.copy()]
    if not test_pred_df.empty:
        predictions_frames.append(test_pred_df)
    pd.concat(predictions_frames, ignore_index=True).to_csv(args.output_dir / "xgboost_predictions.csv", index=False)

    metrics_payload: dict[str, Any] = {
        "feature_mode": args.feature_mode,
        "feature_columns": feature_cols,
        "visual_reduction": args.visual_reduction,
        "n_splits": int(args.n_splits),
        "random_state": int(args.random_state),
        "decision_threshold_metric": args.threshold_metric,
        "decision_threshold_used": float(cv_result["decision_threshold_used"]),
        "excluded_min_core_voxels": excluded,
        "development_n": int(len(dev_df)),
        "heldout_test_n": int(len(test_df)),
        "cv_fold_metrics_mean_sd": cv_result["cv_summary"]["fold_metrics_mean_sd"],
        "cv_oof_metrics_pooled": cv_result["cv_summary"]["oof_metrics"],
        "cv_fold_metrics": cv_result["cv_summary"]["fold_metrics"],
        "final_n_estimators_from_cv": int(cv_result["final_n_estimators_from_cv"]),
        "heldout_test_metrics": test_metrics,
        "heldout_test_bootstrap_ci": test_ci,
        "pinn_used": bool(needs_pinn),
    }
    with open(args.output_dir / "xgboost_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False, default=_json_default)
    logger.info("Finished. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
