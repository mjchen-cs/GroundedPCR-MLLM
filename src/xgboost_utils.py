"""
XGBoost utilities for the post-segmentation pCR pipeline.

This module provides:

- ``MAIN_FEATURE_COLUMNS``: the 12-feature main preset described in the manuscript.
- ``FEATURE_PRESETS``: feature presets for the main model and the Table III ablations.
- ``TabularPreprocessor``: median imputation (and the optional visual-branch
  scaler + PCA/PLS reduction) that is *fitted on training data only* and then
  applied unchanged to validation / held-out / external data (Sec. II-E, II-G).
- ``build_xgb_classifier``: a thin factory wrapping ``xgboost.XGBClassifier``.
- ``run_stratified_kfold_cv``: leakage-safe 5-fold CV with fold-wise Tofts-PINN
  refits, fold-internal early stopping, and a single OOF decision threshold.
- ``fit_final_model_and_evaluate``: final refit on the full development split.

Manuscript correspondence (Sec. II-E / II-F):

- median imputation fitted on the training fold (CV) or on the full
  development set (final model) and *applied* to validation / test data;
- ``scale_pos_weight`` = n_negative / n_positive of the training data
  (up-weights the minority pCR class) unless an explicit value is given;
- shallow trees (``max_depth=2``), low learning rate, row / column
  subsampling, ``min_child_weight``, ``gamma``, L1 / L2 penalties and
  ``max_delta_step``;
- up to 1500 boosting rounds with early stopping inside CV. Early stopping is
  monitored on an *inner* stratified split of the fold's training data, so the
  fold's validation cases are never used for model selection. The final number
  of estimators is the mean of the fold-wise best iterations;
- one decision threshold chosen by maximising F1 on the pooled OOF
  probabilities; fold-wise threshold-dependent metrics are reported at that
  same threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from metrics_utils import compute_binary_metrics, select_best_threshold


# --- Feature presets ---------------------------------------------------------

TOFTS_FEATURE_COLUMNS: list[str] = [
    "core15_Ktrans",
    "core15_kep",
    "safe_rim_Ktrans",
    "safe_rim_kep",
]

VOLUME_FEATURE_COLUMNS: list[str] = [
    "tumor_volume_mm3",
    "core15_volume_mm3",
    "safe_rim_volume_mm3",
    "safe_rim_to_core15_volume_ratio",
]

CLINICAL_FEATURE_COLUMNS: list[str] = [
    "clinical_age",
    "clinical_hr",
    "clinical_her2",
    "clinical_triple_neg",
]

#: The 12-feature main preset reported in the manuscript.
MAIN_FEATURE_COLUMNS: list[str] = TOFTS_FEATURE_COLUMNS + VOLUME_FEATURE_COLUMNS + CLINICAL_FEATURE_COLUMNS

#: Regional contrasts of the Tofts proxies ("Deltas & Ratios", Table III).
TOFTS_DELTA_RATIO_COLUMNS: list[str] = [
    "safe_rim_minus_core15_Ktrans",
    "safe_rim_minus_core15_kep",
    "log_safe_rim_to_core15_Ktrans_ratio",
    "log_safe_rim_to_core15_kep_ratio",
]

#: Exploratory extensions (performance-improvement study).
from heterogeneity_features import HETEROGENEITY_COLUMNS, PERITUMORAL_COLUMNS  # noqa: E402

HET_COLUMNS: list[str] = [c for c in HETEROGENEITY_COLUMNS + PERITUMORAL_COLUMNS
                          if c not in ("het_ftv_mm3", "het_ftv_fraction")]
ARM_COLUMNS: list[str] = ["arm_" + a for a in ("ABT888", "AMG386", "Carboplatin", "Ganetespib", "Ganitumab",
                                                "MK2206", "Neratinib", "Pembrolizumab", "Pertuzumab", "TDM1",
                                                "Trastuzumab")]

#: 8 heuristic enhancement-curve features ("8 Simple Dynamics", Table III footnote 1).
SIMPLE_DYNAMICS_COLUMNS: list[str] = [
    f"{roi}_{name}"
    for roi in ("core15", "safe_rim")
    for name in ("time_to_peak", "wash_in_slope", "wash_out_slope", "auc_normalized_s_over_s0")
]

FEATURE_PRESETS: dict[str, list[str]] = {
    # Main model (Table II, Table III "Proposed PI-ML").
    "main": list(MAIN_FEATURE_COLUMNS),
    "tofts_volume_clinical": list(MAIN_FEATURE_COLUMNS),  # alias of "main"
    # Table III - isolated physiological proxies.
    "ktrans_only": ["core15_Ktrans", "safe_rim_Ktrans"],
    "kep_only": ["core15_kep", "safe_rim_kep"],
    "core15_physics_only": ["core15_Ktrans", "core15_kep"],
    "safe_rim_physics_only": ["safe_rim_Ktrans", "safe_rim_kep"],
    "tofts_only": list(TOFTS_FEATURE_COLUMNS),  # "All physics"
    # Table III - volume and combined feature sets.
    "volume_only": list(VOLUME_FEATURE_COLUMNS),
    "tofts_basic_volume": TOFTS_FEATURE_COLUMNS + ["tumor_volume_mm3"],
    "tofts_full_volume": TOFTS_FEATURE_COLUMNS + VOLUME_FEATURE_COLUMNS,
    "tofts_deltas_ratios": TOFTS_FEATURE_COLUMNS + TOFTS_DELTA_RATIO_COLUMNS,
    # Table III - multimodal comparisons.
    "main_plus_simple_dynamics": MAIN_FEATURE_COLUMNS + SIMPLE_DYNAMICS_COLUMNS,
    # The visual branch is added on top of "main" with --visual_embedding_csv.
    # Additional reference preset.
    "clinical_only": list(CLINICAL_FEATURE_COLUMNS),
    # Performance-improvement study (compare on development CV only).
    "clinical_mp": CLINICAL_FEATURE_COLUMNS + ["clinical_mp"],
    "clinical_mp_volume": CLINICAL_FEATURE_COLUMNS + ["clinical_mp"] + VOLUME_FEATURE_COLUMNS,
    "clinical_mp_volume_het": CLINICAL_FEATURE_COLUMNS + ["clinical_mp"] + VOLUME_FEATURE_COLUMNS + HET_COLUMNS,
    "main_mp": MAIN_FEATURE_COLUMNS + ["clinical_mp"],
    "main_het": MAIN_FEATURE_COLUMNS + HET_COLUMNS,
    "main_mp_het": MAIN_FEATURE_COLUMNS + ["clinical_mp"] + HET_COLUMNS,
    "main_mp_het_arm": MAIN_FEATURE_COLUMNS + ["clinical_mp"] + HET_COLUMNS + ARM_COLUMNS,
    "main_mp_arm": MAIN_FEATURE_COLUMNS + ["clinical_mp"] + ARM_COLUMNS,
}

#: Ordered list used by scripts/05_run_ablations.py to rebuild Table III.
TABLE_III_PRESETS: list[str] = [
    "ktrans_only",
    "kep_only",
    "core15_physics_only",
    "safe_rim_physics_only",
    "tofts_only",
    "volume_only",
    "tofts_basic_volume",
    "tofts_full_volume",
    "tofts_deltas_ratios",
    "main",
    "main_plus_simple_dynamics",
]

PINN_COLUMNS: list[str] = list(TOFTS_FEATURE_COLUMNS)


def feature_columns_for_preset(preset: str) -> list[str]:
    """Return the feature column list for the named preset."""
    if preset not in FEATURE_PRESETS:
        raise ValueError(
            f"Unknown feature preset {preset!r}. "
            f"Available: {sorted(FEATURE_PRESETS.keys())}"
        )
    return list(FEATURE_PRESETS[preset])


def preset_needs_pinn(feature_cols: Sequence[str]) -> bool:
    pinn_dependent = set(PINN_COLUMNS) | set(TOFTS_DELTA_RATIO_COLUMNS)
    return any(c in pinn_dependent for c in feature_cols)


def refresh_tofts_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute derived Tofts contrasts after PINN columns were (re)assigned."""
    from tofts_pinn_utils import add_tofts_derived_feature_columns

    return add_tofts_derived_feature_columns(df)


# --- Preprocessing (fitted on training data only) ----------------------------


@dataclass
class TabularPreprocessor:
    """
    Median imputer for the tabular features plus an optional visual-branch
    reducer (StandardScaler + PCA or PLS). ``fit`` sees training data only;
    ``transform`` applies the frozen statistics to any other split.
    """

    feature_cols: list[str]
    visual_cols: list[str] = field(default_factory=list)
    visual_reduction: str | None = None  # None / "pca" / "pls"
    visual_n_components: int = 8
    random_state: int = 42

    imputer_: SimpleImputer | None = None
    visual_imputer_: SimpleImputer | None = None
    visual_scaler_: StandardScaler | None = None
    visual_reducer_: Any = None

    @property
    def output_columns(self) -> list[str]:
        cols = list(self.feature_cols)
        if self.visual_reduction:
            prefix = f"visual_{self.visual_reduction}"
            cols += [f"{prefix}_{i + 1}" for i in range(int(self.visual_n_components))]
        return cols

    def _base_matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Feature columns not found in dataframe: {missing}")
        return df[self.feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    def _visual_matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.visual_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Visual embedding columns missing (first few): {missing[:5]}")
        return df[self.visual_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame, y: np.ndarray | None = None) -> "TabularPreprocessor":
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.imputer_.fit(self._base_matrix(df))
        if self.visual_reduction:
            if not self.visual_cols:
                raise ValueError("visual_reduction set but no visual_cols supplied.")
            V = self._visual_matrix(df)
            self.visual_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            V = self.visual_imputer_.fit_transform(V)
            self.visual_scaler_ = StandardScaler()
            V = self.visual_scaler_.fit_transform(V)
            n_comp = int(min(self.visual_n_components, V.shape[0] - 1, V.shape[1]))
            if n_comp < int(self.visual_n_components):
                raise ValueError(f"Too few samples/features for {self.visual_n_components} visual components.")
            if self.visual_reduction == "pca":
                self.visual_reducer_ = PCA(n_components=n_comp, random_state=int(self.random_state)).fit(V)
            elif self.visual_reduction == "pls":
                if y is None:
                    raise ValueError("PLS reduction requires labels at fit time.")
                self.visual_reducer_ = PLSRegression(n_components=n_comp, scale=False).fit(V, np.asarray(y, dtype=float))
            else:
                raise ValueError(f"Unknown visual_reduction {self.visual_reduction!r}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.imputer_ is None:
            raise RuntimeError("TabularPreprocessor must be fitted before transform().")
        X = self.imputer_.transform(self._base_matrix(df))
        if self.visual_reduction:
            V = self.visual_imputer_.transform(self._visual_matrix(df))
            V = self.visual_scaler_.transform(V)
            Z = self.visual_reducer_.transform(V)
            X = np.concatenate([X, np.asarray(Z, dtype=float)], axis=1)
        return pd.DataFrame(X, columns=self.output_columns, index=df.index)

    def fit_transform(self, df: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def imputation_medians(self) -> dict[str, float]:
        if self.imputer_ is None:
            return {}
        return {c: float(v) for c, v in zip(self.feature_cols, self.imputer_.statistics_)}


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    preprocessor: TabularPreprocessor | None = None,
) -> tuple[pd.DataFrame, TabularPreprocessor]:
    """
    Backwards-compatible helper. Without ``preprocessor`` the median imputer is
    fitted on ``df`` (use this only for *training* data); with a fitted
    ``preprocessor`` the stored medians are applied to ``df``.
    """
    if preprocessor is None:
        preprocessor = TabularPreprocessor(feature_cols=list(feature_cols)).fit(df)
    return preprocessor.transform(df), preprocessor


# --- Classifier factory ------------------------------------------------------


def compute_scale_pos_weight(y: np.ndarray) -> float:
    """n_negative / n_positive, i.e. up-weight the minority pCR class."""
    y = np.asarray(y).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(n_neg) / float(max(n_pos, 1))


def build_xgb_classifier(
    n_estimators: int = 1500,
    max_depth: int = 2,
    learning_rate: float = 0.02,
    subsample: float = 0.85,
    colsample_bytree: float = 0.75,
    colsample_bylevel: float = 0.90,
    colsample_bynode: float = 0.90,
    min_child_weight: float = 2.0,
    reg_lambda: float = 6.0,
    reg_alpha: float = 0.5,
    gamma: float = 0.1,
    max_delta_step: float = 1.0,
    scale_pos_weight: float = 1.0,
    eval_metric: str = "logloss",
    tree_method: str = "hist",
    n_jobs: int = 1,
    random_state: int = 42,
    early_stopping_rounds: int | None = None,
) -> XGBClassifier:
    """Construct an ``XGBClassifier`` with the project's default hyperparameters."""
    return XGBClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        colsample_bylevel=float(colsample_bylevel),
        colsample_bynode=float(colsample_bynode),
        min_child_weight=float(min_child_weight),
        reg_lambda=float(reg_lambda),
        reg_alpha=float(reg_alpha),
        gamma=float(gamma),
        max_delta_step=float(max_delta_step),
        scale_pos_weight=float(scale_pos_weight),
        eval_metric=str(eval_metric),
        tree_method=str(tree_method),
        n_jobs=int(n_jobs),
        random_state=int(random_state),
        early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
        objective="binary:logistic",
        verbosity=0,
    )


def _resolve_xgb_kwargs(xgb_kwargs: dict[str, Any], y_train: np.ndarray) -> tuple[dict[str, Any], float]:
    kwargs = dict(xgb_kwargs)
    spw = kwargs.get("scale_pos_weight")
    if spw is None or (isinstance(spw, str) and spw.lower() == "auto"):
        spw = compute_scale_pos_weight(y_train)
    kwargs["scale_pos_weight"] = float(spw)
    return kwargs, float(spw)


def fit_xgb_with_inner_early_stopping(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    xgb_kwargs: dict[str, Any],
    random_state: int,
    early_stopping_rounds: int | None,
    es_val_fraction: float,
) -> tuple[XGBClassifier, int, float]:
    """
    Fit XGBoost on ``X_train``. If early stopping is enabled, a stratified
    inner split of the *training* data is held out as the monitoring set;
    the outer validation fold is never touched. Returns (model, best_n, spw).
    """
    kwargs, spw = _resolve_xgb_kwargs(xgb_kwargs, y_train)
    if early_stopping_rounds and es_val_fraction > 0 and len(np.unique(y_train)) == 2:
        X_in, X_es, y_in, y_es = train_test_split(
            X_train,
            y_train,
            test_size=float(es_val_fraction),
            stratify=y_train,
            random_state=int(random_state),
        )
        kwargs, spw = _resolve_xgb_kwargs(xgb_kwargs, y_in)
        clf = build_xgb_classifier(random_state=int(random_state), early_stopping_rounds=int(early_stopping_rounds), **kwargs)
        clf.fit(X_in, y_in, eval_set=[(X_es, y_es)], verbose=False)
        best_iteration = getattr(clf, "best_iteration", None)
        best_n = int(best_iteration) + 1 if best_iteration is not None else int(clf.n_estimators)
        return clf, best_n, spw

    clf = build_xgb_classifier(random_state=int(random_state), early_stopping_rounds=None, **kwargs)
    clf.fit(X_train, y_train, verbose=False)
    return clf, int(clf.n_estimators), spw


# --- Stratified K-fold CV ----------------------------------------------------


def tree_shap_frame(clf: XGBClassifier, X: pd.DataFrame, pids: Sequence[str]) -> pd.DataFrame:
    """Exact TreeSHAP contributions (log-odds) per case, plus the base value."""
    import xgboost as _xgb

    contrib = clf.get_booster().predict(_xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
    out = pd.DataFrame(contrib[:, :-1], columns=list(X.columns))
    out.insert(0, "pid", [str(p) for p in pids])
    out["shap_base_value"] = contrib[:, -1]
    return out


def _assign_pinn_columns(df: pd.DataFrame, wide_df: pd.DataFrame, pid_col: str) -> pd.DataFrame:
    out = df.copy()
    lookup = wide_df.set_index(wide_df["pid"].astype(str))
    for col in PINN_COLUMNS:
        if col in lookup.columns:
            out[col] = out[pid_col].astype(str).map(lookup[col])
    return refresh_tofts_derived_columns(out)


def _summarise_folds(fold_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    keys = ["roc_auc", "average_precision", "f1", "recall", "specificity", "accuracy", "balanced_accuracy", "precision"]
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.asarray([r.get(k, np.nan) for r in fold_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "sd": float(vals.std(ddof=1)) if vals.size > 1 else float("nan"),
        }
    return out


def run_stratified_kfold_cv(
    dev_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    pid_col: str = "pid",
    label_col: str = "pCR",
    n_splits: int = 5,
    random_state: int = 42,
    xgb_kwargs: dict[str, Any] | None = None,
    early_stopping_rounds: int | None = 100,
    es_val_fraction: float = 0.15,
    threshold_metric: str = "f1",
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
    pinn_bundle: dict[str, Any] | None = None,
    pinn_config: dict[str, Any] | None = None,
    visual_cols: Sequence[str] | None = None,
    visual_reduction: str | None = None,
    visual_n_components: int = 8,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Stratified K-fold CV with fold-wise Tofts-PINN regeneration, fold-fitted
    preprocessing, inner early stopping, and a single OOF threshold.
    """
    feature_cols = list(feature_cols)
    xgb_kwargs = dict(xgb_kwargs or {})
    xgb_kwargs.pop("early_stopping_rounds", None)

    if pid_col not in dev_df.columns:
        raise KeyError(f"dev_df is missing required column {pid_col!r}")
    if label_col not in dev_df.columns:
        raise KeyError(f"dev_df is missing required column {label_col!r}")

    dev_df = dev_df.copy().reset_index(drop=True)
    dev_df[label_col] = pd.to_numeric(dev_df[label_col], errors="coerce")
    if dev_df[label_col].isna().any():
        raise ValueError("Development split contains NaN labels; clean before CV.")
    y_dev = dev_df[label_col].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))

    fold_n_estimators: list[int] = []
    fold_spw: list[float] = []
    fold_records: list[dict[str, Any]] = []
    oof_records: list[dict[str, Any]] = []
    pinn_fold_summaries: list[dict[str, Any]] = []
    fold_pinn_features: list[pd.DataFrame] = []
    oof_shap_parts: list[pd.DataFrame] = []
    oof_input_parts: list[pd.DataFrame] = []

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dev_df)), y_dev), start=1):
        fold_train_df = dev_df.iloc[train_idx].copy()
        fold_val_df = dev_df.iloc[val_idx].copy()

        if pinn_bundle is not None:
            if pinn_config is None:
                raise ValueError("pinn_config must be supplied when pinn_bundle is used.")
            from tofts_pinn_utils import train_pinn_and_infer_for_pid_sets

            train_pids = fold_train_df[pid_col].astype(str).tolist()
            val_pids = fold_val_df[pid_col].astype(str).tolist()
            if logger is not None:
                logger.info(
                    "Fold %d/%d PINN fit | train_pids=%d val_pids=%d | heldout_test_seen_in_pinn_training=false",
                    fold_index, n_splits, len(set(train_pids)), len(set(val_pids)),
                )
            pinn_result = train_pinn_and_infer_for_pid_sets(
                curve_records=pinn_bundle["curve_records"],
                train_pids=train_pids,
                infer_pids=train_pids + val_pids,
                time_grid_seconds=pinn_bundle["time_grid_seconds"],
                aif_cp=pinn_bundle["aif_cp"],
                random_state=int(random_state) + int(fold_index),
                config=pinn_config,
                logger=logger,
            )
            wide_df = pinn_result["feature_wide_df"]
            fold_train_df = _assign_pinn_columns(fold_train_df, wide_df, pid_col)
            fold_val_df = _assign_pinn_columns(fold_val_df, wide_df, pid_col)
            val_feat = wide_df.loc[wide_df["pid"].astype(str).isin(set(val_pids))].copy()
            val_feat["fold_index"] = int(fold_index)
            fold_pinn_features.append(val_feat)
            pinn_fold_summaries.append(
                {
                    "fold_index": int(fold_index),
                    "train_pid_count": int(len(set(train_pids))),
                    "val_pid_count": int(len(set(val_pids))),
                    "whether_test_seen_in_pinn_training": False,
                    **pinn_result["fit_summary"],
                }
            )

        y_train = fold_train_df[label_col].astype(int).to_numpy()
        y_val = fold_val_df[label_col].astype(int).to_numpy()

        # Imputer (and visual reducer) fitted on the training fold only.
        pre = TabularPreprocessor(
            feature_cols=feature_cols,
            visual_cols=list(visual_cols or []),
            visual_reduction=visual_reduction,
            visual_n_components=int(visual_n_components),
            random_state=int(random_state),
        ).fit(fold_train_df, y_train)
        X_train = pre.transform(fold_train_df)
        X_val = pre.transform(fold_val_df)

        clf, best_n, spw = fit_xgb_with_inner_early_stopping(
            X_train,
            y_train,
            xgb_kwargs=xgb_kwargs,
            random_state=int(random_state) + int(fold_index),
            early_stopping_rounds=early_stopping_rounds,
            es_val_fraction=float(es_val_fraction),
        )
        fold_n_estimators.append(int(best_n))
        fold_spw.append(float(spw))

        y_val_prob = clf.predict_proba(X_val)[:, 1]
        fold_shap = tree_shap_frame(clf, X_val, fold_val_df[pid_col].astype(str).tolist())
        fold_shap.insert(1, "fold_index", int(fold_index))
        fold_inputs = fold_val_df[[pid_col] + [c for c in feature_cols if c in fold_val_df.columns]].copy()
        fold_inputs.insert(1, "fold_index", int(fold_index))
        oof_shap_parts.append(fold_shap)
        oof_input_parts.append(fold_inputs)
        fold_records.append({"fold_index": int(fold_index), "y_true": y_val, "y_prob": y_val_prob,
                             "best_n_estimators": int(best_n), "scale_pos_weight": float(spw)})

        for pid_value, true_label, prob in zip(fold_val_df[pid_col].astype(str).tolist(), y_val.tolist(), y_val_prob.tolist()):
            oof_records.append(
                {pid_col: pid_value, "fold_index": int(fold_index), "y_true": int(true_label),
                 "y_prob": float(prob), "split_group": "development_oof"}
            )

    oof_df = pd.DataFrame(oof_records)
    oof_threshold = select_best_threshold(
        oof_df["y_true"].to_numpy(), oof_df["y_prob"].to_numpy(), metric=threshold_metric,
        threshold_min=threshold_min, threshold_max=threshold_max, threshold_step=threshold_step,
    )
    decision_threshold = float(oof_threshold["decision_threshold"])
    oof_df["y_pred"] = (oof_df["y_prob"].to_numpy() >= decision_threshold).astype(int)

    # Fold-wise metrics at the single OOF-derived threshold (Table II, development column).
    fold_metrics_rows: list[dict[str, Any]] = []
    for rec in fold_records:
        m = compute_binary_metrics(rec["y_true"], rec["y_prob"], decision_threshold=decision_threshold)
        fold_metrics_rows.append({"fold_index": rec["fold_index"], "best_n_estimators": rec["best_n_estimators"],
                                  "scale_pos_weight": rec["scale_pos_weight"], **m})

    final_n_estimators = int(round(float(np.mean(fold_n_estimators)))) if fold_n_estimators else 0

    cv_summary = {
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "feature_columns": list(feature_cols),
        "visual_reduction": visual_reduction,
        "visual_n_components": int(visual_n_components) if visual_reduction else 0,
        "fold_metrics": fold_metrics_rows,
        "fold_metrics_mean_sd": _summarise_folds(fold_metrics_rows),
        "oof_metrics": compute_binary_metrics(oof_df["y_true"], oof_df["y_prob"], decision_threshold),
        "decision_threshold_from_oof": decision_threshold,
        "threshold_metric": threshold_metric,
        "early_stopping_rounds": int(early_stopping_rounds) if early_stopping_rounds else None,
        "early_stopping_monitor": "inner stratified split of the fold training data" if early_stopping_rounds else None,
        "final_n_estimators_from_cv": final_n_estimators,
        "fold_best_n_estimators": [int(v) for v in fold_n_estimators],
        "fold_scale_pos_weight": [float(v) for v in fold_spw],
        "pinn_fold_summaries": pinn_fold_summaries if pinn_fold_summaries else None,
    }

    return {
        "cv_summary": cv_summary,
        "oof_df": oof_df,
        "decision_threshold_used": decision_threshold,
        "final_n_estimators_from_cv": final_n_estimators,
        "fold_metrics_df": pd.DataFrame(fold_metrics_rows),
        "oof_pinn_features_df": pd.concat(fold_pinn_features, ignore_index=True) if fold_pinn_features else pd.DataFrame(),
        "oof_shap_df": pd.concat(oof_shap_parts, ignore_index=True),
        "oof_raw_inputs_df": pd.concat(oof_input_parts, ignore_index=True),
    }


# --- Final development-fit + held-out test evaluation ------------------------


def fit_final_model_and_evaluate(
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    feature_cols: Sequence[str],
    *,
    pid_col: str = "pid",
    label_col: str = "pCR",
    decision_threshold: float = 0.5,
    n_estimators: int = 1500,
    random_state: int = 42,
    xgb_kwargs: dict[str, Any] | None = None,
    visual_cols: Sequence[str] | None = None,
    visual_reduction: str | None = None,
    visual_n_components: int = 8,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Refit preprocessing + XGBoost on the entire development split (no early
    stopping; ``n_estimators`` comes from the CV fold-wise best iterations)
    and optionally score the held-out test split with the frozen pipeline.
    """
    feature_cols = list(feature_cols)
    xgb_kwargs = dict(xgb_kwargs or {})
    xgb_kwargs["n_estimators"] = int(n_estimators)
    xgb_kwargs.pop("early_stopping_rounds", None)

    y_dev = dev_df[label_col].astype(int).to_numpy()
    pre = TabularPreprocessor(
        feature_cols=feature_cols,
        visual_cols=list(visual_cols or []),
        visual_reduction=visual_reduction,
        visual_n_components=int(visual_n_components),
        random_state=int(random_state),
    ).fit(dev_df, y_dev)
    X_dev = pre.transform(dev_df)
    kwargs, spw = _resolve_xgb_kwargs(xgb_kwargs, y_dev)
    clf = build_xgb_classifier(random_state=int(random_state), early_stopping_rounds=None, **kwargs)
    clf.fit(X_dev, y_dev, verbose=False)

    test_metrics: dict[str, Any] | None = None
    test_shap_df = pd.DataFrame()
    test_raw_inputs_df = pd.DataFrame()
    if test_df is None or test_df.empty:
        test_pred_df = pd.DataFrame(columns=[pid_col, "y_true", "y_prob", "y_pred", "split_group"])
    else:
        X_test = pre.transform(test_df)
        prob = clf.predict_proba(X_test)[:, 1]
        test_shap_df = tree_shap_frame(clf, X_test, test_df[pid_col].astype(str).tolist())
        test_raw_inputs_df = test_df[[pid_col] + [c for c in feature_cols if c in test_df.columns]].copy()
        test_pred_df = pd.DataFrame(
            {
                pid_col: test_df[pid_col].astype(str).to_numpy(),
                "y_true": pd.to_numeric(test_df[label_col], errors="coerce").to_numpy() if label_col in test_df.columns else np.nan,
                "y_prob": prob,
                "y_pred": (prob >= float(decision_threshold)).astype(int),
                "split_group": "heldout_test",
            }
        )
        labeled_mask = pd.notna(test_pred_df["y_true"])
        if labeled_mask.any():
            test_metrics = compute_binary_metrics(
                test_pred_df.loc[labeled_mask, "y_true"].to_numpy(),
                test_pred_df.loc[labeled_mask, "y_prob"].to_numpy(),
                decision_threshold=float(decision_threshold),
            )

    if logger is not None and test_metrics is not None:
        logger.info(
            "Held-out test | n=%d AUROC=%.4f AUPRC=%.4f F1=%.4f",
            test_metrics.get("n_samples", 0),
            float(test_metrics.get("roc_auc", float("nan"))),
            float(test_metrics.get("average_precision", float("nan"))),
            float(test_metrics.get("f1", float("nan"))),
        )

    return {
        "model": clf,
        "preprocessor": pre,
        "scale_pos_weight": float(spw),
        "X_dev": X_dev,
        "test_shap_df": test_shap_df,
        "test_raw_inputs_df": test_raw_inputs_df,
        "test_pred_df": test_pred_df,
        "test_metrics": test_metrics,
    }


__all__ = [
    "MAIN_FEATURE_COLUMNS",
    "TOFTS_FEATURE_COLUMNS",
    "VOLUME_FEATURE_COLUMNS",
    "CLINICAL_FEATURE_COLUMNS",
    "TOFTS_DELTA_RATIO_COLUMNS",
    "SIMPLE_DYNAMICS_COLUMNS",
    "FEATURE_PRESETS",
    "TABLE_III_PRESETS",
    "TabularPreprocessor",
    "feature_columns_for_preset",
    "preset_needs_pinn",
    "prepare_feature_matrix",
    "compute_scale_pos_weight",
    "build_xgb_classifier",
    "fit_xgb_with_inner_early_stopping",
    "run_stratified_kfold_cv",
    "fit_final_model_and_evaluate",
]
