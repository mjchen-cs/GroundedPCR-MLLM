"""
Locked final model: persistence and inference on new cohorts.

After ``scripts/03_train_xgboost.py --run_final_test --save_final_model`` the
final development-only fit is written to ``<output_dir>/final_model/``:

    pinn.pt               Tofts-PINN weights + curve scale + time grid + AIF + config
    preprocessor.pkl      median imputer (+ optional visual scaler / PCA / PLS)
    xgboost.json          XGBoost booster
    model_card.json       feature columns, OOF threshold, n_estimators, etc.

``predict_cohort`` reproduces the full inference path for an independent cohort
(I-SPY 1, Duke, or a perturbed copy of the held-out test set): TCC ->
concentration curves on the *training* time grid -> frozen PINN -> Tofts
features -> frozen imputer -> frozen XGBoost -> probability + label at the
locked OOF threshold. Nothing is refitted on the evaluation cohort.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from feature_table import add_subtype_column, apply_core_voxel_exclusion, load_and_merge_features
from tofts_pinn_utils import (
    infer_tofts_pinn_features,
    load_pinn_checkpoint,
    prepare_curve_records,
    save_pinn_checkpoint,
)
from xgboost_utils import PINN_COLUMNS, preset_needs_pinn, refresh_tofts_derived_columns

FINAL_MODEL_DIRNAME = "final_model"


def save_locked_model(
    model_dir: Path,
    *,
    feature_cols: list[str],
    preprocessor: Any,
    xgb_model: XGBClassifier,
    decision_threshold: float,
    n_estimators: int,
    scale_pos_weight: float,
    pinn_payload: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if pinn_payload is not None:
        save_pinn_checkpoint(
            model_dir / "pinn.pt",
            model=pinn_payload["model"],
            curve_scale=float(pinn_payload["curve_scale"]),
            time_grid_seconds=pinn_payload["time_grid_seconds"],
            aif_cp=pinn_payload["aif_cp"],
            config=pinn_payload["config"],
        )
    with open(model_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    xgb_model.save_model(str(model_dir / "xgboost.json"))
    card = {
        "feature_columns": list(feature_cols),
        "model_input_columns": list(preprocessor.output_columns),
        "decision_threshold": float(decision_threshold),
        "decision_threshold_source": "F1-maximising threshold on pooled development OOF predictions",
        "n_estimators": int(n_estimators),
        "scale_pos_weight": float(scale_pos_weight),
        "imputation_medians_from_development": preprocessor.imputation_medians(),
        "visual_reduction": preprocessor.visual_reduction,
        "uses_pinn": pinn_payload is not None,
        **(extra or {}),
    }
    with open(model_dir / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, ensure_ascii=False)


def load_locked_model(model_dir: Path, device: str = "cpu") -> dict[str, Any]:
    model_dir = Path(model_dir)
    with open(model_dir / "model_card.json", encoding="utf-8") as f:
        card = json.load(f)
    with open(model_dir / "preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    clf = XGBClassifier()
    clf.load_model(str(model_dir / "xgboost.json"))
    pinn = load_pinn_checkpoint(model_dir / "pinn.pt", device=device) if (model_dir / "pinn.pt").exists() else None
    return {"card": card, "preprocessor": preprocessor, "xgb": clf, "pinn": pinn}


def infer_locked_pinn_features(
    locked: dict[str, Any],
    tcc: Path | pd.DataFrame,
    metadata_csv: Path | None,
    pid_col: str = "pid",
    label_col: str = "pCR",
    test_col: str = "test",
    config_overrides: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the frozen PINN on a cohort's TCCs aligned to the training time grid."""
    pinn = locked["pinn"]
    if pinn is None:
        raise ValueError("Locked model has no PINN checkpoint.")
    config = dict(pinn["config"])
    config.update(config_overrides or {})
    bundle = prepare_curve_records(
        tcc_csv=tcc,
        metadata_csv=metadata_csv,
        pid_col=pid_col,
        test_col=test_col,
        label_col=label_col,
        config=config,
        logger=logger,
        fixed_time_grid_seconds=pinn["time_grid_seconds"],
    )
    _, wide_df = infer_tofts_pinn_features(
        model=pinn["model"],
        records=bundle["curve_records"],
        time_grid_seconds=pinn["time_grid_seconds"],
        aif_cp=pinn["aif_cp"],
        curve_scale=float(pinn["curve_scale"]),
        device=pinn["device"],
    )
    return wide_df, bundle


def build_cohort_feature_table(
    locked: dict[str, Any],
    *,
    feature_csv: Path | pd.DataFrame,
    volume_csv: Path | pd.DataFrame,
    metadata_csv: Path,
    tcc: Path | pd.DataFrame,
    pid_col: str = "pid",
    label_col: str = "pCR",
    test_col: str = "test",
    visual_embedding_csv: Path | None = None,
    min_core_voxels: int = 10,
    config_overrides: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    df = load_and_merge_features(
        feature_csv, volume_csv, metadata_csv, pid_col=pid_col, label_col=label_col,
        test_col=test_col, visual_embedding_csv=visual_embedding_csv, logger=logger,
    )
    df, _ = apply_core_voxel_exclusion(df, min_core_voxels, pid_col=pid_col, logger=logger)
    if preset_needs_pinn(locked["card"]["feature_columns"]):
        wide_df, bundle = infer_locked_pinn_features(
            locked, tcc, metadata_csv, pid_col=pid_col, label_col=label_col, test_col=test_col,
            config_overrides=config_overrides, logger=logger,
        )
        eligible = set(bundle["eligible_pids"])
        df = df.loc[df[pid_col].astype(str).isin(eligible)].copy()
        lookup = wide_df.set_index(wide_df["pid"].astype(str))
        for col in list(PINN_COLUMNS) + ["core15_recon_mse", "safe_rim_recon_mse"]:
            if col in lookup.columns:
                df[col] = df[pid_col].astype(str).map(lookup[col])
        df = refresh_tofts_derived_columns(df)
    return add_subtype_column(df)


def predict_with_locked_model(
    locked: dict[str, Any],
    df: pd.DataFrame,
    pid_col: str = "pid",
    label_col: str = "pCR",
    split_group: str = "external",
) -> pd.DataFrame:
    X = locked["preprocessor"].transform(df)
    prob = locked["xgb"].predict_proba(X)[:, 1]
    thr = float(locked["card"]["decision_threshold"])
    out = pd.DataFrame({
        pid_col: df[pid_col].astype(str).to_numpy(),
        "y_true": pd.to_numeric(df[label_col], errors="coerce").to_numpy() if label_col in df.columns else np.nan,
        "y_prob": prob,
        "y_pred": (prob >= thr).astype(int),
        "split_group": split_group,
    })
    for col in ["clinical_hr", "clinical_her2", "clinical_triple_neg", "molecular_subtype"]:
        if col in df.columns:
            out[col] = df[col].to_numpy()
    return out


__all__ = [
    "FINAL_MODEL_DIRNAME",
    "save_locked_model",
    "load_locked_model",
    "infer_locked_pinn_features",
    "build_cohort_feature_table",
    "predict_with_locked_model",
]
