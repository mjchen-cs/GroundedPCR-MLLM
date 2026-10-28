"""
Case-level feature table construction shared by the training, external
validation and sensitivity-analysis scripts.

- merges step-1 dynamics + volume tables with clinical covariates parsed from
  ``metadata.csv`` (label / split columns are taken from metadata only, so the
  step-1 copies of ``pCR`` / ``test`` never collide during the merge);
- applies the cohort exclusion rule of Sec. II-A (fewer than
  ``min_core_voxels`` voxels in the core ROI -> excluded);
- optionally merges a visual-embedding table (exploratory branch, Sec. II-G).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CLINICAL_COLUMNS = ["clinical_age", "clinical_hr", "clinical_her2", "clinical_triple_neg"]
VOLUME_COLUMNS = [
    "tumor_volume_mm3",
    "core15_volume_mm3",
    "safe_rim_volume_mm3",
    "safe_rim_to_core15_volume_ratio",
    "core15_voxel_count",
    "safe_rim_voxel_count",
    "tumor_voxel_count",
]


def coerce_pid(df: pd.DataFrame, pid_col: str) -> pd.DataFrame:
    if pid_col not in df.columns:
        for cand in ["pid", "PatientID", "patient_id", "case_id"]:
            if cand in df.columns and cand != pid_col:
                df = df.rename(columns={cand: pid_col})
                break
    if pid_col not in df.columns:
        raise KeyError(f"required column {pid_col!r} not found in dataframe.")
    df = df.copy()
    df[pid_col] = df[pid_col].astype(str).str.strip()
    return df


def coerce_binary_clinical(series: pd.Series) -> pd.Series:
    """Normalize clinical receptor fields to {0, 1, NaN}."""
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    text = s.astype(str).str.strip().str.lower()
    pos_tokens = {"1", "1.0", "+", "pos", "positive", "yes", "y", "true", "t"}
    neg_tokens = {"0", "0.0", "-", "neg", "negative", "no", "n", "false", "f"}
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[text.isin(pos_tokens)] = 1.0
    out[text.isin(neg_tokens)] = 0.0
    return out


def extract_clinical_columns(metadata_df: pd.DataFrame, pid_col: str = "pid") -> pd.DataFrame:
    """Best-effort extraction of the four clinical covariates (NaN if absent)."""
    out = metadata_df[[pid_col]].copy()
    lower_to_original = {str(c).lower(): str(c) for c in metadata_df.columns}

    def find(*candidates: str) -> str | None:
        for cand in candidates:
            if cand.lower() in lower_to_original:
                return lower_to_original[cand.lower()]
        return None

    age_col = find("clinical_age", "age", "age_at_baseline", "baseline_age")
    out["clinical_age"] = pd.to_numeric(metadata_df[age_col], errors="coerce") if age_col else np.nan
    hr_col = find("clinical_hr", "hr", "hr_status", "hr_positive", "hormone_receptor")
    out["clinical_hr"] = coerce_binary_clinical(metadata_df[hr_col]) if hr_col else np.nan
    her2_col = find("clinical_her2", "her2", "her2_status", "her2_positive")
    out["clinical_her2"] = coerce_binary_clinical(metadata_df[her2_col]) if her2_col else np.nan
    mp_col = find("clinical_mp", "mp", "mammaprint")
    out["clinical_mp"] = coerce_binary_clinical(metadata_df[mp_col]) if mp_col else np.nan
    for c in metadata_df.columns:
        if str(c).startswith("arm_"):
            out[str(c)] = pd.to_numeric(metadata_df[c], errors="coerce")

    tn_col = find("clinical_triple_neg", "triple_neg", "tripleneg", "triple_negative", "tnbc")
    if tn_col:
        out["clinical_triple_neg"] = coerce_binary_clinical(metadata_df[tn_col])
    else:
        hr = pd.to_numeric(out["clinical_hr"], errors="coerce")
        her2 = pd.to_numeric(out["clinical_her2"], errors="coerce")
        out["clinical_triple_neg"] = np.where(hr.isna() | her2.isna(), np.nan, ((hr == 0) & (her2 == 0)).astype(float))
    return out


def load_and_merge_features(
    feature_csv: Path | pd.DataFrame,
    volume_csv: Path | pd.DataFrame,
    metadata_csv: Path | pd.DataFrame,
    pid_col: str = "pid",
    label_col: str = "pCR",
    test_col: str = "test",
    visual_embedding_csv: Path | None = None,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    def _read(x: Path | pd.DataFrame, name: str) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            return x.copy()
        if logger is not None:
            logger.info("Loading %s: %s", name, x)
        return pd.read_csv(x)

    feat = coerce_pid(_read(feature_csv, "dynamics features"), pid_col)
    vol = coerce_pid(_read(volume_csv, "volume features"), pid_col)
    meta = coerce_pid(_read(metadata_csv, "metadata"), pid_col)
    meta = meta.drop_duplicates(subset=[pid_col], keep="first")

    clinical_df = extract_clinical_columns(meta, pid_col=pid_col)
    clinical_df[label_col] = pd.to_numeric(meta[label_col], errors="coerce").to_numpy() if label_col in meta.columns else np.nan
    clinical_df[test_col] = pd.to_numeric(meta[test_col], errors="coerce").to_numpy() if test_col in meta.columns else np.nan

    vol_cols = [pid_col] + [c for c in VOLUME_COLUMNS if c in vol.columns]
    feat = feat.drop(columns=[c for c in vol_cols[1:] if c in feat.columns])
    merged = feat.merge(vol[vol_cols].drop_duplicates(subset=[pid_col]), on=pid_col, how="left")

    # Labels, split flags and clinical covariates come from metadata only.
    merged = merged.drop(columns=[c for c in clinical_df.columns if c != pid_col and c in merged.columns])
    # metadata.csv defines the cohort: cases absent from it are not scored.
    merged = merged.merge(clinical_df, on=pid_col, how="inner")

    if visual_embedding_csv is not None:
        vis = coerce_pid(_read(visual_embedding_csv, "visual embeddings"), pid_col)
        vis = vis.drop(columns=[c for c in vis.columns if c != pid_col and c in merged.columns])
        merged = merged.merge(vis.drop_duplicates(subset=[pid_col]), on=pid_col, how="left")
    return merged


def visual_columns(df: pd.DataFrame, prefix: str = "vis_") -> list[str]:
    return [c for c in df.columns if str(c).startswith(prefix)]


def apply_core_voxel_exclusion(
    df: pd.DataFrame,
    min_core_voxels: int,
    pid_col: str = "pid",
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Exclude cases whose core ROI contains fewer than ``min_core_voxels`` voxels (Sec. II-A)."""
    if min_core_voxels is None or int(min_core_voxels) <= 0 or "core15_voxel_count" not in df.columns:
        return df, []
    counts = pd.to_numeric(df["core15_voxel_count"], errors="coerce")
    drop_mask = counts.isna() | (counts < int(min_core_voxels))
    dropped = df.loc[drop_mask, pid_col].astype(str).tolist()
    if logger is not None and dropped:
        logger.info("Excluded %d case(s) with < %d core voxels.", len(dropped), int(min_core_voxels))
    return df.loc[~drop_mask].copy(), dropped


def add_subtype_column(df: pd.DataFrame) -> pd.DataFrame:
    from metrics_utils import assign_molecular_subtype

    out = df.copy()
    out["molecular_subtype"] = assign_molecular_subtype(out.get("clinical_hr"), out.get("clinical_her2"))
    return out


__all__ = [
    "CLINICAL_COLUMNS",
    "coerce_pid",
    "extract_clinical_columns",
    "load_and_merge_features",
    "visual_columns",
    "apply_core_voxel_exclusion",
    "add_subtype_column",
]
