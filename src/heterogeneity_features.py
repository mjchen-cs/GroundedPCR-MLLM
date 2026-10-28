"""
Voxel-level enhancement heterogeneity, SER / functional tumour volume, and
peritumoral features from the pre / early / late DCE phases.

All features are computed per case from three phases (S0 = pre-contrast,
S1 = early post-contrast, S2 = late post-contrast):

- percent enhancement PE = 100 * (S1 - S0) / S0 per tumour voxel, summarised
  by mean, SD, coefficient of variation, percentiles (10/25/50/75/90),
  skewness, kurtosis and histogram entropy;
- signal enhancement ratio SER = (S1 - S0) / (S2 - S0) in enhancing voxels
  (PE >= ``pe_threshold``, default 70 % as in I-SPY functional tumour volume);
- functional tumour volume FTV (mm^3) and its fraction of the tumour volume,
  plus the fractions of enhancing voxels with washout (SER > 1.1), plateau
  (0.9-1.1) and persistent (SER < 0.9) kinetics;
- core15 vs safe-rim PE contrast (median difference);
- peritumoral shells 0-``ring1_mm`` and ``ring1_mm``-``ring2_mm`` outside the
  tumour (spacing-aware distance), restricted to voxels with non-trivial
  pre-contrast signal: mean / median PE and SER, and their ratio to the tumour.

These are hand-crafted, case-level descriptors; they do not use labels and are
therefore leakage-free by construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

FEATURE_PREFIX = "het_"


def _safe(v: float) -> float:
    try:
        v = float(v)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _entropy(values: np.ndarray, bins: int = 32, value_range: tuple[float, float] = (-50.0, 400.0)) -> float:
    if values.size < 2:
        return float("nan")
    hist, _ = np.histogram(np.clip(values, *value_range), bins=bins, range=value_range)
    p = hist[hist > 0].astype(float)
    p /= p.sum()
    return float(-(p * np.log2(p)).sum())


def _pe(s0: np.ndarray, s1: np.ndarray, floor: float) -> np.ndarray:
    return 100.0 * (s1 - s0) / np.maximum(s0, floor)


def _ser(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    denom = s2 - s0
    with np.errstate(divide="ignore", invalid="ignore"):
        ser = (s1 - s0) / denom
    ser[~np.isfinite(ser) | (np.abs(denom) < 1e-6)] = np.nan
    return np.clip(ser, 0.0, 3.0)


def compute_heterogeneity_features(
    pre: np.ndarray,
    early: np.ndarray,
    late: np.ndarray,
    tumor_mask: np.ndarray,
    core_mask: np.ndarray,
    rim_mask: np.ndarray,
    spacing: tuple[float, float, float],
    pe_threshold: float = 70.0,
    ring1_mm: float = 3.0,
    ring2_mm: float = 6.0,
) -> dict[str, Any]:
    p = FEATURE_PREFIX
    out: dict[str, Any] = {}
    tumor = np.asarray(tumor_mask, dtype=bool)
    if tumor.sum() < 5:
        return out
    s0 = np.asarray(pre, dtype=np.float32)
    s1 = np.asarray(early, dtype=np.float32)
    s2 = np.asarray(late, dtype=np.float32)

    tumor_s0_median = float(np.median(s0[tumor]))
    floor = max(1.0, 0.05 * tumor_s0_median)

    pe_t = _pe(s0[tumor], s1[tumor], floor)
    out[f"{p}pe_mean"] = _safe(pe_t.mean())
    out[f"{p}pe_sd"] = _safe(pe_t.std())
    out[f"{p}pe_cv"] = _safe(pe_t.std() / abs(pe_t.mean())) if abs(pe_t.mean()) > 1e-6 else float("nan")
    for q in (10, 25, 50, 75, 90):
        out[f"{p}pe_p{q}"] = _safe(np.percentile(pe_t, q))
    out[f"{p}pe_skew"] = _safe(stats.skew(pe_t)) if pe_t.size > 2 else float("nan")
    out[f"{p}pe_kurtosis"] = _safe(stats.kurtosis(pe_t)) if pe_t.size > 3 else float("nan")
    out[f"{p}pe_entropy"] = _entropy(pe_t)

    voxel_mm3 = float(np.prod(spacing))
    enh = pe_t >= pe_threshold
    ser_t = _ser(s0[tumor], s1[tumor], s2[tumor])
    ser_e = ser_t[enh & np.isfinite(ser_t)]
    out[f"{p}ftv_mm3"] = _safe(enh.sum() * voxel_mm3)
    out[f"{p}ftv_fraction"] = _safe(enh.mean())
    if ser_e.size > 0:
        out[f"{p}ser_median"] = _safe(np.median(ser_e))
        out[f"{p}ser_p90"] = _safe(np.percentile(ser_e, 90))
        out[f"{p}washout_fraction"] = _safe((ser_e > 1.1).mean())
        out[f"{p}plateau_fraction"] = _safe(((ser_e >= 0.9) & (ser_e <= 1.1)).mean())
        out[f"{p}persistent_fraction"] = _safe((ser_e < 0.9).mean())
        out[f"{p}washout_volume_mm3"] = _safe((ser_e > 1.1).sum() * voxel_mm3)
    else:
        for k in ("ser_median", "ser_p90", "washout_fraction", "plateau_fraction", "persistent_fraction"):
            out[f"{p}{k}"] = float("nan")
        out[f"{p}washout_volume_mm3"] = 0.0

    core = np.asarray(core_mask, dtype=bool)
    rim = np.asarray(rim_mask, dtype=bool)
    if core.any() and rim.any():
        pe_core = _pe(s0[core], s1[core], floor)
        pe_rim = _pe(s0[rim], s1[rim], floor)
        out[f"{p}rim_minus_core_pe_median"] = _safe(np.median(pe_rim) - np.median(pe_core))

    # Peritumoral shells (outside the tumour), spacing-aware.
    dist_out = ndi.distance_transform_edt(~tumor, sampling=spacing)
    tissue = s0 > 0.1 * tumor_s0_median
    for name, lo, hi in (("ring1", 0.0, ring1_mm), ("ring2", ring1_mm, ring2_mm)):
        shell = (dist_out > lo) & (dist_out <= hi) & tissue
        if shell.sum() < 5:
            continue
        pe_s = _pe(s0[shell], s1[shell], floor)
        ser_s = _ser(s0[shell], s1[shell], s2[shell])
        out[f"{p}peri_{name}_pe_mean"] = _safe(pe_s.mean())
        out[f"{p}peri_{name}_pe_median"] = _safe(np.median(pe_s))
        out[f"{p}peri_{name}_ser_median"] = _safe(np.nanmedian(ser_s)) if np.isfinite(ser_s).any() else float("nan")
        out[f"{p}peri_{name}_enh_fraction"] = _safe((pe_s >= pe_threshold).mean())
        if abs(np.median(pe_t)) > 1e-6:
            out[f"{p}peri_{name}_to_tumor_pe_ratio"] = _safe(np.median(pe_s) / np.median(pe_t))
    return out


#: Columns produced above (used for feature presets).
HETEROGENEITY_COLUMNS: list[str] = (
    [f"{FEATURE_PREFIX}pe_{k}" for k in ("mean", "sd", "cv", "p10", "p25", "p50", "p75", "p90", "skew", "kurtosis", "entropy")]
    + [f"{FEATURE_PREFIX}{k}" for k in ("ftv_mm3", "ftv_fraction", "ser_median", "ser_p90", "washout_fraction",
                                        "plateau_fraction", "persistent_fraction", "washout_volume_mm3",
                                        "rim_minus_core_pe_median")]
)
PERITUMORAL_COLUMNS: list[str] = [
    f"{FEATURE_PREFIX}peri_{r}_{k}"
    for r in ("ring1", "ring2")
    for k in ("pe_mean", "pe_median", "ser_median", "enh_fraction", "to_tumor_pe_ratio")
]
