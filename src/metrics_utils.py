"""
Metrics utilities for binary pCR prediction.

All metrics here operate on the standard ``y_true`` (0/1 labels) and
``y_prob`` (predicted positive-class probabilities) pair, plus a chosen
decision threshold for the hard-label metrics.

It also provides the statistics reported in the manuscript:

- ``bootstrap_metric_ci``: non-parametric bootstrap 95% CIs (10,000 resamples
  by default) for AUROC, AUPRC, F1, sensitivity, specificity and accuracy at a
  fixed decision threshold (Table II, Table IV);
- ``assign_molecular_subtype``: HR+/HER2-, HER2+, triple-negative, unknown;
- ``calibration_table`` / ``brier_score``: calibration curves (Fig. 4a);
- ``decision_curve``: net benefit vs. threshold probability (Fig. 4b).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        v = float(value)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _coerce_arrays(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(pd.to_numeric(pd.Series(y_true), errors="coerce"), dtype=float)
    y_prob = np.asarray(pd.to_numeric(pd.Series(y_prob), errors="coerce"), dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(
            f"y_true and y_prob have incompatible shapes: {y_true.shape} vs {y_prob.shape}"
        )
    mask = np.isfinite(y_true) & np.isfinite(y_prob)
    return y_true[mask], y_prob[mask]


def compute_binary_metrics(
    y_true,
    y_prob,
    decision_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute the standard panel of binary-classification metrics used in this
    project: AUROC, AUPRC (a.k.a. average precision), accuracy, balanced
    accuracy, precision (PPV), recall (sensitivity), specificity, F1, log loss,
    plus the confusion-matrix counts and basic sample sizes.

    NaNs and inf values in either input are dropped before scoring.

    Returns a JSON-serialisable dict.
    """
    y_true_clean, y_prob_clean = _coerce_arrays(y_true, y_prob)
    threshold = float(decision_threshold)

    n_total = int(y_true_clean.size)
    n_positive = int((y_true_clean > 0.5).sum()) if n_total > 0 else 0
    n_negative = int(n_total - n_positive)

    out: dict[str, Any] = {
        "n_samples": n_total,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "decision_threshold": threshold,
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
        "accuracy": float("nan"),
        "balanced_accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "specificity": float("nan"),
        "f1": float("nan"),
        "log_loss": float("nan"),
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 0,
    }

    if n_total == 0 or n_positive == 0 or n_negative == 0:
        return out

    y_int = (y_true_clean > 0.5).astype(int)
    y_pred = (y_prob_clean >= threshold).astype(int)

    try:
        out["roc_auc"] = _safe_float(roc_auc_score(y_int, y_prob_clean))
    except Exception:
        pass
    try:
        out["average_precision"] = _safe_float(average_precision_score(y_int, y_prob_clean))
    except Exception:
        pass
    try:
        out["accuracy"] = _safe_float(accuracy_score(y_int, y_pred))
        out["balanced_accuracy"] = _safe_float(balanced_accuracy_score(y_int, y_pred))
        out["precision"] = _safe_float(precision_score(y_int, y_pred, zero_division=0))
        out["recall"] = _safe_float(recall_score(y_int, y_pred, zero_division=0))
        out["f1"] = _safe_float(f1_score(y_int, y_pred, zero_division=0))
    except Exception:
        pass

    try:
        cm = confusion_matrix(y_int, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out["true_negative"] = int(tn)
        out["false_positive"] = int(fp)
        out["false_negative"] = int(fn)
        out["true_positive"] = int(tp)
        denom_spec = tn + fp
        out["specificity"] = _safe_float(tn / denom_spec) if denom_spec > 0 else float("nan")
    except Exception:
        pass

    try:
        # log_loss requires both classes present in y_true; we already checked.
        clipped = np.clip(y_prob_clean, 1e-7, 1.0 - 1e-7)
        out["log_loss"] = _safe_float(log_loss(y_int, clipped, labels=[0, 1]))
    except Exception:
        pass

    return out


def select_best_threshold(
    y_true,
    y_prob,
    metric: str = "f1",
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
) -> dict[str, Any]:
    """
    Sweep a decision threshold over [threshold_min, threshold_max] in steps of
    ``threshold_step`` and select the one that maximises ``metric`` on the
    supplied (y_true, y_prob) pair.

    ``metric`` must be one of: ``"f1"``, ``"balanced_accuracy"``.

    Returns a dict with the chosen threshold, the chosen metric value, and the
    full panel of metrics at that threshold.
    """
    metric = str(metric).strip().lower()
    if metric not in {"f1", "balanced_accuracy"}:
        raise ValueError(f"Unsupported threshold-selection metric: {metric!r}")

    if not (threshold_min < threshold_max):
        raise ValueError("threshold_min must be < threshold_max")
    if threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")

    grid = np.arange(threshold_min, threshold_max + 0.5 * threshold_step, threshold_step)
    grid = np.clip(grid, 1e-3, 1.0 - 1e-3)

    best_score = -np.inf
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None

    for t in grid:
        m = compute_binary_metrics(y_true, y_prob, decision_threshold=float(t))
        score = m.get(metric, float("nan"))
        if not np.isfinite(score):
            continue
        if score > best_score:
            best_score = float(score)
            best_threshold = float(t)
            best_metrics = m

    if best_metrics is None:
        # Fallback: nothing scored finite; return metrics at 0.5 anyway.
        best_metrics = compute_binary_metrics(y_true, y_prob, decision_threshold=0.5)
        best_threshold = 0.5

    return {
        "threshold_metric": metric,
        "decision_threshold": float(best_threshold),
        "best_metric_value": float(best_metrics.get(metric, float("nan"))),
        "metrics_at_threshold": best_metrics,
    }


# --- Bootstrap confidence intervals -----------------------------------------

BOOTSTRAP_METRICS = ("roc_auc", "average_precision", "f1", "recall", "specificity", "accuracy")


def _fast_auroc(y: np.ndarray, p: np.ndarray) -> float:
    from scipy.stats import rankdata

    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fast_average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Step-wise average precision, identical to sklearn.average_precision_score."""
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    p_sorted = p[order]
    y_sorted = y[order]
    # collapse ties: keep the last index of each distinct score
    distinct = np.where(np.diff(p_sorted))[0]
    idx = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = (1 + idx) - tps
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / n_pos
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def _threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    tn = float(np.sum(~pred & (y == 0)))
    n = tp + fp + fn + tn
    return {
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else float("nan"),
        "accuracy": (tp + tn) / n if n > 0 else float("nan"),
    }


def _metric_panel(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    out = {"roc_auc": _fast_auroc(y, p), "average_precision": _fast_average_precision(y, p)}
    out.update(_threshold_metrics(y, p, threshold))
    return out


def bootstrap_metric_ci(
    y_true,
    y_prob,
    decision_threshold: float,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """
    Non-parametric bootstrap (resampling patients with replacement). Resamples
    that contain only one class are skipped for the rank-based metrics.
    Returns {metric: {"point", "ci_low", "ci_high", "n_valid_resamples"}}.
    """
    y, p = _coerce_arrays(y_true, y_prob)
    y = (y > 0.5).astype(int)
    point = _metric_panel(y, p, float(decision_threshold))
    rng = np.random.default_rng(int(random_state))
    n = y.size
    samples = {k: [] for k in BOOTSTRAP_METRICS}
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        m = _metric_panel(yb, pb, float(decision_threshold))
        for k in BOOTSTRAP_METRICS:
            samples[k].append(m[k])
    out: dict[str, dict[str, float]] = {}
    for k in BOOTSTRAP_METRICS:
        arr = np.asarray(samples[k], dtype=float)
        arr = arr[np.isfinite(arr)]
        out[k] = {
            "point": _safe_float(point[k]),
            "ci_low": float(np.percentile(arr, 100 * alpha / 2)) if arr.size else float("nan"),
            "ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))) if arr.size else float("nan"),
            "n_valid_resamples": int(arr.size),
        }
    return out


# --- Molecular subtype --------------------------------------------------------


def assign_molecular_subtype(hr, her2) -> np.ndarray:
    """HER2+ irrespective of HR; TNBC = HR-/HER2-; HR+/HER2-; otherwise Unknown."""
    hr = pd.to_numeric(pd.Series(hr), errors="coerce").to_numpy(dtype=float)
    her2 = pd.to_numeric(pd.Series(her2), errors="coerce").to_numpy(dtype=float)
    out = np.full(hr.shape, "Unknown", dtype=object)
    out[her2 == 1] = "HER2+"
    out[(hr == 1) & (her2 == 0)] = "HR+/HER2-"
    out[(hr == 0) & (her2 == 0)] = "Triple-negative"
    return out


# --- Calibration and decision-curve analysis ---------------------------------


def brier_score(y_true, y_prob) -> float:
    y, p = _coerce_arrays(y_true, y_prob)
    return float(np.mean((p - (y > 0.5).astype(float)) ** 2)) if y.size else float("nan")


def calibration_table(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform") -> pd.DataFrame:
    """Observed pCR fraction vs. mean predicted probability per bin."""
    y, p = _coerce_arrays(y_true, y_prob)
    y = (y > 0.5).astype(int)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = bins == b
        if not m.any():
            continue
        rows.append({
            "bin_index": int(b),
            "bin_low": float(edges[b]),
            "bin_high": float(edges[b + 1]),
            "n": int(m.sum()),
            "mean_predicted": float(p[m].mean()),
            "observed_fraction": float(y[m].mean()),
        })
    return pd.DataFrame(rows)


def decision_curve(y_true, y_prob, thresholds: Sequence[float] | None = None) -> pd.DataFrame:
    """Net benefit of the model, treat-all and treat-none (Vickers & Elkin)."""
    y, p = _coerce_arrays(y_true, y_prob)
    y = (y > 0.5).astype(int)
    n = max(int(y.size), 1)
    prevalence = float(y.mean()) if y.size else float("nan")
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 0.991, 0.01), 4)
    rows = []
    for pt in thresholds:
        pt = float(pt)
        w = pt / (1.0 - pt)
        pred = p >= pt
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        rows.append({
            "threshold": pt,
            "net_benefit_model": tp / n - fp / n * w,
            "net_benefit_treat_all": prevalence - (1.0 - prevalence) * w,
            "net_benefit_treat_none": 0.0,
        })
    return pd.DataFrame(rows)


__all__ = [
    "compute_binary_metrics",
    "select_best_threshold",
    "bootstrap_metric_ci",
    "assign_molecular_subtype",
    "brier_score",
    "calibration_table",
    "decision_curve",
]
