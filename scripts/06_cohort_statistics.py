#!/usr/bin/env python3
"""
Step 6: Cohort-level statistics and clinical-utility analyses
(Table II, Table IV, Fig. 4; Sec. II-H).

Each ``--cohort`` is ``NAME=PREDICTIONS_CSV`` or ``NAME=PREDICTIONS_CSV::SPLIT_VALUE``
(rows filtered by ``split_group == SPLIT_VALUE``). Typical call:

    python scripts/06_cohort_statistics.py \
        --cohort "Development=out_xgb/xgboost_predictions.csv::development_oof" \
        --cohort "Held-out test=out_xgb/xgboost_predictions.csv::heldout_test" \
        --cohort "I-SPY 1=external/ISPY1/predictions_ISPY1.csv" \
        --cohort "Duke=external/Duke/predictions_Duke.csv" \
        --development_name Development \
        --cv_fold_metrics_csv out_xgb/xgboost_cv_fold_metrics.csv \
        --model_dir out_xgb/final_model \
        --output_dir out_stats

Outputs
- ``table_ii_performance.csv``: development = mean +/- SD across CV folds (at
  the single OOF threshold); other cohorts = point estimate (bootstrap 95% CI).
- ``table_iv_subtypes.csv``: subtype-stratified AUROC / AUPRC with bootstrap CIs
  (HER2+ irrespective of HR; TNBC = HR-/HER2-; Unknown subtype not evaluated).
- ``calibration_bins.csv``, ``brier_scores.csv``, ``decision_curves.csv``,
  ``fig4_calibration_dca.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from metrics_utils import (
    assign_molecular_subtype,
    bootstrap_metric_ci,
    brier_score,
    calibration_table,
    decision_curve,
)

METRIC_LABELS = [
    ("roc_auc", "AUROC"),
    ("average_precision", "AUPRC"),
    ("f1", "F1-score"),
    ("recall", "Sensitivity"),
    ("specificity", "Specificity"),
    ("accuracy", "Accuracy"),
]
SUBTYPE_ORDER = ["HR+/HER2-", "HER2+", "Triple-negative", "Unknown"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Table II / IV and Fig. 4 statistics.")
    p.add_argument("--cohort", action="append", required=True, help="NAME=CSV or NAME=CSV::SPLIT_VALUE (repeatable).")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model_dir", type=Path, default=None, help="Locked model dir (reads the decision threshold).")
    p.add_argument("--decision_threshold", type=float, default=None, help="Overrides the threshold from --model_dir.")
    p.add_argument("--development_name", type=str, default=None,
                   help="Cohort reported as mean +/- SD across folds (requires --cv_fold_metrics_csv).")
    p.add_argument("--cv_fold_metrics_csv", type=Path, default=None)
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--calibration_bins", type=int, default=10)
    p.add_argument("--calibration_strategy", type=str, default="uniform", choices=["uniform", "quantile"])
    p.add_argument("--dca_max_threshold", type=float, default=0.80)
    p.add_argument("--split_col", type=str, default="split_group")
    return p


def _load_cohort(spec: str, split_col: str) -> tuple[str, pd.DataFrame]:
    if "=" not in spec:
        raise ValueError(f"--cohort must be NAME=CSV[::SPLIT], got {spec!r}")
    name, rest = spec.split("=", 1)
    path, split_value = (rest.split("::", 1) + [None])[:2] if "::" in rest else (rest, None)
    df = pd.read_csv(path)
    if split_value is not None:
        df = df.loc[df[split_col].astype(str) == split_value].copy()
    df = df.loc[pd.to_numeric(df["y_true"], errors="coerce").notna()].copy()
    if "molecular_subtype" not in df.columns and {"clinical_hr", "clinical_her2"}.issubset(df.columns):
        df["molecular_subtype"] = assign_molecular_subtype(df["clinical_hr"], df["clinical_her2"])
    return name.strip(), df


def _fmt_ci(d: dict) -> str:
    return f"{d['point']:.3f} ({d['ci_low']:.3f}-{d['ci_high']:.3f})"


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.decision_threshold
    if threshold is None and args.model_dir is not None:
        with open(args.model_dir / "model_card.json", encoding="utf-8") as f:
            threshold = float(json.load(f)["decision_threshold"])
    if threshold is None:
        raise ValueError("Provide --model_dir or --decision_threshold.")

    cohorts = [_load_cohort(spec, args.split_col) for spec in args.cohort]

    # ---- Table II -------------------------------------------------------------
    table_ii: dict[str, dict[str, str]] = {}
    raw_ci: dict[str, dict] = {}
    for name, df in cohorts:
        col: dict[str, str] = {}
        if name == args.development_name and args.cv_fold_metrics_csv is not None:
            folds = pd.read_csv(args.cv_fold_metrics_csv)
            for key, label in METRIC_LABELS:
                v = pd.to_numeric(folds[key], errors="coerce").dropna()
                col[label] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}"
        else:
            ci = bootstrap_metric_ci(df["y_true"], df["y_prob"], threshold,
                                     n_bootstrap=args.n_bootstrap, random_state=args.random_state)
            raw_ci[name] = ci
            for key, label in METRIC_LABELS:
                col[label] = _fmt_ci(ci[key])
        col["n"] = str(len(df))
        col["pCR positive"] = f"{int(df['y_true'].sum())} ({100 * df['y_true'].mean():.1f}%)"
        table_ii[name] = col
    table_ii_df = pd.DataFrame(table_ii)
    table_ii_df.index.name = "Performance metric"
    table_ii_df.to_csv(args.output_dir / "table_ii_performance.csv", encoding="utf-8-sig")
    with open(args.output_dir / "table_ii_bootstrap_raw.json", "w", encoding="utf-8") as f:
        json.dump({"decision_threshold": threshold, "n_bootstrap": args.n_bootstrap, "cohorts": raw_ci}, f, indent=2)

    # ---- Table IV -------------------------------------------------------------
    rows = []
    for name, df in cohorts:
        if name == args.development_name or "molecular_subtype" not in df.columns:
            continue
        for subtype in SUBTYPE_ORDER + ["Overall cohort"]:
            sub = df if subtype == "Overall cohort" else df.loc[df["molecular_subtype"] == subtype]
            if sub.empty:
                continue
            row = {"Cohort": name, "Molecular subtype": subtype,
                   "n (pCR%)": f"{len(sub)} ({100 * sub['y_true'].mean():.2f}%)"}
            evaluable = subtype != "Unknown" and sub["y_true"].nunique() == 2
            if evaluable:
                ci = bootstrap_metric_ci(sub["y_true"], sub["y_prob"], threshold,
                                         n_bootstrap=args.n_bootstrap, random_state=args.random_state)
                row["AUROC (95% CI)"] = _fmt_ci(ci["roc_auc"])
                row["AUPRC (95% CI)"] = _fmt_ci(ci["average_precision"])
            else:
                row["AUROC (95% CI)"] = row["AUPRC (95% CI)"] = "Not evaluated"
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_dir / "table_iv_subtypes.csv", index=False, encoding="utf-8-sig")

    # ---- Fig. 4: calibration + DCA -------------------------------------------
    cal_rows, brier_rows, dca_rows = [], [], []
    for name, df in cohorts:
        cal = calibration_table(df["y_true"], df["y_prob"], n_bins=args.calibration_bins, strategy=args.calibration_strategy)
        cal.insert(0, "cohort", name)
        cal_rows.append(cal)
        brier_rows.append({"cohort": name, "brier_score": brier_score(df["y_true"], df["y_prob"]), "n": len(df)})
        dca = decision_curve(df["y_true"], df["y_prob"])
        dca.insert(0, "cohort", name)
        dca_rows.append(dca)
    cal_df = pd.concat(cal_rows, ignore_index=True)
    brier_df = pd.DataFrame(brier_rows)
    dca_df = pd.concat(dca_rows, ignore_index=True)
    cal_df.to_csv(args.output_dir / "calibration_bins.csv", index=False)
    brier_df.to_csv(args.output_dir / "brier_scores.csv", index=False)
    dca_df.to_csv(args.output_dir / "decision_curves.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax_cal = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[1, 0], sharex=ax_cal)
    ax_dca = fig.add_subplot(gs[:, 1])
    ax_cal.plot([0, 1], [0, 1], ":", color="0.5", lw=1)
    for (name, df), (_, b) in zip(cohorts, brier_df.iterrows()):
        c = cal_df.loc[cal_df["cohort"] == name]
        ax_cal.plot(c["mean_predicted"], c["observed_fraction"], "o-", ms=4, label=f"{name} (BS={b['brier_score']:.3f})")
        ax_hist.hist(df["y_prob"], bins=20, range=(0, 1), histtype="step", lw=1.2)
    ax_cal.set_ylabel("Observed pCR fraction")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.legend(fontsize=8, loc="upper left")
    ax_cal.set_title("(a) Calibration")
    ax_hist.set_xlabel("Predicted pCR probability")
    ax_hist.set_ylabel("Count")
    for name, _ in cohorts:
        d = dca_df.loc[(dca_df["cohort"] == name) & (dca_df["threshold"] <= args.dca_max_threshold)]
        ax_dca.plot(d["threshold"], d["net_benefit_model"], lw=1.5, label=name)
    ref = dca_df.loc[(dca_df["cohort"] == cohorts[0][0]) & (dca_df["threshold"] <= args.dca_max_threshold)]
    ax_dca.plot(ref["threshold"], ref["net_benefit_treat_all"], "--", color="0.6", lw=1, label=f"Treat all ({cohorts[0][0]})")
    ax_dca.axhline(0, color="k", ls="--", lw=1, label="Treat none")
    ax_dca.set_ylim(-0.05, max(0.05, float(dca_df["net_benefit_model"].max()) * 1.1))
    ax_dca.set_xlabel("Threshold probability (Pt)")
    ax_dca.set_ylabel("Net benefit")
    ax_dca.set_title("(b) Decision curve analysis")
    ax_dca.legend(fontsize=8)
    fig.savefig(args.output_dir / "fig4_calibration_dca.png", dpi=300)
    print(table_ii_df.to_string())
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
