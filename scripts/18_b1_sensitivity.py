#!/usr/bin/env python3
"""
Step 18: sensitivity analyses of the reference model B1 (development CV only; the
locked B1 and its single held-out evaluation are not changed).

Inputs
  --b1_dir      the locked B1 run (step 03 with --run_final_test): xgboost_predictions.csv,
                xgboost_cv_oof_inputs.csv
  --runs        extra step-03 CV runs (name=dir), e.g. no class weighting, depth 3, + treatment arm
Outputs (in --out_dir)
  tables/table_b1_sensitivity_rows.tex, analysis/b1_sensitivity.csv, figures/fig_b1_calibration.pdf

Analyses
  * recalibration of the locked B1: Platt scaling of logit(p) cross-fitted over the
    development folds (calibrator for fold k fitted on the other folds' OOF predictions);
    for the held-out set the calibrator is fitted on all development OOF predictions
    (development data only) and applied to the locked B1 probabilities;
  * logistic regression on the same 13 inputs (median imputation + standardisation fitted
    per training fold; Tofts-PINN features from the out-of-fold export);
  * the extra XGBoost runs given with --runs.
Metrics: fold-mean AUROC (SD), pooled OOF AUROC, Brier, calibration-in-the-large
(mean predicted - observed), calibration slope.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def metrics(y, p, folds=None):
    y, p = np.asarray(y), np.asarray(p)
    out = {"auroc_pooled": roc_auc_score(y, p), "brier": float(np.mean((p - y) ** 2)),
           "citl": float(p.mean() - y.mean())}
    lr = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    out["slope"] = float(lr.coef_[0, 0])
    if folds is not None:
        fa = [roc_auc_score(y[folds == f], p[folds == f]) for f in np.unique(folds)]
        out["auroc_mean"], out["auroc_sd"] = float(np.mean(fa)), float(np.std(fa, ddof=1))
    return out


def platt_crossfit(y, p, folds):
    q = np.empty_like(p)
    for f in np.unique(folds):
        tr = folds != f
        cal = LogisticRegression(C=1e6).fit(logit(p[tr]).reshape(-1, 1), y[tr])
        q[folds == f] = cal.predict_proba(logit(p[folds == f]).reshape(-1, 1))[:, 1]
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--b1_dir", type=Path, required=True)
    ap.add_argument("--runs", nargs="*", default=[], help="label=run_dir pairs (development CV runs of step 03)")
    ap.add_argument("--out_dir", type=Path, required=True)
    a = ap.parse_args()
    for d in ("tables", "analysis", "figures"):
        (a.out_dir / d).mkdir(parents=True, exist_ok=True)

    pr = pd.read_csv(a.b1_dir / "xgboost_predictions.csv")
    dev = pr[pr.split_group == "development_oof"].copy()
    tst = pr[pr.split_group == "heldout_test"].copy()
    y, p, folds = dev.y_true.to_numpy(int), dev.y_prob.to_numpy(float), dev.fold_index.to_numpy(int)
    rows = []
    rows.append({"model": "B1 (locked)", "split": "development", **metrics(y, p, folds)})
    q = platt_crossfit(y, p, folds)
    rows.append({"model": "B1 + Platt recalibration", "split": "development", **metrics(y, q, folds)})
    cal = LogisticRegression(C=1e6).fit(logit(p).reshape(-1, 1), y)
    yt, pt = tst.y_true.to_numpy(int), tst.y_prob.to_numpy(float)
    qt = cal.predict_proba(logit(pt).reshape(-1, 1))[:, 1]
    rows.append({"model": "B1 (locked)", "split": "heldout_test", **metrics(yt, pt)})
    rows.append({"model": "B1 + Platt recalibration", "split": "heldout_test", **metrics(yt, qt)})

    # logistic regression on the same inputs
    oi = pd.read_csv(a.b1_dir / "xgboost_cv_oof_inputs.csv")
    lab = dev.set_index("pid").y_true
    oi = oi[oi.pid.isin(lab.index)].copy()
    oi["y"] = oi.pid.map(lab).astype(int)
    feats = [c for c in oi.columns if c not in ("pid", "fold_index", "y", "split")]
    plr = np.empty(len(oi))
    for f in sorted(oi.fold_index.unique()):
        tr, te = (oi.fold_index != f).to_numpy(), (oi.fold_index == f).to_numpy()
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(oi.loc[tr, feats], oi.y[tr])
        plr[te] = m.predict_proba(oi.loc[te, feats])[:, 1]
    rows.append({"model": f"Logistic regression, same {len(feats)} inputs", "split": "development",
                 **metrics(oi.y.to_numpy(), plr, oi.fold_index.to_numpy())})

    for spec in a.runs:
        label, d = spec.split("=", 1)
        r = pd.read_csv(Path(d) / "xgboost_predictions.csv")
        r = r[r.split_group == "development_oof"]
        mj = json.loads((Path(d) / "xgboost_metrics.json").read_text())
        rows.append({"model": label, "split": "development", "n_inputs": len(mj["feature_columns"]),
                     **metrics(r.y_true.to_numpy(int), r.y_prob.to_numpy(float), r.fold_index.to_numpy(int))})

    df = pd.DataFrame(rows)
    df.to_csv(a.out_dir / "analysis" / "b1_sensitivity.csv", index=False)
    tex = []
    for _, r in df[df.split == "development"].iterrows():
        tex.append(f"{r.model} & {r.auroc_mean:.3f} $\\pm$ {r.auroc_sd:.3f} & {r.auroc_pooled:.3f} & {r.brier:.3f} & "
                   f"{r.citl:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000") + f" & {r.slope:.2f} \\\\")
    for _, r in df[df.split == "heldout_test"].iterrows():
        tex.append(f"{r.model}, held-out & -- & {r.auroc_pooled:.3f} & {r.brier:.3f} & {r.citl:+.3f} & {r.slope:.2f} \\\\")
    (a.out_dir / "tables" / "table_b1_sensitivity_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # reliability plot: locked vs recalibrated (development OOF and held-out)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    for k, (yy, pp, qq, title) in enumerate([(y, p, q, "Development (OOF)"), (yt, pt, qt, "Held-out")]):
        a2 = ax[k]
        a2.plot([0, 1], [0, 1], color="#e8e7e3", lw=1)
        for vals, col, lab_ in ((pp, "#52514e", "locked B1"), (qq, "#2a78d6", "recalibrated")):
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, 6)))
            idx = np.clip(np.digitize(vals, edges[1:-1]), 0, len(edges) - 2)
            mp = [vals[idx == i].mean() for i in range(len(edges) - 1)]
            oy = [yy[idx == i].mean() for i in range(len(edges) - 1)]
            a2.plot(mp, oy, "-o", color=col, ms=3, lw=1.2, label=lab_)
        a2.set_xlim(0, 1); a2.set_ylim(0, 1); a2.set_title(title, fontsize=7, loc="left")
        a2.set_xlabel("Mean predicted")
    ax[0].set_ylabel("Observed pCR rate"); ax[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(); fig.savefig(a.out_dir / "figures" / "fig_b1_calibration.pdf"); fig.savefig(a.out_dir / "figures" / "fig_b1_calibration.png", dpi=200)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
