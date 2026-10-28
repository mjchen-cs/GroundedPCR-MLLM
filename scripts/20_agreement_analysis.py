#!/usr/bin/env python3
"""
Step 20 (offline): does disagreement between an MLLM and B1 flag cases in which B1 is wrong?

For each provider and each configuration in which the MLLM has NOT seen the B1 output
(M1, M2, M3f), both models are dichotomised at the Youden-optimal threshold of their own
development predictions (B1: out-of-fold). On each split we report
  * the share of patients on which the two disagree,
  * B1's error rate when they agree vs. disagree,
  * B1's error rate in an equally large group flagged by B1's own low confidence
    (smallest |p - threshold|) -- the obvious alternative flag,
  * the AUROC for detecting B1 errors of a logistic model fitted on development data with
    B1 confidence and predicted class, without and with the MLLM's probability for the
    opposite class (paired DeLong test).

Usage: python scripts/20_agreement_analysis.py --mllm_dir results/mllm --out_dir paper
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

_spec = importlib.util.spec_from_file_location("ev14", Path(__file__).resolve().parent / "14_evaluate_mllm.py")
ev14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev14)

LABEL = {"zhipu": "GLM-4.5V", "siliconflow": "Qwen3-VL-32B", "mock": "Mock"}


def youden(y, p):
    f, t, th = roc_curve(y, p)
    return float(th[np.argmax(t - f)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mllm_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--providers", nargs="+", default=["zhipu", "siliconflow"])
    ap.add_argument("--configs", nargs="+", default=["M1", "M2", "M3f"])
    a = ap.parse_args()
    for d in ("tables", "analysis"):
        (a.out_dir / d).mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in (a.mllm_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base = pd.DataFrame([{k: c[k] for k in ("pid", "split", "label", "xgb_probability")} for c in cases])
    dev = base[base.split == "development"]
    tb = youden(dev.label, dev.xgb_probability)
    rows, tex = [], []
    for prov in a.providers:
        rd = sorted((a.mllm_dir / "runs").glob(f"{prov}__*"))[0]
        for cfg in a.configs:
            X = {}
            for sp in ("development", "heldout_test"):
                r = pd.read_csv(rd / f"{cfg}_{sp}.csv").groupby("pid").probability.mean()
                d = base[base.split == sp].copy()
                d["m"] = d.pid.map(r).fillna(0.5)
                X[sp] = d
            tm = youden(X["development"].label, X["development"].m)
            for sp, d in X.items():
                d["b"] = (d.xgb_probability >= tb).astype(int)
                d["err"] = (d.b != d.label).astype(int)
                d["conf"] = (d.xgb_probability - tb).abs()
                d["opp"] = np.where(d.b == 1, 1 - d.m, d.m)       # MLLM probability of the class B1 did not predict
                d["dis"] = d.b != (d.m >= tm).astype(int)
            f0, f1 = ["conf", "b"], ["conf", "b", "opp"]
            lr0 = LogisticRegression().fit(X["development"][f0], X["development"].err)
            lr1 = LogisticRegression().fit(X["development"][f1], X["development"].err)
            row = {"provider": prov, "config": cfg, "b1_threshold": tb, "mllm_threshold": tm}
            for sp, d in X.items():
                k = int(d.dis.sum())
                low = np.zeros(len(d), bool)
                low[np.argsort(d.conf.to_numpy())[:k]] = True
                p0, p1 = lr0.predict_proba(d[f0])[:, 1], lr1.predict_proba(d[f1])[:, 1]
                a0, a1, pv = ev14.delong_test(d.err.to_numpy(), p1, p0)
                s = "dev" if sp == "development" else "test"
                row.update({f"{s}_disagree": d.dis.mean(), f"{s}_err_agree": d.err[~d.dis].mean(),
                            f"{s}_err_disagree": d.err[d.dis].mean(), f"{s}_err_lowconf": d.err[low].mean(),
                            f"{s}_errauc_conf": roc_auc_score(d.err, p0), f"{s}_errauc_conf_mllm": roc_auc_score(d.err, p1),
                            f"{s}_delong_p": pv})
            rows.append(row)
            tex.append(f"{LABEL.get(prov, prov)} {cfg} & {100 * row['test_disagree']:.0f} & {100 * row['test_err_agree']:.0f} & "
                       f"{100 * row['test_err_disagree']:.0f} & {100 * row['test_err_lowconf']:.0f} & "
                       f"{row['dev_errauc_conf']:.3f} $\\to$ {row['dev_errauc_conf_mllm']:.3f} & "
                       f"{row['test_errauc_conf']:.3f} $\\to$ {row['test_errauc_conf_mllm']:.3f} ({row['test_delong_p']:.2f}) \\\\")
    df = pd.DataFrame(rows)
    df.to_csv(a.out_dir / "analysis" / "agreement.csv", index=False)
    (a.out_dir / "tables" / "table_agreement_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    pd.set_option("display.width", 220)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
