#!/usr/bin/env python3
"""
Step 14: Evaluate MLLM predictions and report faithfulness; write LaTeX tables.

Prediction (per provider x configuration, plus B1 and stacked fusion F1):
- development: AUROC / AUPRC per fold (mean +/- SD) and pooled, Brier score;
- held-out test: AUROC / AUPRC with bootstrap 95% CI, Brier score, and a
  DeLong test of AUROC against B1 (paired, same cases).
- Stacked fusion (F3 / F4) = logistic regression on [logit(B1), logit(MLLM)] fitted on the
  training folds of each development fold (out-of-fold), and on the whole
  development set for the held-out test.

Faithfulness (per reply, then averaged over replies):
- hallucination (unsupported-number) rate: numbers stated in the evidence list
  or in the report that are not among the numbers shown in that prompt
  (relative tolerance 1% or half a unit of the reported precision; percentages
  accepted; thousands separators removed). Report numbers that are bounded or
  rounded paraphrases (within 10%) or mm^3 -> cm^3 conversions of a provided
  value are counted separately as approximations, not as unsupported;
- direction contradiction rate: evidence items whose stated direction
  (supports / against pCR) disagrees with the sign of the case's B1 SHAP
  value (|SHAP| >= 0.02);
- anchoring rate: share of replies whose probability is within 0.01 of the
  B1 probability (copying the reference model);
- conclusion consistency: "likely_pcr" with p >= 0.5, "unlikely_pcr" with p < 0.5;
- stability: SD of the probability across repeats; Jaccard overlap of cited
  evidence keys across repeats.
Case-level bootstrap CIs are reported for the hallucination and contradiction rates.

Writes ``evaluation/*.csv`` and, with ``--paper_dir``, ``tables/*.tex`` and
``results_macros.tex`` that the manuscript \\input{}s.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mllm_prompts import build_messages, pick_examples, provided_values

CONFIG_WORD = {"M1": "MOne", "M2": "MTwo", "M3f": "MThreeF", "M3": "MThree", "M4": "MFour",
               "F3f": "FThreeF", "F3": "FThree", "F4": "FFour"}
CONFIG_LABEL = {"M1": "M1 text only", "M2": "M2 image + text", "M3f": "M3f + physics features",
                "M3": "M3 + features, B1 prob., SHAP", "M4": "M4 = M3 + few-shot",
                "F3f": "Fusion (B1 + M3f)", "F3": "Fusion (B1 + M3)", "F4": "Fusion (B1 + M4)"}
CONFIG_ORDER = ["M1", "M2", "M3f", "M3", "M4", "F3f", "F3", "F4"]
PROVIDER_WORD = {"zhipu": "Zhipu", "siliconflow": "SiliconFlow", "mock": "Mock"}


# ----------------------------------------------------------------------------- metrics
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def delong_test(y, p1, p2):
    """Paired DeLong test for two correlated AUROCs (Sun & Xu fast algorithm)."""
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)

    def comps(p):
        p = np.asarray(p, dtype=float)
        tx = stats.rankdata(p[pos]); ty = stats.rankdata(p[neg]); tz = stats.rankdata(np.r_[p[pos], p[neg]])
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01 = (tz[:m] - tx) / n
        v10 = 1.0 - (tz[m:] - ty) / m
        return auc, v01, v10

    a1, v01a, v10a = comps(p1)
    a2, v01b, v10b = comps(p2)
    s01 = np.cov(np.vstack([v01a, v01b]))
    s10 = np.cov(np.vstack([v10a, v10b]))
    s = s01 / m + s10 / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return a1, a2, float("nan")
    z = (a1 - a2) / math.sqrt(var)
    return a1, a2, float(2 * stats.norm.sf(abs(z)))


def boot_ci(y, p, fn, n_boot, seed=42):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(fn(y[idx], p[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def pred_metrics(df: pd.DataFrame, prob_col: str, split: str, n_boot: int, b1_col: str = "xgb_probability") -> dict:
    y, p = df["label"].to_numpy(), df[prob_col].to_numpy()
    out = {"n": int(len(df)), "brier": float(np.mean((p - y) ** 2))}
    out["auroc"] = float(roc_auc_score(y, p)); out["auprc"] = float(average_precision_score(y, p))
    if split == "development":
        fa = [roc_auc_score(g["label"], g[prob_col]) for _, g in df.groupby("fold_index") if g["label"].nunique() == 2]
        fp = [average_precision_score(g["label"], g[prob_col]) for _, g in df.groupby("fold_index") if g["label"].nunique() == 2]
        out.update(auroc_fold_mean=float(np.mean(fa)), auroc_fold_sd=float(np.std(fa, ddof=1)),
                   auprc_fold_mean=float(np.mean(fp)), auprc_fold_sd=float(np.std(fp, ddof=1)))
    else:
        out["auroc_ci"] = boot_ci(y, p, roc_auc_score, n_boot)
        out["auprc_ci"] = boot_ci(y, p, average_precision_score, n_boot)
    if prob_col != b1_col:
        _, _, pval = delong_test(y, p, df[b1_col].to_numpy())
        out["delong_p_vs_b1"] = pval
    return out


# ----------------------------------------------------------------------------- faithfulness
# A sign counts only when it does not follow a digit, so "10-15%" is read as 10 and 15, not 10 and -15.
NUM_RE = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*%?")
THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def _approximates(x: float, provided: list[float]) -> bool:
    """x is a rounded/bounded paraphrase (within 10%) or a mm^3 -> cm^3 conversion of a provided value."""
    for v in provided:
        if abs(v) < 100:  # only large measurements (volumes) are paraphrased this way
            continue
        for t in (v, v / 1000.0):
            if t != 0 and abs(x - t) <= 0.10 * abs(t):
                return True
    return False


def _decimals(token: str) -> int:
    t = token.strip().rstrip("%").strip()
    return len(t.split(".", 1)[1]) if "." in t else 0


def _supported(x: float, pool: list[float], pct: bool, token: str | None = None) -> bool:
    """x matches a provided value up to 1% or up to the rounding the writer used (half a unit in its last digit)."""
    half_unit = 0.5 * 10 ** (-_decimals(token)) if token is not None else 0.0
    cands = [(x, half_unit)]
    if pct or abs(x) > 1:
        cands.append((x / 100.0, half_unit / 100.0))
    for c, hu in cands:
        for v in pool:
            if abs(c - v) <= max(0.01 * abs(v), 1e-3, hu + 1e-9):
                return True
    return False


def faithfulness_row(rec: pd.Series, case: dict, prompt_text: str, provided: dict[str, float]) -> dict:
    pool = list(provided.values())
    for m in NUM_RE.finditer(prompt_text):  # numbers printed anywhere in the prompt (units, ranges) are allowed
        try:
            pool.append(float(m.group(0).rstrip("% ").strip()))
        except ValueError:
            pass
    own_p = rec["probability"]
    if own_p is not None and not (isinstance(own_p, float) and math.isnan(own_p)):
        pool.append(float(own_p))
    try:
        evidence = json.loads(rec["evidence"]) if isinstance(rec["evidence"], str) else []
    except json.JSONDecodeError:
        evidence = []
    n_num = n_bad = 0
    n_dir = n_contra = 0
    keys = []
    for e in evidence:
        if not isinstance(e, dict):
            continue
        key = str(e.get("feature", ""))
        keys.append(key)
        val = e.get("value")
        if isinstance(val, (int, float)) and key != "image":
            n_num += 1
            ref = provided.get(key)
            if ref is None or not _supported(float(val), [ref], False):
                n_bad += 1
        sv = case["shap"].get(key)
        d = e.get("direction")
        if sv is not None and abs(sv) >= 0.02 and d in ("supports_pcr", "against_pcr"):
            n_dir += 1
            if (d == "supports_pcr") != (sv > 0):
                n_contra += 1
    report = THOUSANDS_RE.sub("", str(rec.get("report") or ""))  # "34,739" -> "34739"
    for k in sorted(set(list(provided) + list(case["features"]) + ["core15", "safe_rim", "ring1", "ring2"]), key=len, reverse=True):
        report = report.replace(k, " ")
    n_rep = n_rep_bad = n_rep_approx = 0
    prov_vals = [v for v in provided.values() if not math.isnan(v)]
    for m in NUM_RE.finditer(report):
        s = m.group(0).strip()
        try:
            x = float(s.rstrip("%").strip())
        except ValueError:
            continue
        n_rep += 1
        if not _supported(x, pool, s.endswith("%"), s):
            if _approximates(x, prov_vals):
                n_rep_approx += 1  # e.g. "over 14,000 mm3" for 14788, "93 cm3" for 93328 mm3
            else:
                n_rep_bad += 1
    concl = rec.get("conclusion")
    consistent = None
    if own_p is not None and not (isinstance(own_p, float) and math.isnan(own_p)) and concl in ("likely_pcr", "unlikely_pcr"):
        consistent = int((concl == "likely_pcr") == (float(own_p) >= 0.5))
    return {"n_numbers": n_num + n_rep, "n_unsupported": n_bad + n_rep_bad, "n_approx": n_rep_approx,
            "n_dir": n_dir, "n_contra": n_contra,
            "consistent": consistent, "keys": "|".join(sorted(set(keys)))}


def case_boot_rate(df: pd.DataFrame, num: str, den: str, n_boot: int, seed: int = 7):
    g = df.groupby("pid")[[num, den]].sum()
    tot = g[den].sum()
    if tot == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = g.to_numpy()
    vals = []
    for _ in range(n_boot):
        s = arr[rng.integers(0, len(arr), len(arr))].sum(axis=0)
        if s[1] > 0:
            vals.append(s[0] / s[1])
    return float(g[num].sum() / tot), (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mllm_dir", type=Path, required=True)
    ap.add_argument("--paper_dir", type=Path, default=None, help="Write tables/*.tex and results_macros.tex here.")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--n_shots", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--b1_preset", default="", help="Name of the B1 feature preset (reported in the paper).")
    args = ap.parse_args()

    cases = [json.loads(l) for l in (args.mllm_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_pid = {c["pid"]: c for c in cases}
    dev_cases = [c for c in cases if c["split"] == "development"]
    base = pd.DataFrame([{k: c[k] for k in ("pid", "split", "fold_index", "label", "xgb_probability")} for c in cases])
    out_dir = args.mllm_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_rows, faith_rows, merged = [], [], {}
    for split in ("development", "heldout_test"):
        d = base[base["split"] == split]
        if len(d) and d["label"].nunique() == 2:
            pred_rows.append({"provider": "-", "model": "-", "config": "B1", "split": split,
                              **pred_metrics(d, "xgb_probability", split, args.n_bootstrap)})

    for run_dir in sorted((args.mllm_dir / "runs").glob("*__*")):
        provider, model = run_dir.name.split("__", 1)
        for csv in sorted(run_dir.glob("M*_*.csv")):
            config, split = csv.stem.split("_", 1)
            raw = pd.read_csv(csv)
            if raw.empty:
                continue
            raw["probability"] = pd.to_numeric(raw["probability"], errors="coerce")
            agg = raw.groupby("pid").agg(prob=("probability", "mean"), prob_sd=("probability", "std"),
                                         parse_ok=("parse_ok", "mean")).reset_index()
            d = base[base["split"] == split].merge(agg, on="pid", how="left")
            n_missing = int(d["prob"].isna().sum())
            d["prob"] = d["prob"].fillna(0.5)  # no parsable reply: neutral probability (counted below)
            merged[(provider, model, config, split)] = d
            if d["label"].nunique() == 2:
                pm = pred_metrics(d, "prob", split, args.n_bootstrap)
                xr = raw.merge(base[["pid", "xgb_probability"]], on="pid", how="left")
                anchor = float((np.abs(xr["probability"] - xr["xgb_probability"]) <= 0.01).mean())
                pred_rows.append({"provider": provider, "model": model, "config": config, "split": split,
                                  "anchor_rate": anchor,
                                  "parse_rate": float(raw["parse_ok"].mean()), "n_no_reply": n_missing,
                                  "prob_sd_mean": float(d["prob_sd"].mean()), **pm})
            # faithfulness, per reply
            frs = []
            for _, rec in raw.iterrows():
                case = by_pid[rec["pid"]]
                ex = pick_examples(case, dev_cases, args.n_shots, args.seed) if config == "M4" else None
                msgs = build_messages(config, case, None, ex)
                prompt_text = msgs[-1]["content"] if isinstance(msgs[-1]["content"], str) else ""
                fr = faithfulness_row(rec, case, prompt_text, provided_values(config, case))
                frs.append({"pid": rec["pid"], "repeat": rec["repeat"], **fr})
            fdf = pd.DataFrame(frs)
            fdf.to_csv(out_dir / f"faithfulness_{provider}__{model.replace('/', '_')}_{config}_{split}.csv", index=False)
            hal, hal_ci = case_boot_rate(fdf, "n_unsupported", "n_numbers", args.n_bootstrap)
            con, con_ci = case_boot_rate(fdf, "n_contra", "n_dir", args.n_bootstrap)
            jac = []
            for _, g in fdf.groupby("pid"):
                sets = [set(k.split("|")) - {""} for k in g["keys"]]
                for a, b in itertools.combinations(sets, 2):
                    if a or b:
                        jac.append(len(a & b) / len(a | b))
            faith_rows.append({
                "provider": provider, "model": model, "config": config, "split": split, "n_replies": len(fdf),
                "numbers_per_reply": float(fdf["n_numbers"].mean()),
                "hallucination_rate": hal, "hallucination_ci_low": hal_ci[0], "hallucination_ci_high": hal_ci[1],
                "replies_with_hallucination": float((fdf["n_unsupported"] > 0).mean()),
                "approximation_rate": float(fdf["n_approx"].sum() / max(1, fdf["n_numbers"].sum())),
                "contradiction_rate": con, "contradiction_ci_low": con_ci[0], "contradiction_ci_high": con_ci[1],
                "conclusion_consistency": float(pd.to_numeric(fdf["consistent"], errors="coerce").mean()),
                "evidence_jaccard": float(np.mean(jac)) if jac else float("nan"),
                "parse_rate": float(raw["parse_ok"].mean()),
            })

    # stacked fusion (B1 + M3 / M4)
    for (provider, model, config, split), d in list(merged.items()):
        if config not in ("M3f", "M3", "M4") or split != "development":
            continue
        fused = d.copy()
        X = np.c_[_logit(d["xgb_probability"]), _logit(d["prob"])]
        fused["fusion"] = np.nan
        for f in sorted(d["fold_index"].unique()):
            tr, te = (d["fold_index"] != f).to_numpy(), (d["fold_index"] == f).to_numpy()
            lr = LogisticRegression(C=1.0).fit(X[tr], d["label"][tr])
            fused.loc[te, "fusion"] = lr.predict_proba(X[te])[:, 1]
        tag = {"M3f": "F3f", "M3": "F3", "M4": "F4"}[config]
        pred_rows.append({"provider": provider, "model": model, "config": tag, "split": "development",
                          **pred_metrics(fused, "fusion", "development", args.n_bootstrap)})
        test = merged.get((provider, model, config, "heldout_test"))
        if test is not None and test["label"].nunique() == 2:
            lr = LogisticRegression(C=1.0).fit(X, d["label"])
            t = test.copy()
            t["fusion"] = lr.predict_proba(np.c_[_logit(t["xgb_probability"]), _logit(t["prob"])])[:, 1]
            pred_rows.append({"provider": provider, "model": model, "config": tag, "split": "heldout_test",
                              **pred_metrics(t, "fusion", "heldout_test", args.n_bootstrap),
                              "fusion_coef": json.dumps([round(float(v), 4) for v in lr.coef_[0]])})

    pred = pd.DataFrame(pred_rows)
    faith = pd.DataFrame(faith_rows)
    pred.to_csv(out_dir / "prediction_metrics.csv", index=False)
    faith.to_csv(out_dir / "faithfulness_metrics.csv", index=False)
    print(pred.drop(columns=[c for c in pred.columns if c.endswith("_ci")], errors="ignore").to_string(index=False))
    print(faith.to_string(index=False))
    if args.paper_dir is not None:
        write_latex(pred, faith, args.paper_dir, cases, args.b1_preset)


# ----------------------------------------------------------------------------- LaTeX
def _f(v, nd=3):
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _p(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return "$<$0.001" if v < 0.001 else f"{v:.3f}"


def _tex_escape(s: str) -> str:
    return str(s).replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write_latex(pred: pd.DataFrame, faith: pd.DataFrame, paper_dir: Path,
                cases: list[dict] | None = None, b1_preset: str = "") -> None:
    tdir = paper_dir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    macros = ["% Auto-generated by scripts/14_evaluate_mllm.py -- do not edit by hand."]

    def get(provider, config, split):
        r = pred[(pred["provider"] == provider) & (pred["config"] == config) & (pred["split"] == split)]
        return r.iloc[0] if len(r) else None

    order = [("-", "B1")] + [(p, c) for p in sorted(set(pred["provider"]) - {"-"}) for c in CONFIG_ORDER]
    lines = []
    for prov, cfg in order:
        dv, ts = get(prov, cfg, "development"), get(prov, cfg, "heldout_test")
        if dv is None and ts is None:
            continue
        name = "B1 XGBoost (reference)" if cfg == "B1" else f"{PROVIDER_WORD.get(prov, prov)}: {CONFIG_LABEL[cfg]}"
        dev_auc = f"{_f(dv['auroc_fold_mean'])} $\\pm$ {_f(dv['auroc_fold_sd'])}" if dv is not None else "--"
        dev_ap = f"{_f(dv['auprc_fold_mean'])} $\\pm$ {_f(dv['auprc_fold_sd'])}" if dv is not None else "--"
        if ts is not None:
            lo, hi = ts["auroc_ci"]
            t_auc = f"{_f(ts['auroc'])} ({_f(lo)}--{_f(hi)})"
            t_ap, t_br = _f(ts["auprc"]), _f(ts["brier"])
            t_p = "ref." if cfg == "B1" else _p(ts.get("delong_p_vs_b1"))
        else:
            t_auc = t_ap = t_br = t_p = "--"
        lines.append(f"{name} & {dev_auc} & {dev_ap} & {t_auc} & {t_ap} & {t_br} & {t_p} \\\\")
        pw = "BOne" if cfg == "B1" else PROVIDER_WORD.get(prov, prov) + CONFIG_WORD[cfg]
        if dv is not None:
            macros.append(f"\\def\\res{pw}DevAUROC{{{_f(dv['auroc_fold_mean'])} $\\pm$ {_f(dv['auroc_fold_sd'])}}}")
        for sp, row in (("", dv), ("Test", ts)):
            if row is not None and "anchor_rate" in row.index and not pd.isna(row["anchor_rate"]):
                macros.append(f"\\def\\res{pw}{sp}Anchor{{{100 * row['anchor_rate']:.1f}\\%}}")
        if ts is not None:
            lo, hi = ts["auroc_ci"]
            macros.append(f"\\def\\res{pw}TestAUROC{{{_f(ts['auroc'])} (95\\% CI {_f(lo)}--{_f(hi)})}}")
    (tdir / "table_prediction_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for split, fname in (("development", "table_faithfulness_rows.tex"), ("heldout_test", "table_faithfulness_test_rows.tex")):
        flines = []
        for _, r in faith[faith["split"] == split].assign(_o=lambda d: d["config"].map(CONFIG_ORDER.index)).sort_values(["provider", "_o"]).iterrows():
            name = f"{PROVIDER_WORD.get(r['provider'], r['provider'])}: {CONFIG_LABEL[r['config']]}"
            hal = f"{100 * r['hallucination_rate']:.1f} ({100 * r['hallucination_ci_low']:.1f}--{100 * r['hallucination_ci_high']:.1f})" if not math.isnan(r["hallucination_rate"]) else "--"
            con = f"{100 * r['contradiction_rate']:.1f} ({100 * r['contradiction_ci_low']:.1f}--{100 * r['contradiction_ci_high']:.1f})" if not math.isnan(r["contradiction_rate"]) else "--"
            flines.append(f"{name} & {_f(r['numbers_per_reply'], 1)} & {hal} & {100 * r['replies_with_hallucination']:.1f} & "
                          f"{100 * r['approximation_rate']:.1f} & {con} & {_f(r['evidence_jaccard'], 2)} \\\\")
            pw = PROVIDER_WORD.get(r["provider"], r["provider"]) + CONFIG_WORD[r["config"]] + ("" if split == "development" else "Test")
            if not math.isnan(r["hallucination_rate"]):
                macros.append(f"\\def\\res{pw}Halluc{{{100 * r['hallucination_rate']:.1f}\\%}}")
            if not math.isnan(r["contradiction_rate"]):
                macros.append(f"\\def\\res{pw}Contra{{{100 * r['contradiction_rate']:.1f}\\%}}")
        (tdir / fname).write_text("\n".join(flines) + "\n", encoding="utf-8")
    n_dev = int(pred[(pred["config"] == "B1") & (pred["split"] == "development")]["n"].iloc[0]) if len(pred) else 0
    macros.append(f"\\def\\resNDev{{{n_dev}}}")
    if cases:
        dev = [c for c in cases if c["split"] == "development"]
        tst = [c for c in cases if c["split"] == "heldout_test"]
        macros.append(f"\\def\\resNTest{{{len(tst)}}}")
        if dev:
            macros.append(f"\\def\\resPCRDev{{{sum(c['label'] for c in dev)} ({100 * np.mean([c['label'] for c in dev]):.1f}\\%)}}")
            macros.append(f"\\def\\resNImageFeatures{{{len(dev[0]['features'])}}}")
        if tst:
            macros.append(f"\\def\\resPCRTest{{{sum(c['label'] for c in tst)} ({100 * np.mean([c['label'] for c in tst]):.1f}\\%)}}")
    for prov, model in sorted({(a, b) for a, b in zip(pred["provider"], pred["model"]) if a != "-"}):
        macros.append(f"\\def\\res{PROVIDER_WORD.get(prov, prov)}Model{{\\texttt{{{_tex_escape(model)}}}}}")
    if b1_preset:
        macros.append(f"\\def\\resBOnePreset{{\\texttt{{{_tex_escape(b1_preset)}}}}}")
    (paper_dir / "results_macros.tex").write_text("\n".join(m for m in macros if m) + "\n", encoding="utf-8")
    print(f"LaTeX tables and macros written to {paper_dir}")


if __name__ == "__main__":
    main()
