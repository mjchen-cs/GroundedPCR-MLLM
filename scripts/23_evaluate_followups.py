#!/usr/bin/env python3
"""
Step 23 (offline): evaluate the counterfactual test (step 21) and the verify-and-revise
loop (step 22). Writes tables/table_cf_rows.tex, tables/table_revise_rows.tex and
analysis/counterfactual.csv, analysis/revise.csv into --out_dir.

Usage: python scripts/23_evaluate_followups.py --mllm_dir results/mllm --out_dir paper
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
from mllm_prompts import build_messages, pick_examples  # noqa: E402

_s = importlib.util.spec_from_file_location("rv22", _HERE / "22_verify_revise.py")
rv22 = importlib.util.module_from_spec(_s); _s.loader.exec_module(rv22)

LABEL = {"zhipu": "GLM-4.5V", "siliconflow": "Qwen3-VL-32B", "mock": "Mock"}
CONFLICT_RE = re.compile(r"inconsisten|conflict|contradict|discrepan|does not match|doesn't match|mismatch|despite the (reference )?model|"
                      r"at odds|disagree", re.I)


def _dirs(ev: str, shap: dict) -> tuple[int, int]:
    """(# directional items with |SHAP|>=0.02, # that contradict the given SHAP signs)."""
    try:
        items = json.loads(ev) if isinstance(ev, str) else []
    except json.JSONDecodeError:
        items = []
    n = c = 0
    for e in items:
        if not isinstance(e, dict):
            continue
        sv, d = shap.get(str(e.get("feature", ""))), e.get("direction")
        if sv is not None and abs(sv) >= 0.02 and d in ("supports_pcr", "against_pcr"):
            n += 1
            c += int((d == "supports_pcr") != (sv > 0))
    return n, c


def eval_cf(mllm_dir: Path, by: dict) -> pd.DataFrame:
    rows = []
    for rd in sorted((mllm_dir / "runs_cf").glob("*__*")):
        prov = rd.name.split("__", 1)[0]
        orig_dir = mllm_dir / "runs" / rd.name
        for f in sorted(rd.glob("M3-*_*.csv")):
            variant = "flip_all" if "flip_all" in f.stem else "flip_prob"
            split = f.stem.split(f"{variant}_", 1)[1]
            d = pd.read_csv(f)
            o = pd.read_csv(orig_dir / f"M3_{split}.csv")
            o = o[o.repeat == 0].set_index("pid")
            d["label"] = d.pid.map(lambda p: by[p]["label"])
            d["p_orig_reply"] = d.pid.map(o.probability)
            ok = d.probability.notna()
            p = d.probability.astype(float)
            n_dir = n_c_shown = n_c_true = 0
            for _, r in d.iterrows():
                true_shap = by[r.pid]["shap"]
                shown = {k: -v for k, v in true_shap.items()} if variant == "flip_all" else true_shap
                a, b = _dirs(r.evidence, shown)
                _, c = _dirs(r.evidence, true_shap)
                n_dir += a; n_c_shown += b; n_c_true += c
            rows.append({
                "provider": prov, "variant": variant, "split": split, "n": len(d), "parse_rate": float(ok.mean()),
                "copy_shown": float((abs(p - d.b1_shown) <= 0.01)[ok].mean()),
                "copy_original": float((abs(p - d.b1_original) <= 0.01)[ok].mean()),
                "closer_to_shown": float((abs(p - d.b1_shown) < abs(p - d.b1_original))[ok].mean()),
                "auroc_cf": float(roc_auc_score(d.label[ok], p[ok])),
                "auroc_orig_reply": float(roc_auc_score(d.label[d.p_orig_reply.notna()], d.p_orig_reply.dropna())),
                "auroc_b1": float(roc_auc_score(d.label, d.b1_original)),
                "contra_vs_shown_shap": n_c_shown / max(1, n_dir),
                "contra_vs_true_shap": n_c_true / max(1, n_dir),
                "conflict_mentioned": float(d.report.fillna("").apply(lambda t: bool(CONFLICT_RE.search(t))).mean()),
            })
    return pd.DataFrame(rows)


def eval_revise(mllm_dir: Path, by: dict, cases: list) -> pd.DataFrame:
    dev_cases = [c for c in cases if c["split"] == "development"]
    rows = []
    for rd in sorted((mllm_dir / "runs_revise").glob("*__*")):
        prov = rd.name.split("__", 1)[0]
        for audit_f in sorted(rd.glob("*_audit.csv")):
            cfg, split = audit_f.stem.replace("_audit", "").split("_", 1)
            audit = pd.read_csv(audit_f)
            rev_f = rd / f"{cfg}_{split}.csv"
            try:
                rev = pd.read_csv(rev_f)
            except (FileNotFoundError, pd.errors.EmptyDataError):  # no failing replies (empty file)
                rev = pd.DataFrame()
            n_fail = int((audit.n_violations > 0).sum())
            still, v_before, v_after, dp = 0, 0, 0, []
            for _, r in rev.iterrows():
                case = by[r.pid]
                ex = pick_examples(case, dev_cases, 6, 42) if cfg == "M4" else None
                pt = build_messages(cfg, case, None, ex)[-1]["content"]
                after = rv22.violations(cfg, case, r, pt if isinstance(pt, str) else "") if r.parse_ok == 1 else ["unparsed"]
                v_before += len(json.loads(r.violations)); v_after += len(after); still += int(bool(after))
                if r.parse_ok == 1 and r.orig_probability == r.orig_probability:
                    dp.append(abs(float(r.probability) - float(r.orig_probability)))
            tok = float(rev.prompt_tokens.fillna(0).sum() + rev.completion_tokens.fillna(0).sum()) if len(rev) else 0.0
            rows.append({"provider": prov, "config": cfg, "split": split, "n_replies": len(audit), "n_failed": n_fail,
                         "n_revised": len(rev), "n_still_failing": still,
                         "violations_before": v_before, "violations_after": v_after,
                         "parse_rate_revised": float(rev.parse_ok.mean()) if len(rev) else float("nan"),
                         "mean_abs_prob_change": float(np.mean(dp)) if dp else float("nan"), "tokens": tok})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mllm_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    a = ap.parse_args()
    for d in ("tables", "analysis"):
        (a.out_dir / d).mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in (a.mllm_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by = {c["pid"]: c for c in cases}
    pd.set_option("display.width", 220)
    if (a.mllm_dir / "runs_cf").exists():
        cf = eval_cf(a.mllm_dir, by)
        cf.to_csv(a.out_dir / "analysis" / "counterfactual.csv", index=False)
        tex = []
        for _, r in cf.sort_values(["provider", "variant"], ascending=[False, False]).iterrows():
            v = "$1-p$ only" if r.variant == "flip_prob" else "$1-p$, SHAP negated"
            tex.append(f"{LABEL.get(r.provider, r.provider)} & {v} & {100 * r.copy_shown:.1f} & {100 * r.copy_original:.1f} & "
                       f"{100 * r.closer_to_shown:.1f} & {r.auroc_orig_reply:.3f} $\\to$ {r.auroc_cf:.3f} & "
                       f"{100 * r.contra_vs_shown_shap:.1f} & {100 * r.conflict_mentioned:.1f} \\\\")
        (a.out_dir / "tables" / "table_cf_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        print(cf.round(3).to_string())
    if (a.mllm_dir / "runs_revise").exists():
        rv = eval_revise(a.mllm_dir, by, cases)
        rv.to_csv(a.out_dir / "analysis" / "revise.csv", index=False)
        g = rv.groupby(["provider", "config"], sort=False)[["n_replies", "n_failed", "n_revised", "n_still_failing",
                                                            "violations_before", "violations_after", "tokens"]].sum().reset_index()
        order = {"M1": 0, "M2": 1, "M3f": 2, "M3": 3, "M4": 4}
        g = g.assign(_o=g.config.map(order)).sort_values(["provider", "_o"], ascending=[False, True])
        tex = []
        for _, r in g.iterrows():
            tex.append(f"{LABEL.get(r.provider, r.provider)} {r.config} & {int(r.n_replies)} & {int(r.n_failed)} "
                       f"({100 * r.n_failed / r.n_replies:.1f}) & {int(r.n_still_failing)} ({100 * r.n_still_failing / r.n_replies:.1f}) & "
                       f"{int(r.violations_before)} $\\to$ {int(r.violations_after)} & {r.tokens / 1000:.0f}k \\\\")
        (a.out_dir / "tables" / "table_revise_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
        print(rv.round(3).to_string())


if __name__ == "__main__":
    main()
