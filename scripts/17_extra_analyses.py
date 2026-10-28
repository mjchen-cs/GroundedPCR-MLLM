#!/usr/bin/env python3
"""
Step 17: secondary analyses of the MLLM study (no new API calls).

Uses the cached replies (step 13), cases.jsonl (step 12), the B1 SHAP export
(step 07) and the step-1 heterogeneity table, and writes
  <out>/tables/table_cohort_rows.tex       cohort characteristics (Table I)
  <out>/tables/table_subtype_rows.tex      subtype-stratified AUROC
  <out>/tables/table_errors_rows.tex       taxonomy of unsupported numbers + semantic flags
  <out>/tables/table_visual_rows.tex       verification of image-based claims
  <out>/figures/fig_calib_dca.pdf          calibration, probability granularity, decision curves
  <out>/figures/fig_shap_citation.pdf      B1 SHAP summary + evidence citation frequency
  <out>/analysis/*.csv                     all numbers behind the tables and figures

Usage:
  python scripts/17_extra_analyses.py --mllm_dir <results>/mllm --b1_dir <results>/b1 \
      --roi_dir <results>/ispy2_roi --out_dir <paper folder>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))
from mllm_prompts import build_messages, provided_values  # noqa: E402

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("ev14", Path(__file__).resolve().parent / "14_evaluate_mllm.py")
ev14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev14)

LABEL = {"zhipu": "GLM-4.5V", "siliconflow": "Qwen3-VL-32B", "mock": "Mock"}
SHORT = {"zhipu": "GLM", "siliconflow": "Qwen", "mock": "Mock"}
PROV: dict[str, tuple[str, str]] = {}   # provider -> (run-folder model name, label); filled in main()
COL: dict[str, str] = {}
INK, INK2, LINE = "#0b0b0b", "#52514e", "#e8e7e3"
CFGS = ["M1", "M2", "M3f", "M3", "M4"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def load(mllm_dir: Path):
    cases = [json.loads(l) for l in (mllm_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    base = pd.DataFrame([{k: c[k] for k in ("pid", "split", "fold_index", "label", "xgb_probability", "subtype")} for c in cases])
    runs = {}
    for prov, (model, _) in PROV.items():
        for cfg in CFGS:
            for split in ("development", "heldout_test"):
                f = mllm_dir / "runs" / f"{prov}__{model}" / f"{cfg}_{split}.csv"
                if f.exists():
                    runs[(prov, cfg, split)] = pd.read_csv(f)
    return cases, base, runs


def mean_prob(base, runs, prov, cfg, split):
    r = runs[(prov, cfg, split)]
    agg = r.groupby("pid").probability.mean().rename("prob").reset_index()
    return base[base.split == split].merge(agg, on="pid", how="left").fillna({"prob": 0.5})


def fused(base, runs, prov, cfg):
    """Stacked fusion of B1 and MLLM, out-of-fold in development and refit for the held-out set (as in step 14)."""
    d = mean_prob(base, runs, prov, cfg, "development")
    X = np.c_[logit(d.xgb_probability), logit(d.prob)]
    d["fusion"] = np.nan
    for f in sorted(d.fold_index.unique()):
        tr, te = (d.fold_index != f).to_numpy(), (d.fold_index == f).to_numpy()
        d.loc[te, "fusion"] = LogisticRegression(C=1.0).fit(X[tr], d.label[tr]).predict_proba(X[te])[:, 1]
    t = mean_prob(base, runs, prov, cfg, "heldout_test")
    lr = LogisticRegression(C=1.0).fit(X, d.label)
    t["fusion"] = lr.predict_proba(np.c_[logit(t.xgb_probability), logit(t.prob)])[:, 1]
    return d, t


SUB = [("HR+/HER2-", "HR+/HER2$-$"), ("HER2+", "HER2+"), ("Triple-negative", "TNBC")]


def cohort_table(cases, out):
    """Table I: cohort characteristics by split."""
    d = pd.DataFrame([{**c["clinical"], "split": c["split"], "label": c["label"], "subtype": c["subtype"],
                       "vol": c["features"].get("tumor_volume_mm3")} for c in cases])
    S = [d[d.split == "development"], d[d.split == "heldout_test"]]
    pct = lambda n, N: f"{n} ({100 * n / N:.1f}\\%)"
    rows = [("Patients ($n$)", [str(len(g)) for g in S]),
            ("Age (years)", [f"{g.age.mean():.1f} $\\pm$ {g.age.std():.1f}" for g in S]),
            ("pCR", [pct(int(g.label.sum()), len(g)) for g in S])]
    for sub, tex in (("HR+/HER2-", "HR+/HER2$-$"), ("HER2+", "HER2+"), ("Triple-negative", "Triple-negative")):
        rows.append((tex, [pct(int((g.subtype == sub).sum()), len(g)) for g in S]))
        rows.append(("\\quad pCR within subtype", [pct(int(g[g.subtype == sub].label.sum()), int((g.subtype == sub).sum())) for g in S]))
    if "MammaPrint" in d:
        rows.append(("MammaPrint = 1", [pct(int((g.MammaPrint == 1).sum()), len(g)) for g in S]))
    rows.append(("Tumor volume (cm$^3$)", [f"{g.vol.median() / 1000:.1f} [{g.vol.quantile(.25) / 1000:.1f}--{g.vol.quantile(.75) / 1000:.1f}]" for g in S]))
    (out / "tables" / "table_cohort_rows.tex").write_text("\n".join(f"{a} & {b[0]} & {b[1]}\\\\" for a, b in rows) + "\n", encoding="utf-8")


def subtype_table(base, runs, out):
    rows, tex = [], []
    specs = [("B1", None, None)] + [(f"{PROV[p][1]} {c}", p, c) for p in PROV for c in ("M1", "M2", "M3f")] + \
            [(f"{PROV[p][1]} B1+M3f fusion", p, "F3f") for p in PROV]
    for name, prov, cfg in specs:
        row = {"model": name}
        cells = []
        for sub, _ in SUB:
            for split in ("development", "heldout_test"):
                if prov is None:
                    d = base[base.split == split].assign(prob=lambda x: x.xgb_probability)
                elif cfg == "F3f":
                    dd, tt = fused(base, runs, prov, "M3f")
                    d = (dd if split == "development" else tt).assign(prob=lambda x: x.fusion)
                else:
                    d = mean_prob(base, runs, prov, cfg, split)
                g = d[d.subtype == sub]
                a = auc(g.label, g.prob)
                row[f"{sub}|{split}"] = a
                row[f"{sub}|{split}|n"] = len(g)
                cells.append(f"{a:.3f}")
        rows.append(row)
        tex.append(f"{name} & " + " & ".join(cells) + " \\\\")
    pd.DataFrame(rows).to_csv(out / "analysis" / "subtype_auroc.csv", index=False)
    (out / "tables" / "table_subtype_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- error taxonomy
CTX_POP = re.compile(r"rate|rates|typically|historical|historically|generally|studies|trial|literature|population|patients with|subtype|range", re.I)
CTX_PROB = re.compile(r"probabilit|chance|likelihood|estimated|estimate|around|approximately|about", re.I)
CTX_UNIT = re.compile(r"mm|cm|cc|min|/min|%\s*enh|ml|years", re.I)
VISUAL = {
    "washout": re.compile(r"wash-?out", re.I),
    "rim": re.compile(r"rim enhanc|rim-like|peripheral enhanc|ring enhanc|rim pattern", re.I),
    "heterogeneous": re.compile(r"heterogene", re.I),
    "irregular/spiculated": re.compile(r"irregular|spiculat", re.I),
    "large": re.compile(r"\b(large|bulky|sizable|sizeable|big)\b", re.I),
    "small": re.compile(r"\bsmall\b", re.I),
}
MP_LOW = re.compile(r"low[- ](genomic )?risk", re.I)


def classify_unsupported(report: str, provided: dict, prompt_text: str, own_p: float):
    """Return list of (category, token) for unsupported report numbers (same rules as step 14)."""
    rep = ev14.THOUSANDS_RE.sub("", str(report or ""))
    for k in sorted(set(provided) | {"core15", "safe_rim", "ring1", "ring2"}, key=len, reverse=True):
        rep = rep.replace(k, " ")
    pool = list(provided.values()) + [own_p]
    for m in ev14.NUM_RE.finditer(prompt_text):
        try:
            pool.append(float(m.group(0).rstrip("% ").strip()))
        except ValueError:
            pass
    prov_vals = [v for v in provided.values() if not np.isnan(v)]
    out = []
    for m in ev14.NUM_RE.finditer(rep):
        s = m.group(0).strip()
        try:
            x = float(s.rstrip("%").strip())
        except ValueError:
            continue
        if ev14._supported(x, pool, s.endswith("%"), s) or ev14._approximates(x, prov_vals):
            continue
        ctx = rep[max(0, m.start() - 70): m.end() + 25]
        if CTX_UNIT.search(rep[m.end(): m.end() + 6]):
            cat = "measurement"
        elif CTX_POP.search(ctx):
            cat = "population statistic"
        elif CTX_PROB.search(ctx):
            cat = "probability restated"
        else:
            cat = "other"
        out.append((cat, s))
    return out


def error_tables(cases, runs, het, out):
    by = {c["pid"]: c for c in cases}
    rows, ex = [], []
    for prov in PROV:
        for cfg in CFGS:
            r = pd.concat([runs[(prov, cfg, s)] for s in ("development", "heldout_test") if (prov, cfg, s) in runs])
            cats = {"population statistic": 0, "probability restated": 0, "measurement": 0, "other": 0}
            n_vis = {k: 0 for k in VISUAL}
            mp0 = mp0_low = 0
            for _, x in r.iterrows():
                c = by[x.pid]
                prompt = build_messages(cfg, c, None)[-1]["content"]
                prompt = prompt if isinstance(prompt, str) else ""
                for cat, tok in classify_unsupported(x.report, provided_values(cfg, c), prompt, x.probability):
                    cats[cat] += 1
                    if len(ex) < 400:
                        ex.append({"provider": prov, "config": cfg, "pid": x.pid, "category": cat, "token": tok,
                                   "report": str(x.report)[:400]})
                for k, rx in VISUAL.items():
                    n_vis[k] += bool(rx.search(str(x.report)))
                if c["clinical"].get("MammaPrint") == 0:
                    mp0 += 1
                    mp0_low += bool(MP_LOW.search(str(x.report)))
            rows.append({"provider": prov, "config": cfg, "n_replies": len(r), **cats,
                         **{f"vis_{k}": v / len(r) for k, v in n_vis.items()},
                         "mp0_low_risk": mp0_low / max(1, mp0)})
    er = pd.DataFrame(rows)
    er.to_csv(out / "analysis" / "error_taxonomy.csv", index=False)
    pd.DataFrame(ex).to_csv(out / "analysis" / "unsupported_examples.csv", index=False)
    tex = []
    for _, x in er.iterrows():
        tot = x[["population statistic", "probability restated", "measurement", "other"]].sum()
        tex.append(f"{PROV[x.provider][1]} {x.config} & {int(tot)} & {int(x['population statistic'])} & {int(x['probability restated'])} & "
                   f"{int(x['measurement'])} & {int(x['other'])} & {100 * x.mp0_low_risk:.1f} \\\\")
    (out / "tables" / "table_errors_rows.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    # verification of image-type claims against measured quantities (all cases, both splits)
    vrows, vtex = [], []
    checks = [("washout", "het_washout_fraction"), ("heterogeneous", "het_pe_cv"), ("large", "tumor_volume_mm3")]
    for prov in PROV:
        for cfg in ("M1", "M2", "M3f"):
            r = pd.concat([runs[(prov, cfg, s)] for s in ("development", "heldout_test")])
            r = r[r.repeat == 0].merge(het, on="pid", how="left")
            row = {"provider": prov, "config": cfg}
            cells = []
            for word, col in checks:
                flag = r.report.fillna("").apply(lambda t: bool(VISUAL[word].search(t)))
                ok = r[col].notna()
                a = auc(flag[ok].astype(int), r.loc[ok, col]) if flag[ok].nunique() == 2 else float("nan")
                row[f"{word}_rate"] = float(flag.mean()); row[f"{word}_auc"] = a
                cells += [f"{100 * flag.mean():.1f}", "--" if np.isnan(a) else f"{a:.2f}"]
            vrows.append(row)
            vtex.append(f"{PROV[prov][1]} {cfg} & " + " & ".join(cells) + " \\\\")
    pd.DataFrame(vrows).to_csv(out / "analysis" / "visual_claims.csv", index=False)
    (out / "tables" / "table_visual_rows.tex").write_text("\n".join(vtex) + "\n", encoding="utf-8")
    return er, pd.DataFrame(vrows)


# ----------------------------------------------------------------------------- figures
def net_benefit(y, p, thr):
    y = np.asarray(y); n = len(y)
    out = []
    for t in thr:
        pos = p >= t
        tp = np.sum(pos & (y == 1)); fp = np.sum(pos & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def fig_calib(base, runs, out):
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.55), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    dev = base[base.split == "development"]
    series = [("B1 (OOF)", dev.xgb_probability.to_numpy(), "#52514e", "-", "o")]
    for prov in PROV:
        d = mean_prob(base, runs, prov, "M3f", "development")
        series.append((f"{PROV[prov][1]} M3f", d.prob.to_numpy(), COL[prov], "-", "s" if list(PROV).index(prov) == 1 else "o"))
        d1 = mean_prob(base, runs, prov, "M1", "development")
        series.append((f"{PROV[prov][1]} M1", d1.prob.to_numpy(), COL[prov], ":", "s" if list(PROV).index(prov) == 1 else "o"))
    y = dev.label.to_numpy()
    a = ax[0]
    a.plot([0, 1], [0, 1], color=LINE, lw=1)
    calib_rows = []
    for name, p, c, ls, mk in series:
        bins = np.unique(np.quantile(p, np.linspace(0, 1, 6)))
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, len(bins) - 2)
        mp = [p[idx == i].mean() for i in range(len(bins) - 1) if np.any(idx == i)]
        oy = [y[idx == i].mean() for i in range(len(bins) - 1) if np.any(idx == i)]
        a.plot(mp, oy, ls, color=c, marker=mk, ms=3.5, lw=1.2, label=name)
        calib_rows.append({"model": name, "mean_pred": np.mean(p), "observed": y.mean(), "n_unique": len(np.unique(np.round(p, 3)))})
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xlabel("Mean predicted probability"); a.set_ylabel("Observed pCR rate")
    a.set_title("(a) Calibration (development)", loc="left", fontsize=7.5, fontweight="bold")
    a.legend(frameon=False, fontsize=5.6, loc="upper left")
    # granularity
    a = ax[1]
    labels, data, cols = [], [], []
    for name, p, c, ls, mk in series:
        labels.append(name.replace(" (OOF)", "")); data.append(p); cols.append(c)
    parts = a.violinplot(data, orientation="horizontal", showextrema=False, widths=0.8)
    for b, c in zip(parts["bodies"], cols):
        b.set_facecolor(c); b.set_alpha(0.35); b.set_edgecolor(c)
    for i, (p, c) in enumerate(zip(data, cols), 1):
        a.text(0.99, i + 0.32, f"{len(np.unique(np.round(p, 3)))} distinct", va="center", ha="right", fontsize=5.4, color=INK2)
    a.axvline(y.mean(), color=INK2, lw=0.8, ls="--")
    a.set_yticks(range(1, len(labels) + 1)); a.set_yticklabels(labels, fontsize=6); a.set_xlim(0, 1)
    a.set_xlabel("Predicted probability (dashed: pCR prevalence)")
    a.set_title("(b) Predicted probabilities (dev.)", loc="left", fontsize=7.5, fontweight="bold")
    # DCA
    a = ax[2]
    thr = np.linspace(0.05, 0.6, 56)
    a.plot(thr, net_benefit(y, np.ones_like(y, float), thr), color=LINE, lw=1.2, label="Treat all")
    a.axhline(0, color=INK2, lw=0.8, label="Treat none")
    dca = {"threshold": thr}
    for name, p, c, ls, mk in series:
        if "M1" in name:
            continue
        nb = net_benefit(y, p, thr); dca[name] = nb
        a.plot(thr, nb, ls, color=c, lw=1.3, label=name)
    for prov in PROV:
        d, _ = fused(base, runs, prov, "M3f")
        nb = net_benefit(d.label.to_numpy(), d.fusion.to_numpy(), thr); dca[f"{PROV[prov][1]} fusion"] = nb
        a.plot(thr, nb, "--", color=COL[prov], lw=1.0, label=f"B1+{PROV[prov][1]} M3f")
    a.set_ylim(-0.05, 0.45); a.set_xlim(0.05, 0.6); a.set_xlabel("Threshold probability"); a.set_ylabel("Net benefit")
    a.set_title("(c) Decision curves (dev.)", loc="left", fontsize=7.5, fontweight="bold")
    a.legend(frameon=False, fontsize=5.2, loc="upper right", bbox_to_anchor=(1.0, 1.0), handlelength=1.8)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig_calib_dca.pdf"); fig.savefig(out / "figures" / "fig_calib_dca.png", dpi=180)
    pd.DataFrame(calib_rows).to_csv(out / "analysis" / "calibration_summary.csv", index=False)
    pd.DataFrame(dca).to_csv(out / "analysis" / "decision_curves.csv", index=False)


PRETTY = {"clinical_mp": "MammaPrint", "clinical_her2": "HER2", "core15_kep": "core $k_{ep}$", "clinical_hr": "HR",
          "tumor_volume_mm3": "tumor volume", "safe_rim_to_core15_volume_ratio": "rim/core ratio",
          "core15_Ktrans": "core $K^{trans}$", "safe_rim_kep": "rim $k_{ep}$", "safe_rim_Ktrans": "rim $K^{trans}$",
          "clinical_age": "age", "core15_volume_mm3": "core volume", "safe_rim_volume_mm3": "rim volume",
          "clinical_triple_neg": "TNBC"}
KEY2COL = {"MammaPrint": "clinical_mp", "HER2": "clinical_her2", "HR": "clinical_hr", "age": "clinical_age", "TNBC": "clinical_triple_neg"}


def fig_shap_citation(runs, b1_dir, out):
    sv = pd.read_csv(b1_dir / "shap" / "shap_values.csv")
    xi = pd.read_csv(b1_dir / "final_development_model_inputs.csv")
    feats = [c for c in sv.columns if c not in ("pid", "base_value")]
    order = sorted(feats, key=lambda c: sv[c].abs().mean())  # ascending for barh
    fig = plt.figure(figsize=(7.16, 3.0))
    a = fig.add_axes([0.13, 0.15, 0.30, 0.74])
    rng = np.random.default_rng(0)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bv", ["#86b6ef", "#0d366b"])
    for i, c in enumerate(order):
        v = xi.set_index("pid").loc[sv.pid, c].to_numpy(float)
        r = pd.Series(v).rank(pct=True).to_numpy()
        a.scatter(sv[c], i + rng.uniform(-0.28, 0.28, len(sv)), c=r, cmap=cmap, s=2.5, lw=0, alpha=0.8, rasterized=True)
    a.axvline(0, color=INK2, lw=0.6)
    a.set_yticks(range(len(order))); a.set_yticklabels([PRETTY.get(c, c) for c in order], fontsize=6.2)
    a.set_xlabel("TreeSHAP value (log-odds; > 0 toward pCR)")
    a.set_title("(a) B1 attributions (development)", loc="left", fontsize=7.5, fontweight="bold", pad=10)
    a.text(0.0, 1.0, "dot colour: feature value (light low, dark high)", transform=a.transAxes, ha="left", va="bottom", fontsize=5.4, color=INK2)
    # citation heatmap
    b = fig.add_axes([0.58, 0.15, 0.33, 0.74])
    cols, mat = [], []
    for prov in PROV:
        for cfg in CFGS:
            r = pd.concat([runs[(prov, cfg, s)] for s in ("development", "heldout_test")])
            cnt = {c: 0 for c in order}
            for e in r.evidence.fillna("[]"):
                try:
                    keys = {d.get("feature") for d in json.loads(e) if isinstance(d, dict)}
                except json.JSONDecodeError:
                    keys = set()
                for k in keys:
                    col = KEY2COL.get(k, k)
                    if col in cnt:
                        cnt[col] += 1
            mat.append([cnt[c] / len(r) for c in order]); cols.append(f"{SHORT.get(prov, prov)} {cfg}")
    M = np.array(mat).T
    im = b.imshow(M, aspect="auto", cmap=LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#2a78d6", "#0d366b"]), vmin=0, vmax=1, origin="lower")
    b.set_yticks(range(len(order))); b.set_yticklabels([PRETTY.get(c, c) for c in order], fontsize=6.2)
    b.set_xticks(range(len(cols))); b.set_xticklabels(cols, rotation=60, ha="right", fontsize=5.8)
    if len(PROV) > 1:
        b.axvline(len(CFGS) - 0.5, color="white", lw=2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            b.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=4.6, color="white" if M[i, j] > 0.55 else INK)
    b.set_title("(b) Replies citing each input (%)", loc="left", fontsize=7.5, fontweight="bold", pad=10)
    cb = fig.colorbar(im, ax=b, fraction=0.04, pad=0.02, ticks=[0, 0.5, 1]); cb.ax.set_yticklabels(["0", "50", "100"]); cb.ax.tick_params(labelsize=5.5)
    fig.savefig(out / "figures" / "fig_shap_citation.pdf"); fig.savefig(out / "figures" / "fig_shap_citation.png", dpi=180)
    imp = sv[order].abs().mean().to_numpy()
    rho = {cols[j]: float(pd.Series(M[:, j]).corr(pd.Series(imp), method="spearman")) for j in range(len(cols))}
    pd.DataFrame(M, index=order, columns=cols).to_csv(out / "analysis" / "citation_frequency.csv")
    pd.Series(rho, name="spearman_citation_vs_mean_abs_shap").to_csv(out / "analysis" / "citation_vs_shap_spearman.csv")
    return rho


def stability(runs, out):
    rows = []
    for (prov, cfg, split), r in runs.items():
        w = r.pivot_table(index="pid", columns="repeat", values="probability")
        c = r.pivot_table(index="pid", columns="repeat", values="conclusion", aggfunc="first")
        if w.shape[1] < 2:
            continue
        rows.append({"provider": prov, "config": cfg, "split": split,
                     "identical_probability": float((w[0] == w[1]).mean()),
                     "same_conclusion": float((c[0] == c[1]).mean()),
                     "mean_abs_diff": float((w[0] - w[1]).abs().mean())})
    s = pd.DataFrame(rows)
    s.to_csv(out / "analysis" / "repeat_stability.csv", index=False)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mllm_dir", type=Path, required=True)
    ap.add_argument("--b1_dir", type=Path, required=True)
    ap.add_argument("--roi_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--providers", nargs="+", default=["zhipu", "siliconflow"])
    a = ap.parse_args()
    for i, prov in enumerate(a.providers):
        d = sorted((a.mllm_dir / "runs").glob(f"{prov}__*"))
        if not d:
            raise SystemExit(f"no runs for provider {prov} in {a.mllm_dir / 'runs'}")
        PROV[prov] = (d[0].name.split("__", 1)[1], LABEL.get(prov, prov))
        COL[prov] = ["#2a78d6", "#eb6834"][i % 2]
    for d in ("tables", "figures", "analysis"):
        (a.out_dir / d).mkdir(parents=True, exist_ok=True)
    cases, base, runs = load(a.mllm_dir)
    het = pd.read_csv(a.roi_dir / "dynamics_feature_summary.csv")
    het["pid"] = het["pid"].astype(str)
    feats = pd.DataFrame([{"pid": c["pid"], "tumor_volume_mm3": c["features"].get("tumor_volume_mm3")} for c in cases])
    het = het[["pid", "het_washout_fraction", "het_pe_cv"]].merge(feats, on="pid", how="right")
    cohort_table(cases, a.out_dir)
    sub = subtype_table(base, runs, a.out_dir)
    er, vis = error_tables(cases, runs, het, a.out_dir)
    fig_calib(base, runs, a.out_dir)
    rho = fig_shap_citation(runs, a.b1_dir, a.out_dir)
    st = stability(runs, a.out_dir)
    pd.set_option("display.width", 200)
    print(sub.round(3).to_string()); print(er.round(3).to_string()); print(vis.round(3).to_string())
    print(rho); print(st.round(3).to_string())


if __name__ == "__main__":
    main()
