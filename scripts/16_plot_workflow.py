#!/usr/bin/env python3
"""Step 16: Fig. 1 workflow figure, drawn with one real case.

Usage:
  python scripts/16_plot_workflow.py <cases.jsonl> <images/PID.png> <ispy2_roi/tcc_long.csv> \
      <runs/<provider>__<model>/M3_<split>.csv> <PID> <out_stem> [--eval_dir <mllm>/evaluation] \
      [--model_label GLM-4.5V] [--models_line "GLM-4.5V or Qwen3-VL-32B"] [--report_excerpt "..."]

Every value in the figure is read from the case record (step 12), the B1 SHAP values,
the cached MLLM reply (step 13) and, for the bottom band, the evaluation tables (step 14).
"""
import argparse
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ap = argparse.ArgumentParser(description="Fig. 1 workflow figure")
for a_ in ("cases", "panel", "tcc", "reply", "pid", "out"):
    ap.add_argument(a_)
ap.add_argument("--eval_dir", default=None, help="step-14 evaluation folder; fills the bottom summary band")
ap.add_argument("--model_label", default="GLM-4.5V", help="name of the model whose reply is shown")
ap.add_argument("--models_line", default="GLM-4.5V or Qwen3-VL-32B")
ap.add_argument("--report_excerpt", default=None, help="excerpt of the report to print (default: start of the report)")
ap.add_argument("--n_repeats", type=int, default=2)
A = ap.parse_args()
CASES, PANEL, TCC, REPLY, PID, OUT = A.cases, A.panel, A.tcc, A.reply, A.pid, A.out
import os as _os; _os.makedirs(_os.path.dirname(_os.path.abspath(OUT)), exist_ok=True)
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here.parent / "src"))
from mllm_prompts import build_messages, provided_values  # noqa: E402
_spec = importlib.util.spec_from_file_location("ev14", _here / "14_evaluate_mllm.py")
ev14 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ev14)
case = next(json.loads(l) for l in open(CASES, encoding="utf-8") if json.loads(l)["pid"] == PID)
rep = pd.read_csv(REPLY)
rep = rep[(rep.pid == PID)].sort_values("repeat").iloc[0]
ev = json.loads(rep.evidence)
tcc = pd.read_csv(TCC)
tcc = tcc[tcc.pid == PID]

INK, INK2, MUTED, LINE, SURF, PANELBG = "#0b0b0b", "#52514e", "#8a8984", "#d6d5d0", "#fcfcfb", "#f4f3f0"
BLUE, RED, ORANGE, GOOD = "#2a78d6", "#e34948", "#eb6834", "#0ca30c"
MONO = "DejaVu Sans Mono"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK})

W, H = 7.16, 4.25
fig = plt.figure(figsize=(W, H))
fig.patch.set_facecolor("white")
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, W); bg.set_ylim(0, H); bg.axis("off")

def box(x, y, w, h, fc=PANELBG, ec=LINE, lw=0.8, r=0.05, z=0):
    bg.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))

def header(x, y, w, n, title, sub):
    bg.add_patch(Circle((x + 0.11, y), 0.085, fc=INK, ec="none", zorder=3))
    bg.text(x + 0.11, y - 0.004, str(n), ha="center", va="center", color="white", fontsize=6.5, fontweight="bold", zorder=4)
    bg.text(x + 0.24, y + 0.035, title, ha="left", va="center", fontsize=6.6, fontweight="bold", color=INK)
    bg.text(x + 0.24, y - 0.095, sub, ha="left", va="center", fontsize=5.3, color=INK2)

def arrow(x0, y0, x1, y1, color=INK2, ls="-"):
    bg.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.9, color=color, ls=ls, zorder=5))

# column geometry (inches)
TOP, BOT = 3.72, 0.62
cols = [(0.08, 1.20), (1.42, 1.50), (3.06, 1.40), (4.60, 1.28), (6.02, 1.06)]
titles = [("Pretreatment data", "T0 DCE-MRI + clinical"),
          ("Physics model B1", "hospital side, validated"),
          ("Grounded prompt", "configuration M3"),
          ("MLLM reply", "commercial API, zero-shot"),
          ("Faithfulness", "automatic check")]
for i, ((x, w), (t, s)) in enumerate(zip(cols, titles)):
    box(x, BOT, w, TOP - BOT, fc=PANELBG if i != 3 else "#fdf3ee", ec=LINE if i != 3 else "#f2c3ad")
    header(x + 0.02, TOP + 0.2, w, i + 1, t, s)
for (x, w), (x2, _) in zip(cols[:-1], cols[1:]):
    arrow(x + w + 0.005, 2.2, x2 - 0.005, 2.2)

# ---------- 1. data
x, w = cols[0]
img = plt.imread(PANEL)
ax = fig.add_axes([(x + 0.07) / W, 2.42 / H, (w - 0.14) / W, (w - 0.14) / H])
ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_color(LINE)
bg.text(x + w / 2, 2.36, "pre | early post-contrast\nlate | early + tumor mask", fontsize=5.0, color=INK2, va="top", ha="center")
c = {k: (float("nan") if case["clinical"].get(k) is None else case["clinical"][k]) for k in ("age", "HR", "HER2", "TNBC", "MammaPrint")}
_pn = lambda v: "n/a" if v != v else ("negative" if v == 0 else "positive")
rows = [("Age", f"{c['age']:.0f} y"), ("HR", _pn(c["HR"])),
        ("HER2", _pn(c["HER2"])), ("Subtype", "triple-negative" if c["TNBC"] == 1 else "non-TNBC"),
        ("MammaPrint", f"{c['MammaPrint']:.0f}")]
yy = 1.96
bg.text(x + 0.07, yy, "Clinical covariates", fontsize=6, fontweight="bold", color=INK); yy -= 0.15
for k, v in rows:
    bg.text(x + 0.09, yy, k, fontsize=5.4, color=INK2); bg.text(x + w - 0.07, yy, v, fontsize=5.4, color=INK, ha="right"); yy -= 0.125
bg.plot([x + 0.07, x + w - 0.07], [yy + 0.04, yy + 0.04], color=LINE, lw=0.6)
_all = [json.loads(l) for l in open(CASES, encoding="utf-8") if l.strip()]
_nd = sum(c_["split"] == "development" for c_ in _all); _nt = len(_all) - _nd
bg.text(x + 0.07, yy - 0.08, f"I-SPY2 (BreastDCEDL)\n{_nd} development\n{_nt} held-out\nshown: {'held-out' if case['split'] == 'heldout_test' else 'development'} case,\n{'pCR' if case['label'] == 1 else 'no pCR'} at surgery", fontsize=5.0, color=INK2, va="top")

# ---------- 2. B1
x, w = cols[1]
ax = fig.add_axes([(x + 0.30) / W, 2.98 / H, (w - 0.40) / W, 0.52 / H])
names = {"core15": ("core (inner 15%)", BLUE, "-"), "safe_rim": ("rim", INK2, "--")}
for roi, (lab, col, ls) in names.items():
    d = tcc[tcc.roi_type == roi].sort_values("timepoint_idx")
    ax.plot(d.timepoint_idx, d.normalized_delta_s_over_s0, ls, color=col, lw=1.2, marker="o", ms=2.2, label=lab)
ax.set_xlabel("DCE phase", fontsize=5.6, labelpad=1); ax.set_ylabel("ΔS / S$_0$", fontsize=5.6, labelpad=1)
ax.tick_params(labelsize=5.2, length=2, pad=1); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=4.9, loc="lower right", handlelength=1.6)
bg.text(x + 0.07, 3.60, "Regional enhancement", fontsize=6, fontweight="bold", color=INK, va="top")
f = case["features"]
yy = 2.58
bg.text(x + 0.07, yy, "Tofts-PINN", fontsize=6, fontweight="bold", color=INK)
bg.text(x + 1.06, yy, "core", fontsize=5.3, color=INK2, ha="right"); bg.text(x + 1.42, yy, "rim", fontsize=5.3, color=INK2, ha="right")
for k in ("Ktrans", "kep"):
    yy -= 0.13
    lab = "$K^{trans}$ (min$^{-1}$)" if k == "Ktrans" else "$k_{ep}$ (min$^{-1}$)"
    bg.text(x + 0.09, yy, lab, fontsize=5.4, color=INK2)
    bg.text(x + 1.06, yy, f"{f['core15_' + k]:.3f}", fontsize=5.3, family=MONO, ha="right")
    bg.text(x + 1.42, yy, f"{f['safe_rim_' + k]:.3f}", fontsize=5.3, family=MONO, ha="right")
yy -= 0.15
bg.text(x + 0.09, yy, "Tumor volume", fontsize=5.4, color=INK2); bg.text(x + 1.42, yy, f"{f['tumor_volume_mm3'] / 1000:.1f} cm$^3$", fontsize=5.4, ha="right")
yy -= 0.19
N_IN = len(case["features"]) + sum(v is not None for v in case["clinical"].values())
bg.text(x + 0.07, yy, f"XGBoost, {N_IN} inputs", fontsize=5.8, fontweight="bold", color=INK)
bg.text(x + w - 0.07, yy, f"$\\hat p_{{B1}}$ = {case['xgb_probability']:.2f}", fontsize=6.0, ha="right", color=BLUE, fontweight="bold")
# SHAP bars
sh = sorted(case["shap"].items(), key=lambda kv: -abs(kv[1]))[:6]
pretty = {"MammaPrint": "MammaPrint", "HR": "HR", "HER2": "HER2", "core15_kep": "core $k_{ep}$", "safe_rim_Ktrans": "rim $K^{trans}$",
          "tumor_volume_mm3": "tumor volume", "age": "age", "safe_rim_to_core15_volume_ratio": "rim/core ratio",
          "core15_volume_mm3": "core volume", "safe_rim_kep": "rim $k_{ep}$", "TNBC": "TNBC", "core15_Ktrans": "core $K^{trans}$"}
ax = fig.add_axes([(x + 0.62) / W, 0.86 / H, (w - 0.72) / W, 0.88 / H])
vals = [v for _, v in sh][::-1]; labs = [pretty.get(k, k) for k, _ in sh][::-1]
ax.barh(range(len(vals)), vals, color=[BLUE if v > 0 else RED for v in vals], height=0.62, edgecolor="white", lw=0.6)
ax.axvline(0, color=INK2, lw=0.6)
ax.set_yticks(range(len(vals))); ax.set_yticklabels(labs, fontsize=5.0)
ax.tick_params(axis="x", labelsize=5.0, length=2, pad=1); ax.tick_params(axis="y", length=0, pad=2)
ax.set_xlabel("TreeSHAP, log-odds", fontsize=5.0, labelpad=1); ax.spines[["top", "right", "left"]].set_visible(False)
ax.set_xlim(-0.5, 0.35); ax.set_xticks([-0.4, 0, 0.2]); ax.set_xticklabels(["−0.4", "0", "0.2"])
bg.text(x + 0.62 + (w - 0.72) * 0.25, 1.80, "← against", color=RED, fontsize=5.0, ha="center")
bg.text(x + 0.62 + (w - 0.72) * 0.80, 1.80, "pCR →", color=BLUE, fontsize=5.0, ha="center")

# ---------- 3. prompt
x, w = cols[2]
pr = [("SYSTEM", "never invent measurements;\nreturn one JSON object"),
      ("IMAGE", "2×2 DCE panel (step 1)"),
      ("CLINICAL", f"age = {c['age']:.0f}, HR = {c['HR']:.0f}\nHER2 = {c['HER2']:.0f}, TNBC = {c['TNBC']:.0f}, MP = {c['MammaPrint']:.0f}"),
      (f"FEATURES ({len(f)}, with units)", f"core15_kep = {f['core15_kep']:.3g} /min\nsafe_rim_Ktrans = {f['safe_rim_Ktrans']:.3g} /min\ntumor_volume = {f['tumor_volume_mm3']:.0f} mm³ …"),
      ("B1 OUTPUT", f"probability = {case['xgb_probability']:.3g}\n" + "\n".join(f"SHAP {k_} {v_:+.3f}" for k_, v_ in sorted(case['shap'].items(), key=lambda kv: -abs(kv[1]))[:2]) + " …"),
      ("REQUIRED OUTPUT", "probability, conclusion,\n2–6 evidence items,\nreport ≤ 120 words")]
yy = 3.60
for k, v in pr:
    bg.text(x + 0.07, yy, k, fontsize=5.1, fontweight="bold", color=BLUE if k.startswith(("FEATURES", "B1")) else INK2, va="top")
    bg.text(x + 0.07, yy - 0.105, v, fontsize=4.6, family=MONO, color=INK, va="top", linespacing=1.25)
    yy -= 0.13 + 0.092 * (v.count("\n") + 1) + 0.05
bg.text(x + 0.07, 0.70, "Other rungs: M1 text, M2 + image,\nM3f + features, M4 + 6 cases", fontsize=4.8, color=INK2, va="bottom")

# ---------- 4. reply
x, w = cols[3]
bg.text(x + 0.07, 3.60, A.models_line, fontsize=5.4, color=ORANGE, fontweight="bold", va="top")
bg.text(x + 0.07, 3.46, f"T = 0.01, {A.n_repeats} repeats, cached", fontsize=5.0, color=INK2, va="top")
bg.text(x + 0.07, 3.30, f"{A.model_label} reply:", fontsize=5.0, color=INK2, va="top")
js = [f'probability  {rep.probability:g}', f'conclusion   {rep.conclusion}', 'evidence']
short = {"supports_pcr": "supports", "against_pcr": "against", "neutral": "neutral"}
for e in ev[:5]:
    v = "null" if e.get("value") is None else f"{e['value']:g}"
    fn = {"tumor_volume_mm3": "tumor_volume"}.get(e["feature"], e["feature"])
    js.append(f' {fn[:12]:<12}{v:>6} {short.get(e["direction"], e["direction"])[:7]}')
yy = 3.14
for line in js:
    bg.text(x + 0.07, yy, line, fontsize=4.35, family=MONO, color=INK, va="top"); yy -= 0.108
_ex = A.report_excerpt if A.report_excerpt else str(rep.report)[:220].rsplit(" ", 1)[0] + " …"
rtxt = '"' + "\n".join(textwrap.wrap(_ex, 31)[:8]) + '"'
bg.text(x + 0.07, yy - 0.06, "Report (excerpt)", fontsize=5.1, fontweight="bold", color=INK2, va="top")
bg.text(x + 0.07, yy - 0.18, rtxt, fontsize=4.7, color=INK, va="top", style="italic", linespacing=1.25)

bg.text(x + 0.07, 0.70, f"Probability equals\n$\\hat p_{{B1}}$ = {case['xgb_probability']:.3f} (anchoring)", fontsize=4.9, color=ORANGE, va="bottom")

# ---------- 5. gate
x, w = cols[4]
shap = case["shap"]
dir_items = [e for e in ev if e["feature"] in shap and abs(shap[e["feature"]]) >= 0.02 and e["direction"] in ("supports_pcr", "against_pcr")]
agree = sum((e["direction"] == "supports_pcr") == (shap[e["feature"]] > 0) for e in dir_items)
_cfg = "M3"
_prompt = build_messages(_cfg, case, None)[-1]["content"]
FR = ev14.faithfulness_row(rep, case, _prompt if isinstance(_prompt, str) else "", provided_values(_cfg, case))
n_ok = FR["n_numbers"] - FR["n_unsupported"]
_cons = FR["consistent"]
checks = [("Numbers", f"{n_ok} of {FR['n_numbers']} found\nin the prompt"),
          ("Directions", f"{agree} of {len(dir_items)} match the\nSHAP sign"),
          ("Conclusion", ("consistent with\n" if _cons in (1, None) else "inconsistent with\n") + f"p = {rep.probability:g}"),
          ("Schema", "valid JSON" if int(rep.parse_ok) == 1 else "not parsed")]
_pass = [FR["n_unsupported"] == 0, agree == len(dir_items), _cons in (1, None), int(rep.parse_ok) == 1]
yy = 3.56
for k, v in checks:
    _ok = _pass[[c_[0] for c_ in checks].index(k)]
    bg.add_patch(Circle((x + 0.13, yy - 0.06), 0.055, fc=GOOD if _ok else RED, ec="none", zorder=3))
    bg.text(x + 0.13, yy - 0.065, "✓" if _ok else "✗", color="white", fontsize=5.5, ha="center", va="center", zorder=4, fontweight="bold")
    bg.text(x + 0.23, yy, k, fontsize=5.4, fontweight="bold", color=INK, va="top")
    bg.text(x + 0.23, yy - 0.115, v, fontsize=4.9, color=INK2, va="top", linespacing=1.2)
    yy -= 0.27 + 0.085 * v.count("\n")
box(x + 0.07, yy - 0.40, w - 0.14, 0.36, fc="#e9f6e9", ec=GOOD, lw=0.8, r=0.04, z=2)
bg.text(x + w / 2, yy - 0.22, "Pass: report shown\nfor clinician review", fontsize=5.0, ha="center", va="center", color=INK, zorder=3)
yy -= 0.58
box(x + 0.07, yy - 0.36, w - 0.14, 0.36, fc="white", ec=RED, lw=0.8, r=0.04, z=2)
bg.text(x + w / 2, yy - 0.18, "Fail: report withheld,\nB1 output only", fontsize=5.0, ha="center", va="center", color=INK, zorder=3)
_n_all = None
if A.eval_dir:
    _fm = pd.read_csv(Path(A.eval_dir) / "faithfulness_metrics.csv")
    _n_all = int(_fm.n_replies.sum())
bg.text(x + 0.07, 0.70, (f"All {_n_all:,} replies\n" if _n_all else "Every reply\n") + "audited, no clinician\nrating needed", fontsize=4.8, color=INK2, va="bottom")

# bottom band: key findings, read from the step-14 tables
LABEL = {"zhipu": "GLM-4.5V", "siliconflow": "Qwen3-VL-32B", "mock": "mock model"}
if A.eval_dir:
    pm = pd.read_csv(Path(A.eval_dir) / "prediction_metrics.csv")
    fm = pd.read_csv(Path(A.eval_dir) / "faithfulness_metrics.csv")
    T = pm[pm.split == "heldout_test"]
    b1 = T[T.config == "B1"].auroc.iloc[0]
    noref = T[T.config.isin(["M1", "M2", "M3f"])].auroc
    anc = T[T.config == "M3"].anchor_rate
    F = fm[fm.split == "heldout_test"]
    c1, c3 = F[F.config == "M1"].contradiction_rate, F[F.config == "M3"].contradiction_rate
    h1 = F[F.config == "M1"].sort_values("hallucination_rate").iloc[-1]
    h3 = F[(F.config == "M3") & (F.provider == h1.provider)].hallucination_rate.iloc[0]
    def rng_(s_, pct=True):
        lo, hi = (100 * s_.min(), 100 * s_.max()) if pct else (s_.min(), s_.max())
        f_ = "{:.1f}" if pct else "{:.2f}"
        return (f_.format(lo) if f_.format(lo) == f_.format(hi) else f_.format(lo) + "–" + f_.format(hi)) + ("%" if pct else "")
    box(0.08, 0.08, W - 0.16, 0.42, fc="white", ec=LINE, lw=0.8)
    bg.text(0.20, 0.37, f"Prediction (held-out, n = {int(T[T.config == 'B1'].n.iloc[0])}):", fontsize=5.6, fontweight="bold", color=INK, va="center")
    bg.text(2.05, 0.37, f"B1 AUROC {b1:.3f}; MLLMs without B1 output {rng_(noref, False)}; given the B1 probability, {rng_(anc)} of replies copy it.", fontsize=5.4, color=INK2, va="center")
    bg.text(0.20, 0.20, "Explanation (held-out, M1 → M3):", fontsize=5.6, fontweight="bold", color=INK, va="center")
    bg.text(2.05, 0.20, f"directions contradicting SHAP {rng_(c1)} → {rng_(c3)}; numbers absent from the prompt {100 * h1.hallucination_rate:.1f}% → {100 * h3:.1f}% ({LABEL.get(h1.provider, h1.provider)}).", fontsize=5.4, color=INK2, va="center")

fig.savefig(OUT + ".pdf"); fig.savefig(OUT + ".png", dpi=220)
