#!/usr/bin/env python3
"""
Step 7: Global SHAP analysis of the locked final model (Sec. III-D, Fig. 5a).

Exact TreeSHAP values are computed with XGBoost's built-in
``pred_contribs=True`` (no extra dependency) on the final development model
inputs written by step 3 (``final_development_model_inputs.csv``; imputed
features, i.e. exactly what the booster saw). The ``shap`` package, if
installed, is used only for the beeswarm plot.

Outputs: ``shap_values.csv``, ``shap_importance.csv`` (mean |SHAP| ranking,
plus the Spearman correlation between each feature value and its SHAP value
to indicate direction), ``fig5a_shap_summary.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from locked_model import load_locked_model

DISPLAY_NAMES = {
    "core15_Ktrans": "Core K$^{trans}$",
    "core15_kep": "Core k$_{ep}$",
    "safe_rim_Ktrans": "Safe-rim K$^{trans}$",
    "safe_rim_kep": "Safe-rim k$_{ep}$",
    "tumor_volume_mm3": "Total tumor volume",
    "core15_volume_mm3": "Core15 volume",
    "safe_rim_volume_mm3": "Safe-rim volume",
    "safe_rim_to_core15_volume_ratio": "Safe-rim/core volume ratio",
    "clinical_age": "Age",
    "clinical_hr": "HR status",
    "clinical_her2": "HER2 status",
    "clinical_triple_neg": "TNBC",
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Global SHAP summary for the locked model.")
    p.add_argument("--model_dir", type=Path, required=True)
    p.add_argument("--model_inputs_csv", type=Path, required=True,
                   help="final_development_model_inputs.csv from step 3 (or any table with the model input columns).")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--pid_col", type=str, default="pid")
    p.add_argument("--max_display", type=int, default=12)
    return p


def _beeswarm_matplotlib(shap_vals: np.ndarray, X: pd.DataFrame, order: list[int], names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(order) + 1.2), constrained_layout=True)
    for row, j in enumerate(reversed(order)):
        v = X.iloc[:, j].to_numpy(dtype=float)
        lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        c = np.clip((v - lo) / (hi - lo if hi > lo else 1.0), 0, 1)
        y = row + rng.uniform(-0.3, 0.3, size=v.size)
        sc = ax.scatter(shap_vals[:, j], y, c=c, cmap="coolwarm", s=8, alpha=0.8, vmin=0, vmax=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[j] for j in reversed(order)])
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("SHAP value (impact on log-odds of pCR)")
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 1], aspect=30)
    cb.ax.set_yticklabels(["Low", "High"])
    cb.set_label("Feature value")
    fig.savefig(path, dpi=300)


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    locked = load_locked_model(args.model_dir)
    input_cols = list(locked["card"]["model_input_columns"])

    df = pd.read_csv(args.model_inputs_csv)
    missing = [c for c in input_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Model input columns missing from {args.model_inputs_csv}: {missing}")
    X = df[input_cols].astype(float)

    booster = locked["xgb"].get_booster()
    contribs = booster.predict(xgb.DMatrix(X, feature_names=input_cols), pred_contribs=True)
    shap_vals, base = contribs[:, :-1], contribs[:, -1]

    out = pd.DataFrame(shap_vals, columns=input_cols)
    if args.pid_col in df.columns:
        out.insert(0, args.pid_col, df[args.pid_col].astype(str).to_numpy())
    out["base_value"] = base
    out.to_csv(args.output_dir / "shap_values.csv", index=False)

    from scipy.stats import spearmanr

    imp_rows = []
    for j, c in enumerate(input_cols):
        rho = spearmanr(X[c], shap_vals[:, j]).statistic if X[c].nunique() > 1 else np.nan
        imp_rows.append({"feature": c, "mean_abs_shap": float(np.mean(np.abs(shap_vals[:, j]))),
                         "spearman_value_vs_shap": float(rho)})
    imp = pd.DataFrame(imp_rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.insert(0, "rank", np.arange(1, len(imp) + 1))
    imp.to_csv(args.output_dir / "shap_importance.csv", index=False)

    order = [input_cols.index(f) for f in imp["feature"].head(args.max_display)]
    names = [DISPLAY_NAMES.get(c, c) for c in input_cols]
    fig_path = args.output_dir / "fig5a_shap_summary.png"
    try:
        import shap  # type: ignore
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shap.summary_plot(shap_vals, X.rename(columns=dict(zip(input_cols, names))),
                          max_display=args.max_display, show=False)
        plt.gcf().savefig(fig_path, dpi=300, bbox_inches="tight")
    except ImportError:
        _beeswarm_matplotlib(shap_vals, X, order, names, fig_path)

    print(imp.to_string(index=False))


if __name__ == "__main__":
    main()
