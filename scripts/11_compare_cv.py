#!/usr/bin/env python3
"""
Step 11: Compare feature presets on development cross-validation only.

Reads ``xgboost_metrics.json`` from each sub-folder of ``--runs_dir`` (one
``03_train_xgboost.py`` run per preset, WITHOUT ``--run_final_test``) and
writes ``cv_comparison.csv`` sorted by mean fold AUROC. The held-out test set
is not used here; evaluate it once, for the single configuration chosen from
this table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs_dir", type=Path, required=True)
    p.add_argument("--candidates", nargs="*", default=None,
                   help="Presets eligible for best_preset.txt (default: all runs). All runs are still tabulated.")
    p.add_argument("--paper_dir", type=Path, default=None, help="Also write tables/table_cv_rows.tex here.")
    args = p.parse_args()
    rows = []
    for mpath in sorted(args.runs_dir.glob("*/xgboost_metrics.json")):
        m = json.loads(mpath.read_text(encoding="utf-8"))
        msd = m["cv_fold_metrics_mean_sd"]
        rows.append({
            "run": mpath.parent.name,
            "feature_mode": m["feature_mode"],
            "n_features": len(m["feature_columns"]),
            "auroc_mean": msd["roc_auc"]["mean"], "auroc_sd": msd["roc_auc"]["sd"],
            "auprc_mean": msd["average_precision"]["mean"], "auprc_sd": msd["average_precision"]["sd"],
            "auroc_pooled_oof": m["cv_oof_metrics_pooled"]["roc_auc"],
            "heldout_evaluated": m.get("heldout_test_metrics") is not None,
        })
    df = pd.DataFrame(rows).sort_values("auroc_mean", ascending=False)
    df.to_csv(args.runs_dir / "cv_comparison.csv", index=False, float_format="%.3f")
    eligible = df if not args.candidates else df[df["feature_mode"].isin(args.candidates)]
    if eligible.empty:
        raise SystemExit(f"None of the candidate presets {args.candidates} has a completed run in {args.runs_dir}")
    best = eligible.iloc[0]["feature_mode"]
    if args.paper_dir is not None:
        tdir = args.paper_dir / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        lines = []
        for _, r in df.iterrows():
            name = str(r["feature_mode"]).replace("_", "\\_")
            mark = "$^{*}$" if r["feature_mode"] == best else ""
            lines.append(f"\\texttt{{{name}}}{mark} & {int(r['n_features'])} & {r['auroc_mean']:.3f} $\\pm$ {r['auroc_sd']:.3f} & "
                         f"{r['auprc_mean']:.3f} $\\pm$ {r['auprc_sd']:.3f} & {r['auroc_pooled_oof']:.3f} \\\\")
        (tdir / "table_cv_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.runs_dir / "best_preset.txt").write_text(str(best), encoding="utf-8")
    print(f"Best preset by development mean fold AUROC: {best} (written to best_preset.txt)")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
