#!/usr/bin/env python3
"""
Step 9: Feature-group ablation (Table III).

Runs ``03_train_xgboost.py --run_final_test`` once per preset (each with its
own fold-wise PINN refits, fold-fitted imputation and OOF threshold), and, if
``--visual_embedding_csv`` is given, the two visual-branch variants
(main + PCA-8 and main + PLS-8). Results are collected into
``table_iii_ablation.csv``.

    python scripts/09_run_ablations.py \
        --feature_csv  out_roi/dynamics_feature_summary.csv \
        --volume_csv   out_roi/volume_summary.csv \
        --metadata_csv /path/to/metadata.csv \
        --tcc_csv      out_roi/tcc_long.csv \
        --output_dir   out_ablation \
        [--visual_embedding_csv out_vis/ispy2_vis.csv] \
        [-- <extra args passed to 03_train_xgboost.py>]

Reported per configuration: number of model inputs, development OOF AUROC /
AUPRC (mean across folds, and pooled OOF), held-out test AUROC / AUPRC.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from xgboost_utils import FEATURE_PRESETS, TABLE_III_PRESETS

ROW_LABELS = {
    "ktrans_only": "Ktrans only (Mcore + Mrim)",
    "kep_only": "kep only (Mcore + Mrim)",
    "core15_physics_only": "Core15 physics only (Mcore)",
    "safe_rim_physics_only": "Safe rim physics only (Mrim)",
    "tofts_only": "All physics (Mcore + Mrim)",
    "volume_only": "Morphological volumes only",
    "tofts_basic_volume": "All physics + Basic Volume",
    "tofts_full_volume": "All physics + Full Volume",
    "tofts_deltas_ratios": "All physics + Deltas & Ratios",
    "main": "Proposed PI-ML (Clinical + Vol + Phys)",
    "main_plus_simple_dynamics": "PI-ML + 8 Simple Dynamics",
    "main+visual_pca": "PI-ML + Visual Branch (PCA, 8-D)",
    "main+visual_pls": "PI-ML + Visual Branch (PLS, 8-D)",
}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Table III feature-group ablations.")
    p.add_argument("--feature_csv", type=Path, required=True)
    p.add_argument("--volume_csv", type=Path, required=True)
    p.add_argument("--metadata_csv", type=Path, required=True)
    p.add_argument("--tcc_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--presets", type=str, nargs="*", default=None, help="Subset of presets (default: all Table III rows).")
    p.add_argument("--visual_embedding_csv", type=Path, default=None)
    p.add_argument("--visual_n_components", type=int, default=8)
    p.add_argument("--skip_existing", action="store_true", help="Reuse runs whose xgboost_metrics.json already exists.")
    return p


def main() -> None:
    if "--" in sys.argv:
        cut = sys.argv.index("--")
        own, passthrough = sys.argv[1:cut], sys.argv[cut + 1:]
    else:
        own, passthrough = sys.argv[1:], []
    args = build_argparser().parse_args(own)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent / "03_train_xgboost.py"

    runs: list[tuple[str, list[str]]] = []
    for preset in (args.presets or TABLE_III_PRESETS):
        if preset not in FEATURE_PRESETS:
            raise ValueError(f"Unknown preset {preset}")
        runs.append((preset, ["--feature_mode", preset]))
    if args.visual_embedding_csv is not None:
        for red in ("pca", "pls"):
            runs.append((f"main+visual_{red}", ["--feature_mode", "main", "--visual_embedding_csv", str(args.visual_embedding_csv),
                                                "--visual_reduction", red, "--visual_n_components", str(args.visual_n_components)]))

    rows = []
    for name, extra in runs:
        out = args.output_dir / name.replace("+", "_plus_")
        metrics_path = out / "xgboost_metrics.json"
        if not (args.skip_existing and metrics_path.exists()):
            cmd = [sys.executable, str(script),
                   "--feature_csv", str(args.feature_csv), "--volume_csv", str(args.volume_csv),
                   "--metadata_csv", str(args.metadata_csv), "--tcc_csv", str(args.tcc_csv),
                   "--output_dir", str(out), "--run_final_test", "--no_save_final_model", "--n_bootstrap", "0",
                   *extra, *passthrough]
            print(f"[ablation] {name}", flush=True)
            subprocess.run(cmd, check=True)
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        msd = m["cv_fold_metrics_mean_sd"]
        test = m.get("heldout_test_metrics") or {}
        n_feat = len(m["feature_columns"]) + (args.visual_n_components if m.get("visual_reduction") else 0)
        rows.append({
            "configuration": name,
            "feature_representation": ROW_LABELS.get(name, name),
            "n_features": n_feat,
            "dev_oof_auroc_fold_mean": msd["roc_auc"]["mean"],
            "dev_oof_auroc_fold_sd": msd["roc_auc"]["sd"],
            "dev_oof_auprc_fold_mean": msd["average_precision"]["mean"],
            "dev_oof_auprc_fold_sd": msd["average_precision"]["sd"],
            "dev_oof_auroc_pooled": m["cv_oof_metrics_pooled"]["roc_auc"],
            "dev_oof_auprc_pooled": m["cv_oof_metrics_pooled"]["average_precision"],
            "heldout_auroc": test.get("roc_auc"),
            "heldout_auprc": test.get("average_precision"),
        })
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "table_iii_ablation.csv", index=False, float_format="%.3f")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
