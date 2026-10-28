#!/usr/bin/env python3
"""
Step 19: package derived results as a zip (e.g. a GitHub Release asset or Zenodo record), so that
steps 14-18 -- every table, figure and number of the paper -- can be regenerated
without the images and without API calls:

    OUT_DIR=<unzipped bundle> bash scripts/run_study.sh evaluate

The bundle mirrors the layout of OUT_DIR and contains only derived, de-identified
data: per-case model inputs (clinical covariates, Tofts-PINN and volume features),
B1 predictions and SHAP values, the parsed MLLM replies (runs/*.csv, including the
raw reply text), evaluation tables, and the ROI summaries. Image panels are left out
except the single case drawn in Fig. 1 (--fig1_pid; use --no_images to drop it).
The on-disk API cache is left out unless --include_cache is given.

Check the TCIA / BreastDCEDL data-use terms before publishing the bundle.

Usage:
  python scripts/19_make_release_bundle.py --out_dir results --bundle release/GroundedPCR-MLLM-results.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

INCLUDE = [
    "cv/best_preset.txt", "cv/cv_comparison.csv", "cv/*/xgboost_metrics.json",
    "b1/xgboost_predictions.csv", "b1/xgboost_metrics.json", "b1/xgboost_cv_oof_inputs.csv",
    "b1/xgboost_cv_oof_shap.csv", "b1/heldout_test_inputs.csv", "b1/heldout_test_shap.csv",
    "b1/final_development_model_inputs.csv", "b1/shap/shap_values.csv", "b1/shap/shap_importance.csv",
    "b1/tofts_pinn_training_summary.json", "b1/final_model/model_card.json",
    "sens/*/xgboost_predictions.csv", "sens/*/xgboost_metrics.json",
    "ispy2_roi/dynamics_feature_summary.csv", "ispy2_roi/volume_summary.csv", "ispy2_roi/tcc_long.csv",
    "ispy2_roi/qc_report.csv",
    "mllm/cases.jsonl", "mllm/export_report.json", "mllm/runs/*/*.csv", "mllm/evaluation/*.csv",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out_dir", type=Path, required=True, help="OUT_DIR of run_study (results folder)")
    ap.add_argument("--bundle", type=Path, required=True, help="zip file to write")
    ap.add_argument("--fig1_pid", default="ACRIN-6698-102212")
    ap.add_argument("--no_images", action="store_true")
    ap.add_argument("--all_images", action="store_true", help="include all 2x2 panels (needed to re-query the MLLMs without the raw images)")
    ap.add_argument("--include_cache", action="store_true", help="also ship mllm/cache (allows re-running step 13 offline)")
    a = ap.parse_args()
    files: list[Path] = []
    for pat in INCLUDE:
        files += sorted(a.out_dir.glob(pat))
    if a.all_images:
        files += sorted((a.out_dir / "mllm" / "images").glob("*.png"))
    elif not a.no_images:
        files += sorted((a.out_dir / "mllm" / "images").glob(f"{a.fig1_pid}.png"))
    if a.include_cache:
        files += sorted((a.out_dir / "mllm" / "cache").glob("*.json"))
    a.bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(a.out_dir).as_posix())
        z.writestr("README.txt",
                   "Derived results of the I-SPY2 MLLM study. Unzip and run\n"
                   "  OUT_DIR=<this folder> bash scripts/run_study.sh evaluate\n"
                   "from the code repository to regenerate all tables and figures.\n"
                   "Source data: BreastDCEDL / TCIA I-SPY2 (see their data-use terms).\n")
    size = a.bundle.stat().st_size / 1e6
    print(f"wrote {a.bundle} ({len(files)} files, {size:.1f} MB)")
    missing = [p for p in INCLUDE if not list(a.out_dir.glob(p))]
    if missing:
        print("not found (skipped):", ", ".join(missing))


if __name__ == "__main__":
    main()
