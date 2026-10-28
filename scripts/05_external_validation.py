#!/usr/bin/env python3
"""
Step 5: External validation with the locked final model (Sec. II-A, II-F; Table II).

Run step 1 (``01_extract_roi_tcc.py``) on the external cohort (I-SPY 1, Duke),
then apply the model saved by ``03_train_xgboost.py --run_final_test``:

    python scripts/05_external_validation.py \
        --model_dir     /path/to/output_xgb/final_model \
        --cohort_name   ISPY1 \
        --feature_csv   /path/to/ispy1_roi/dynamics_feature_summary.csv \
        --volume_csv    /path/to/ispy1_roi/volume_summary.csv \
        --tcc_csv       /path/to/ispy1_roi/tcc_long.csv \
        --metadata_csv  /path/to/ispy1/metadata.csv \
        --output_dir    /path/to/external/ISPY1

Every labelled case of the external metadata is scored (the ``test`` column
is ignored). Nothing is refitted: curves are aligned to the training time grid,
the frozen PINN yields K_trans / k_ep, missing values are filled with the
*development* medians, and the locked OOF threshold is used for hard labels.

Outputs: ``predictions_<cohort>.csv``, ``features_<cohort>.csv``,
``metrics_<cohort>.json`` (point estimates + bootstrap 95% CIs).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from locked_model import build_cohort_feature_table, load_locked_model, predict_with_locked_model
from metrics_utils import bootstrap_metric_ci, compute_binary_metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="External validation with the locked final model.")
    p.add_argument("--model_dir", type=Path, required=True, help="<output_xgb>/final_model from step 3.")
    p.add_argument("--cohort_name", type=str, required=True, help="Label used in output file names, e.g. ISPY1 / Duke.")
    p.add_argument("--feature_csv", type=Path, required=True, help="dynamics_feature_summary.csv of the external cohort.")
    p.add_argument("--volume_csv", type=Path, required=True, help="volume_summary.csv of the external cohort.")
    p.add_argument("--tcc_csv", type=Path, required=True, help="tcc_long.csv of the external cohort.")
    p.add_argument("--metadata_csv", type=Path, required=True, help="metadata.csv of the external cohort.")
    p.add_argument("--visual_embedding_csv", type=Path, default=None, help="Only if the locked model uses the visual branch.")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--pid_col", type=str, default="pid")
    p.add_argument("--label_col", type=str, default="pCR")
    p.add_argument("--min_core_voxels", type=int, default=None, help="Default: value stored with the locked model.")
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("external_validation")

    locked = load_locked_model(args.model_dir, device=args.device)
    card = locked["card"]
    min_core = args.min_core_voxels if args.min_core_voxels is not None else int(card.get("min_core_voxels", 10))

    df = build_cohort_feature_table(
        locked,
        feature_csv=args.feature_csv,
        volume_csv=args.volume_csv,
        metadata_csv=args.metadata_csv,
        tcc=args.tcc_csv,
        pid_col=args.pid_col,
        label_col=args.label_col,
        test_col="__no_test_column__",
        visual_embedding_csv=args.visual_embedding_csv,
        min_core_voxels=min_core,
        logger=logger,
    )
    df.to_csv(args.output_dir / f"features_{args.cohort_name}.csv", index=False)

    pred = predict_with_locked_model(locked, df, pid_col=args.pid_col, label_col=args.label_col,
                                     split_group=f"external_{args.cohort_name}")
    pred.to_csv(args.output_dir / f"predictions_{args.cohort_name}.csv", index=False)

    labeled = pred["y_true"].notna()
    thr = float(card["decision_threshold"])
    payload = {
        "cohort": args.cohort_name,
        "n_scored": int(len(pred)),
        "n_labeled": int(labeled.sum()),
        "decision_threshold": thr,
        "metrics": compute_binary_metrics(pred.loc[labeled, "y_true"], pred.loc[labeled, "y_prob"], thr),
        "bootstrap_ci": (
            bootstrap_metric_ci(pred.loc[labeled, "y_true"], pred.loc[labeled, "y_prob"], thr,
                                n_bootstrap=args.n_bootstrap, random_state=args.random_state)
            if args.n_bootstrap and labeled.sum() > 1 else None
        ),
        "model_dir": str(args.model_dir),
        "refit_on_external_cohort": False,
    }
    with open(args.output_dir / f"metrics_{args.cohort_name}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    m = payload["metrics"]
    logger.info("%s | n=%d | AUROC=%.4f | AUPRC=%.4f | F1=%.4f", args.cohort_name, m["n_samples"],
                m["roc_auc"], m["average_precision"], m["f1"])


if __name__ == "__main__":
    main()
