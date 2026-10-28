#!/usr/bin/env python3
"""
Step 10: Sensitivity of held-out test AUROC to input perturbations (Fig. 5b).

The locked final model (PINN + imputer + XGBoost + threshold, from step 3) is
kept fixed; only the *inputs* of the held-out test cases (``test in {1, 2}``)
are perturbed, and AUROC with a bootstrap 95% CI is recomputed:

- Temporal sampling
    * ``drop_last_phase``          remove the last post-contrast phase
    * ``first_k_post:K``           keep the pre-contrast + first K post-contrast phases
    * ``every_other_post``         keep pre-contrast + every second post-contrast phase
    * ``dt:SECONDS``               assumed phase interval when no acquisition times exist
- ROI definition (``--roi_variant NAME=DIR``): step-1 outputs recomputed with
  other ROI parameters, e.g. ``01_extract_roi_tcc.py --core_fraction 0.10`` or
  ``--uncertain_boundary_fraction 0.05``; DIR must contain ``tcc_long.csv``,
  ``volume_summary.csv`` and ``dynamics_feature_summary.csv``.
- AIF timing: ``aif_shift:SECONDS`` shifts the curve time axis relative to the
  population AIF (bolus-arrival mismatch of +/- SECONDS).
- Gaussian noise: ``snr:VALUE`` adds N(0, (S0/SNR)^2) to each ROI mean
  signal, ``--noise_repeats`` independent draws (median AUROC and median CI
  bounds are reported).

Example:

    python scripts/10_sensitivity_analysis.py \
        --model_dir out_xgb/final_model --roi_dir out_roi \
        --metadata_csv /path/to/metadata.csv --output_dir out_sensitivity \
        --roi_variant core10=out_roi_core10 --roi_variant core20=out_roi_core20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from locked_model import build_cohort_feature_table, load_locked_model, predict_with_locked_model
from metrics_utils import bootstrap_metric_ci
from tofts_pinn_utils import HELDOUT_TEST_VALUES

DEFAULT_SCENARIOS = [
    "drop_last_phase", "first_k_post:3", "every_other_post", "dt:80", "dt:100",
    "aif_shift:-30", "aif_shift:-15", "aif_shift:15", "aif_shift:30",
    "snr:50", "snr:20", "snr:10",
]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fig. 5b sensitivity analysis on the held-out test set.")
    p.add_argument("--model_dir", type=Path, required=True)
    p.add_argument("--roi_dir", type=Path, required=True, help="Step-1 output dir (tcc_long / volume / dynamics CSVs).")
    p.add_argument("--metadata_csv", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--scenarios", type=str, nargs="*", default=DEFAULT_SCENARIOS)
    p.add_argument("--roi_variant", action="append", default=[], help="NAME=DIR of step-1 outputs with other ROI params.")
    p.add_argument("--noise_repeats", type=int, default=20)
    p.add_argument("--n_bootstrap", type=int, default=10000)
    p.add_argument("--pid_col", type=str, default="pid")
    p.add_argument("--label_col", type=str, default="pCR")
    p.add_argument("--test_col", type=str, default="test")
    p.add_argument("--random_state", type=int, default=42)
    return p


def _post_rows(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df.get("is_pre", 0), errors="coerce").fillna(0).astype(int) == 0


def perturb_tcc(tcc: pd.DataFrame, kind: str, value: str | None, rng: np.random.Generator) -> pd.DataFrame:
    tcc = tcc.copy()
    keep = pd.Series(True, index=tcc.index)
    if kind in {"drop_last_phase", "first_k_post", "every_other_post"}:
        for _, g in tcc.groupby(["pid", "roi_type"]):
            post = g.loc[_post_rows(g)].sort_values("timepoint_idx")
            order = post["timepoint_idx"].tolist()
            if kind == "drop_last_phase":
                drop = order[-1:] if len(order) > 1 else []
            elif kind == "first_k_post":
                drop = order[int(value):]
            else:
                drop = order[1::2]
            keep.loc[post.index[post["timepoint_idx"].isin(drop)]] = False
        return tcc.loc[keep].copy()
    if kind == "snr":
        snr = float(value)
        base = pd.to_numeric(tcc["baseline_mean_signal"], errors="coerce") if "baseline_mean_signal" in tcc else None
        if base is None or base.isna().all():
            base = tcc.groupby(["pid", "roi_type"])["mean_signal"].transform("first")
        tcc["mean_signal"] = pd.to_numeric(tcc["mean_signal"], errors="coerce") + rng.normal(0.0, 1.0, len(tcc)) * base / snr
        return tcc
    raise ValueError(kind)


def evaluate(locked, roi_dir: Path, meta_csv: Path, args, tcc_override: pd.DataFrame | None = None,
             config_overrides: dict | None = None, seed: int = 0) -> dict:
    tcc = tcc_override if tcc_override is not None else pd.read_csv(roi_dir / "tcc_long.csv")
    df = build_cohort_feature_table(
        locked, feature_csv=roi_dir / "dynamics_feature_summary.csv", volume_csv=roi_dir / "volume_summary.csv",
        metadata_csv=meta_csv, tcc=tcc, pid_col=args.pid_col, label_col=args.label_col,
        test_col=args.test_col, min_core_voxels=int(locked["card"].get("min_core_voxels", 10)),
        config_overrides=config_overrides,
    )
    pred = predict_with_locked_model(locked, df, pid_col=args.pid_col, label_col=args.label_col, split_group="heldout_test")
    pred = pred.loc[pred["y_true"].notna()]
    ci = bootstrap_metric_ci(pred["y_true"], pred["y_prob"], float(locked["card"]["decision_threshold"]),
                             n_bootstrap=args.n_bootstrap, random_state=args.random_state + seed)["roc_auc"]
    return {"n": int(len(pred)), "auroc": ci["point"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}


def main() -> None:
    args = build_argparser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    locked = load_locked_model(args.model_dir)

    meta = pd.read_csv(args.metadata_csv)
    meta = meta.loc[pd.to_numeric(meta[args.test_col], errors="coerce").isin(list(HELDOUT_TEST_VALUES))]
    meta_csv = args.output_dir / "_heldout_metadata.csv"
    meta.to_csv(meta_csv, index=False)
    base_tcc = pd.read_csv(args.roi_dir / "tcc_long.csv")
    base_tcc["pid"] = base_tcc["pid"].astype(str)
    base_tcc = base_tcc.loc[base_tcc["pid"].isin(set(meta["pid"].astype(str)))].copy()

    rows = [{"group": "Baseline", "scenario": "unperturbed", **evaluate(locked, args.roi_dir, meta_csv, args, base_tcc)}]
    print(f"baseline AUROC={rows[0]['auroc']:.3f}", flush=True)

    for sc in args.scenarios:
        kind, _, value = sc.partition(":")
        if kind in {"drop_last_phase", "first_k_post", "every_other_post"}:
            r = evaluate(locked, args.roi_dir, meta_csv, args, perturb_tcc(base_tcc, kind, value, np.random.default_rng(0)))
            group = "Temporal sampling"
        elif kind == "dt":
            r = evaluate(locked, args.roi_dir, meta_csv, args, base_tcc,
                         config_overrides={"default_temporal_resolution_seconds": float(value)})
            group = "Temporal sampling"
        elif kind == "aif_shift":
            r = evaluate(locked, args.roi_dir, meta_csv, args, base_tcc, config_overrides={"time_shift_seconds": float(value)})
            group = "AIF timing"
        elif kind == "snr":
            reps = [evaluate(locked, args.roi_dir, meta_csv, args,
                             perturb_tcc(base_tcc, "snr", value, np.random.default_rng(args.random_state + i)), seed=i)
                    for i in range(int(args.noise_repeats))]
            r = {"n": reps[0]["n"], "auroc": float(np.median([x["auroc"] for x in reps])),
                 "ci_low": float(np.median([x["ci_low"] for x in reps])),
                 "ci_high": float(np.median([x["ci_high"] for x in reps])), "noise_repeats": len(reps)}
            group = "Gaussian noise"
        else:
            raise ValueError(f"Unknown scenario {sc!r}")
        rows.append({"group": group, "scenario": sc, **r})
        print(f"{sc}: AUROC={r['auroc']:.3f}", flush=True)

    for spec in args.roi_variant:
        name, path = spec.split("=", 1)
        r = evaluate(locked, Path(path), meta_csv, args)
        rows.append({"group": "ROI definition", "scenario": name, **r})
        print(f"ROI {name}: AUROC={r['auroc']:.3f}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(args.output_dir / "sensitivity_results.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot = res.iloc[1:].reset_index(drop=True)
    base = res.iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(plot) + 1.2), constrained_layout=True)
    ax.axvspan(base["ci_low"], base["ci_high"], color="0.9")
    ax.axvline(base["auroc"], ls="--", color="0.4", lw=1)
    y = np.arange(len(plot))[::-1]
    ax.errorbar(plot["auroc"], y, xerr=[plot["auroc"] - plot["ci_low"], plot["ci_high"] - plot["auroc"]],
                fmt="o", ms=4, capsize=2, color="C0")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}: {s}" for g, s in zip(plot["group"], plot["scenario"])], fontsize=8)
    ax.set_xlabel("Held-out test AUROC (95% CI)")
    fig.savefig(args.output_dir / "fig5b_sensitivity.png", dpi=300)
    meta_csv.unlink(missing_ok=True)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
