#!/usr/bin/env python3
"""
Step 12: Export per-case MLLM inputs (image panel + case JSON).

For every I-SPY2 case with a B1 prediction (development OOF or held-out test):

- ``images/<pid>.png``: 2x2 panel (pre / early / late post-contrast, early with
  the tumour outline in red) of the axial slice with the largest tumour
  cross-section, cropped around the tumour (1.5x its extent, at least 64 px),
  resized to 256 x 256 per panel. The three intensity panels share one window
  (1st-99.5th percentile of the crop across phases).
- ``cases.jsonl``: one record per case with the clinical covariates, the B1
  model features (raw values), the B1 probability and per-case TreeSHAP values.
  Development cases use the fold-wise (out-of-fold) model; held-out cases use
  the final development model, so no case is described by a model that saw it.
  The pCR label is stored for evaluation only and is never put in a prompt
  except as the outcome of a few-shot reference case from the training folds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from geometry_utils import ensure_3d_image, load_nifti
from io_utils import build_mask_index, discover_dce_timepoints, resolve_mask_path_for_pid

CLINICAL_MAP = {"clinical_age": "age", "clinical_hr": "HR", "clinical_her2": "HER2",
                "clinical_triple_neg": "TNBC", "clinical_mp": "MammaPrint"}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export MLLM image panels and case records.")
    p.add_argument("--data_root", type=Path, required=True, help="BreastDCEDL I-SPY2 folder (<pid>/dce, <pid>/mask).")
    p.add_argument("--mask_dir", type=Path, default=None, help="Defaults to --data_root.")
    p.add_argument("--metadata_csv", type=Path, required=True, help="Prepared I-SPY2 metadata (step 0).")
    p.add_argument("--xgb_dir", type=Path, required=True, help="Step-3 output of the chosen B1 preset (with --run_final_test).")
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--slice_axis", type=int, default=0, help="Axis indexing axial slices (BreastDCEDL: 0).")
    p.add_argument("--panel_size", type=int, default=256)
    p.add_argument("--skip_images", action="store_true")
    return p


def _clean(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _norm(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a - lo) / (hi - lo if hi > lo else 1.0), 0, 1)


def make_panel(vols: list[np.ndarray], mask: np.ndarray, size: int) -> np.ndarray:
    from PIL import Image

    areas = mask.reshape(mask.shape[0], -1).sum(axis=1)
    z = int(np.argmax(areas))
    m2 = mask[z]
    ys, xs = np.nonzero(m2)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    extent = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    half = int(max(32, 1.5 * extent / 2))
    crops = []
    for v in vols + [m2.astype(np.float32)[None]]:
        sl = v[z] if v.ndim == 3 and v.shape[0] == mask.shape[0] else v[0]
        pad = np.pad(sl.astype(np.float32), half, mode="constant")
        crops.append(pad[cy: cy + 2 * half, cx: cx + 2 * half])
    imgs, mcrop = crops[:3], crops[3] > 0.5
    lo, hi = np.percentile(np.concatenate([c.ravel() for c in imgs]), [1, 99.5])
    gray = [(_norm(c, lo, hi) * 255).astype(np.uint8) for c in imgs]
    # outline of the closed, hole-filled mask (BreastDCEDL masks are voxel-wise and patchy)
    from scipy import ndimage as ndi

    solid = ndi.binary_fill_holes(ndi.binary_closing(mcrop, structure=np.ones((3, 3)), iterations=3))
    lab, n = ndi.label(solid)
    if n > 1:
        sizes = ndi.sum(solid, lab, range(1, n + 1))
        solid = lab == (1 + int(np.argmax(sizes)))
    edge = solid & ~ndi.binary_erosion(solid, structure=np.ones((3, 3)))
    over = np.stack([gray[1]] * 3, axis=-1)
    over[edge] = [255, 0, 0]
    tiles = [np.stack([g] * 3, axis=-1) for g in gray] + [over]
    tiles = [np.asarray(Image.fromarray(t).resize((size, size), Image.NEAREST if i == 3 else Image.BILINEAR)) for i, t in enumerate(tiles)]
    top = np.concatenate([tiles[0], tiles[1]], axis=1)
    bottom = np.concatenate([tiles[2], tiles[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main() -> None:
    args = build_argparser().parse_args()
    from PIL import Image

    out = args.output_dir
    (out / "images").mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.metadata_csv)
    meta["pid"] = meta["pid"].astype(str).str.strip()
    meta = meta.set_index("pid")

    preds = pd.read_csv(args.xgb_dir / "xgboost_predictions.csv")
    preds["pid"] = preds["pid"].astype(str)
    oof_in = pd.read_csv(args.xgb_dir / "xgboost_cv_oof_inputs.csv")
    oof_shap = pd.read_csv(args.xgb_dir / "xgboost_cv_oof_shap.csv")
    test_in = pd.read_csv(args.xgb_dir / "heldout_test_inputs.csv")
    test_shap = pd.read_csv(args.xgb_dir / "heldout_test_shap.csv")
    inputs = pd.concat([oof_in.assign(split="development"), test_in.assign(split="heldout_test", fold_index=0)], ignore_index=True)
    shaps = pd.concat([oof_shap.assign(split="development"), test_shap.assign(split="heldout_test", fold_index=0)], ignore_index=True)
    for d in (inputs, shaps):
        d["pid"] = d["pid"].astype(str)
    shap_cols = [c for c in shaps.columns if c not in ("pid", "fold_index", "shap_base_value", "split")]

    mask_index = build_mask_index(args.mask_dir or args.data_root, None)
    n_ok, failures = 0, {}
    with open(out / "cases.jsonl", "w", encoding="utf-8") as f:
        n_total = len(inputs)
        print(f"Exporting {n_total} cases (panel images take ~1 s per case)...", flush=True)
        for i_row, (_, row) in enumerate(inputs.iterrows(), 1):
            if i_row % 25 == 0 or i_row == n_total:
                print(f"  {i_row}/{n_total}", flush=True)
            pid = row["pid"]
            split = row["split"]
            pr = preds[(preds["pid"] == pid) & (preds["split_group"] == ("development_oof" if split == "development" else "heldout_test"))]
            sh = shaps[(shaps["pid"] == pid) & (shaps["split"] == split)]
            if pr.empty or sh.empty:
                failures[pid] = "missing prediction or SHAP"
                continue
            clinical, features = {}, {}
            for c in inputs.columns:
                if c in ("pid", "fold_index", "split") or c.startswith("arm_"):
                    continue
                if c in CLINICAL_MAP:
                    clinical[CLINICAL_MAP[c]] = _clean(row[c])
                else:
                    features[c] = _clean(row[c])
            if "MammaPrint" not in clinical and pid in meta.index and "MP" in meta.columns:
                clinical["MammaPrint"] = _clean(meta.loc[pid, "MP"])
            rec = {
                "pid": pid, "split": split, "fold_index": int(row["fold_index"]),
                "label": int(pr["y_true"].iloc[0]),
                "subtype": str(pr["molecular_subtype"].iloc[0]) if "molecular_subtype" in pr.columns else None,
                "clinical": clinical, "features": features,
                "xgb_probability": float(pr["y_prob"].iloc[0]),
                # SHAP keys use the prompt's names (clinical_hr -> HR, ...)
                "shap": {CLINICAL_MAP.get(c, c): float(sh[c].iloc[0]) for c in shap_cols if not c.startswith("arm_")},
            }
            if not args.skip_images:
                try:
                    m = meta.loc[pid]
                    tps = dict(discover_dce_timepoints(args.data_root, pid))
                    idx = [int(m["pre"]), int(m["post_early"]), int(m["post_late"])]
                    vols = [np.moveaxis(np.asarray(ensure_3d_image(load_nifti(tps[i])).dataobj, dtype=np.float32), args.slice_axis, 0) for i in idx]
                    mask = np.moveaxis(np.asarray(ensure_3d_image(load_nifti(resolve_mask_path_for_pid(pid, mask_index, None))).dataobj) > 0, args.slice_axis, 0)
                    Image.fromarray(make_panel(vols, mask, args.panel_size)).save(out / "images" / f"{pid}.png")
                except Exception as exc:
                    failures[pid] = f"image: {exc}"
                    continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
    (out / "export_report.json").write_text(json.dumps({"n_cases": n_ok, "failures": failures}, indent=1), encoding="utf-8")
    print(f"Exported {n_ok} cases to {out} ({len(failures)} failures)")


if __name__ == "__main__":
    main()
