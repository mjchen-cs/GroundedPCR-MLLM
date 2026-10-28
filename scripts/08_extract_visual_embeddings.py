#!/usr/bin/env python3
"""
Step 8 (exploratory, Sec. II-G): frozen RadImageNet-ResNet50 visual embeddings.

For each case, a tumor-centred 2D patch is cut from the axial slice with the
largest tumor cross-section. Three DCE phases (pre-contrast, early and late
post-contrast) are stacked as a pseudo-RGB image, resized to 224 x 224 and
passed through a *frozen* ResNet50 whose classification head is removed,
giving a 2048-D embedding (global average pool).

The embeddings are NOT reduced here. PCA / PLS to 8 components is fitted
fold-wise inside CV by ``03_train_xgboost.py --visual_embedding_csv ...
--visual_reduction {pca,pls}`` so no reduction statistics leak across folds.

RadImageNet weights are not redistributed. Provide a PyTorch checkpoint of
the RadImageNet ResNet50 with ``--weights``; keys with ``fc.`` / module
prefixes are handled and the load report is written to the output JSON.

Output: CSV with ``pid`` + ``vis_0000 ... vis_2047``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from geometry_utils import ensure_3d_image, load_nifti
from io_utils import build_mask_index, discover_dce_timepoints, load_case_map_csv, resolve_mask_path_for_pid

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Frozen RadImageNet-ResNet50 embeddings for the visual branch.")
    p.add_argument("--full_root", type=Path, required=True)
    p.add_argument("--metadata_csv", type=Path, required=True)
    p.add_argument("--mask_dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True, help="RadImageNet ResNet50 PyTorch checkpoint (.pt/.pth).")
    p.add_argument("--output_csv", type=Path, required=True)
    p.add_argument("--case_map_csv", type=Path, default=None)
    p.add_argument("--margin", type=float, default=1.5, help="Crop side = margin x max tumor extent on the slice.")
    p.add_argument("--min_crop", type=int, default=32)
    p.add_argument("--slice_axis", type=int, default=2,
                   help="Array axis indexing axial slices (BreastDCEDL NIfTI: 0).")
    p.add_argument("--device", type=str, default="cpu")
    return p


def build_frozen_resnet50(weights: Path, device: str) -> tuple[torch.nn.Module, dict]:
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    report = model.load_state_dict(state, strict=False)
    missing = [k for k in report.missing_keys if not k.startswith("fc.")]
    if missing:
        raise RuntimeError(f"Checkpoint does not match ResNet50 backbone; missing keys e.g. {missing[:5]}")
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"missing_keys": report.missing_keys, "unexpected_keys": report.unexpected_keys}


def _phase_indices(row: pd.Series, discovered: list[int]) -> list[int]:
    def get(key: str, default: int) -> int:
        v = row.get(key) if key in row.index else None
        return int(v) if v is not None and not pd.isna(v) and int(v) in discovered else default

    pre = get("pre", discovered[0])
    early = get("post_early", discovered[1] if len(discovered) > 1 else discovered[0])
    late = get("post_late", discovered[-1])
    return [pre, early, late]


def _normalise(slice2d: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(slice2d, [1, 99])
    return np.clip((slice2d - lo) / (hi - lo if hi > lo else 1.0), 0, 1).astype(np.float32)


def extract_patch(volumes: list[np.ndarray], mask: np.ndarray, margin: float, min_crop: int) -> np.ndarray:
    areas = mask.reshape(-1, mask.shape[2]).sum(axis=0)
    z = int(np.argmax(areas))
    m2 = mask[:, :, z]
    xs, ys = np.nonzero(m2)
    cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    extent = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    half = int(max(min_crop, margin * extent) // 2)
    chans = []
    for vol in volumes:
        sl = np.pad(vol[:, :, z].astype(np.float32), half, mode="constant")
        crop = sl[cx: cx + 2 * half, cy: cy + 2 * half]
        chans.append(_normalise(crop))
    rgb = np.stack(chans, axis=0)[None]  # 1 x 3 x H x W
    rgb = F.interpolate(torch.from_numpy(rgb), size=(224, 224), mode="bilinear", align_corners=False)[0].numpy()
    rgb = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return rgb.astype(np.float32)


def main() -> None:
    args = build_argparser().parse_args()
    meta = pd.read_csv(args.metadata_csv)
    meta["pid"] = meta["pid"].astype(str).str.strip()
    case_map = load_case_map_csv(args.case_map_csv)
    mask_index = build_mask_index(args.mask_dir, case_map)
    model, load_report = build_frozen_resnet50(args.weights, args.device)

    rows, failures = [], {}
    for _, row in meta.iterrows():
        pid = row["pid"]
        try:
            tps = discover_dce_timepoints(args.full_root, pid)
            paths = dict(tps)
            phases = _phase_indices(row, [t for t, _ in tps])
            vols = [np.asarray(ensure_3d_image(load_nifti(paths[t])).get_fdata(), dtype=np.float32) for t in phases]
            mask_path = resolve_mask_path_for_pid(pid, mask_index, case_map)
            mask = np.asarray(ensure_3d_image(load_nifti(mask_path)).get_fdata()) > 0
            if args.slice_axis != 2:
                vols = [np.moveaxis(v, args.slice_axis, 2) for v in vols]
                mask = np.moveaxis(mask, args.slice_axis, 2)
            if mask.shape != vols[0].shape or not mask.any():
                raise ValueError("mask empty or not aligned to DCE volume")
            x = torch.from_numpy(extract_patch(vols, mask, args.margin, args.min_crop))[None].to(args.device)
            with torch.no_grad():
                emb = model(x).cpu().numpy()[0]
            rows.append({"pid": pid, **{f"vis_{i:04d}": float(v) for i, v in enumerate(emb)}})
        except Exception as exc:  # keep going; failures are reported
            failures[pid] = str(exc)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    with open(args.output_csv.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"n_cases": len(rows), "failures": failures, "weights": str(args.weights),
                   "load_report": load_report, "frozen": True, "embedding_dim": 2048}, f, indent=2)
    print(f"Wrote {len(rows)} embeddings to {args.output_csv} ({len(failures)} failures)")


if __name__ == "__main__":
    main()
