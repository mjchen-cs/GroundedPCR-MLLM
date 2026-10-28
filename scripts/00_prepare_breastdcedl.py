#!/usr/bin/env python3
"""
Step 0: Prepare per-cohort metadata from the BreastDCEDL release.

Splits ``BreastDCEDL_metadata.csv`` (GitHub repo naomifridman/BreastDCEDL) into
one ``metadata.csv`` per cohort in the format expected by this pipeline:

- ``ISPY2/metadata.csv``: all 982 cases; ``test`` in {1, 2} = held-out test
  (784 development / 198 held-out), i.e. the BreastDCEDL benchmark split.
- ``ISPY1/metadata.csv`` and ``Duke/metadata.csv``: cases with a pCR label;
  ``test`` is dropped because the whole cohort is external.

BreastDCEDL NIfTI headers store 1 x 1 x 1 mm spacing and (z, y, x) array
order; run step 1 with ``--spacing_source metadata --slice_axis 0`` so that
``xy_spacing`` / ``slice_thick`` (or ``voxel_z``) from these files are used.

    python scripts/00_prepare_breastdcedl.py \
        --breastdcedl_metadata /path/to/BreastDCEDL-main/BreastDCEDL_metadata.csv \
        --duke_metadata /path/to/BreastDCEDL-main/DUKE/BreastDCEDL_duke_metadata.csv \
        --output_dir /path/to/prepared_metadata
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COHORTS = {"spy2": "ISPY2", "spy1": "ISPY1", "duke": "Duke"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--breastdcedl_metadata", type=Path, required=True)
    p.add_argument("--duke_metadata", type=Path, default=None,
                   help="Optional DUKE/BreastDCEDL_duke_metadata.csv (adds voxel_z for exact slice spacing).")
    p.add_argument("--ispy2_clinical", type=Path, default=None,
                   help="ISPY2/BreastDCEDL_spy2_tcia_metadata.csv: adds MammaPrint (MP) and treatment-arm columns to I-SPY 2.")
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()

    meta = pd.read_csv(args.breastdcedl_metadata)
    meta["pid"] = meta["pid"].astype(str).str.strip()
    if args.duke_metadata is not None:
        duke = pd.read_csv(args.duke_metadata)[["pid", "voxel_z"]]
        duke["pid"] = duke["pid"].astype(str).str.strip()
        meta = meta.merge(duke, on="pid", how="left")

    if args.ispy2_clinical is not None:
        clin = pd.read_csv(args.ispy2_clinical)
        clin["pid"] = clin["pid"].astype(str).str.strip()
        arm_cols = ["ABT 888", "AMG 386", "Carboplatin", "Ganetespib", "Ganitumab", "MK-2206", "Neratinib",
                    "Pembrolizumab", "Pertuzumab", "T-DM1", "Trastuzumab"]
        keep = ["pid", "MP"] + [c for c in arm_cols if c in clin.columns]
        clin = clin[keep].rename(columns={c: "arm_" + c.replace(" ", "").replace("-", "") for c in arm_cols})
        meta = meta.merge(clin, on="pid", how="left")

    for key, name in COHORTS.items():
        sub = meta.loc[meta["dataset"] == key].copy()
        if key != "spy2":
            sub = sub.loc[sub["pCR"].notna()].drop(columns=["test"], errors="ignore")
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out / "metadata.csv", index=False)
        msg = f"{name}: n={len(sub)} pCR={int(sub['pCR'].sum())}"
        if key == "spy2":
            held = sub["test"].isin([1, 2])
            msg += f" | development={int((~held).sum())} held-out={int(held.sum())}"
        print(msg)


if __name__ == "__main__":
    main()
