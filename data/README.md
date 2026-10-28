# Study data included in this repository

This folder holds the **derived data of the I-SPY2 study exactly as used in the paper** (≈ 48 MB, plain text). It has the same layout as the `OUT_DIR` of `scripts/run_study.*`, so every later step can start from it:

```bash
# regenerate every table, figure and number of the paper (no images, no API key, ~10 min)
OUT_DIR=data PAPER_DIR=paper bash scripts/run_study.sh evaluate
```

(`evaluate` rewrites `data/mllm/evaluation/`; the regenerated files are identical to the shipped ones.)

| Folder / file | Content | Produced by |
|---|---|---|
| `metadata/ISPY2/metadata.csv` | 982 patients: pCR label, BreastDCEDL train/test flag, voxel spacing, DCE phase indices, age, HR, HER2, subtype, MammaPrint, treatment-arm indicators | step 00 |
| `ispy2_roi/tcc_long.csv` | mean signal curves of tumor, core (inner 15 %) and safe rim for every DCE phase | step 01 |
| `ispy2_roi/volume_summary.csv`, `dynamics_feature_summary.csv` | volumes, curve summaries, 28 heterogeneity descriptors (`het_*`) | step 01 |
| `ispy2_roi/qc_report.csv` | per-case geometry / ROI checks (image paths replaced by `<DATA_DIR>`) | step 01 |
| `cv/` | development cross-validation of the 7 feature sets, `cv_comparison.csv`, `best_preset.txt` (= `main_mp`) | steps 03, 11 |
| `b1/` | locked reference model B1: out-of-fold and held-out predictions and inputs (incl. fold-wise Tofts-PINN $K^{trans}$, $k_{ep}$), TreeSHAP values, model card | steps 03, 07 |
| `sens/` | development CV of the B1 sensitivity variants (no class weighting, depth 3, + treatment arm, re-run in a second environment) | step 03 |
| `mllm/cases.jsonl` | one record per patient: split, fold, label, subtype, clinical covariates, B1 features, B1 probability, SHAP values — exactly what the prompts were built from | step 12 |
| `mllm/runs/<provider>__<model>/<config>_<split>.csv` | **all 19,640 MLLM replies** (GLM-4.5V, Qwen3-VL-32B-Instruct; M1, M2, M3f, M3, M4; development and held-out; 2 repeats): parsed probability, conclusion, evidence, report, raw reply text, tokens, latency | step 13 |
| `mllm/images/ACRIN-6698-102212.png` | the input panel of the case shown in Fig. 1 | step 12 |

## What is *not* here, and how to get it

- **Raw DCE-MRI volumes and tumor masks** (NIfTI, ≈ 54 GB). They cannot be stored on GitHub and are distributed by The Cancer Imaging Archive under its data-use terms. Download the BreastDCEDL I-SPY2 package as described in the main README; they are only needed to re-run steps 01 and 12.
- **The other 981 image panels and the API reply cache.** These are only needed to re-query the MLLMs and are attached to the GitHub Release as `results_with_images.zip` (created with `python scripts/19_make_release_bundle.py --out_dir <results> --bundle results_with_images.zip --all_images --include_cache`).

## Starting points

| You want to … | Run |
|---|---|
| regenerate the paper's tables and figures | `OUT_DIR=data PAPER_DIR=paper bash scripts/run_study.sh evaluate` |
| re-train B1 from the shipped curves (no images) | copy `data/metadata` and `data/ispy2_roi` into a new `OUT_DIR`, then `bash scripts/run_study.sh b1` |
| query other MLLMs with the same prompts | use `data/mllm/cases.jsonl` plus the panels from the release asset, then `scripts/13_run_mllm.py` |
| re-run everything from the images | follow "Reproducing the study" in the main README |

## Provenance and terms of use

All files are derived from the I-SPY2 collection of The Cancer Imaging Archive (W. Li *et al.*, doi:10.7937/TCIA.D8Z0-9T85) as redistributed by BreastDCEDL (N. Fridman *et al.*, *Scientific Data*, 2026). Patient identifiers are the public TCIA identifiers; no other identifying information is included. Use of these derived data is subject to the terms of the original collections; please cite them together with this repository.
