# GroundedPCR-MLLM

Code for the manuscript

> **Physics-Grounded Multimodal Large Language Models for Pretreatment pCR Prediction from Breast DCE-MRI: Grounded Explainers, Not Independent Predictors**
> W. Xiao, Y. Long, C. Liang, M. Chen, A. Farouk *et al.* (under review, *IEEE Journal of Biomedical and Health Informatics*).

The repository contains everything needed to go from the public BreastDCEDL I-SPY2 images to every table, figure and number in the paper:

1. a physics-informed reference model (**B1**): tumor core/rim partition, Tofts-constrained physics-informed neural network (Tofts-PINN) estimates of $K^{trans}$ and $k_{ep}$, tumor volumes and clinical covariates in a shallow XGBoost classifier with TreeSHAP attributions, all trained with fold-wise isolation;
2. a prespecified prompt ladder (**M1, M2, M3f, M3, M4**) that queries commercially hosted multimodal LLMs (GLM-4.5V on Zhipu AI, Qwen3-VL-32B-Instruct on SiliconFlow) through OpenAI-compatible APIs, with every reply cached on disk;
3. an automatic **faithfulness audit** of the replies (numbers that are not in the prompt, evidence directions that contradict the SHAP sign, anchoring on the reference model, stability) plus discrimination, calibration, fusion and secondary analyses.

> **Step-by-step reproduction guide:** [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md)
>
> **The study data are included.** [`data/`](data/README.md) contains everything derived from the images that the paper uses — the ROI curves, B1 inputs, predictions and SHAP values, and all 19,640 MLLM replies — so the paper can be reproduced without downloading 54 GB of images or paying for API calls:
>
> ```bash
> pip install -r requirements.txt
> OUT_DIR=data PAPER_DIR=paper bash scripts/run_study.sh evaluate    # ~10 min, CPU
> ```

---

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick check without data or API keys](#quick-check-without-data-or-api-keys)
- [Included data](#included-data)
- [Data](#data)
- [Reproducing the study](#reproducing-the-study)
- [Reproducing the tables and figures from released results](#reproducing-the-tables-and-figures-from-released-results)
- [What each step produces](#what-each-step-produces)
- [Paper ↔ code](#paper--code)
- [Configuration notes](#configuration-notes)
- [Troubleshooting](#troubleshooting)
- [Legacy pipeline of the original manuscript](#legacy-pipeline-of-the-original-manuscript)
- [Data use, privacy and API providers](#data-use-privacy-and-api-providers)
- [Citation](#citation) · [License](#license)

---

## Repository layout

```
src/                         library code
  roi_utils.py               EDT core / safe-rim partition
  tofts_pinn_utils.py        signal→concentration, Parker AIF, Tofts-PINN training
  heterogeneity_features.py  voxel-level enhancement / SER / peritumoral descriptors
  xgboost_utils.py           feature presets, leakage-safe CV, final model, TreeSHAP export
  mllm_client.py             OpenAI-compatible client (zhipu, siliconflow, offline mock) + reply cache
  mllm_prompts.py            prompt ladder M1–M4, output schema, few-shot selection
scripts/
  00–03, 07, 11              data preparation, ROI/curves, B1 cross-validation, held-out test, SHAP
  12–14                      MLLM inputs, queries, evaluation (writes LaTeX tables and macros)
  15–18                      figures and secondary / sensitivity analyses
  19_make_release_bundle.py  zip of derived results for public release
  check_api.py               one-request connectivity test for each provider
  run_study.sh / .bat        the whole study, stage by stage (Linux-macOS / Windows)
tests/
  test_mllm_pipeline.sh      offline end-to-end test on synthetic data (mock model)
  smoke_test.sh              legacy test of steps 1–10
docs/                        REPRODUCTION_GUIDE.md, paper_to_code_map.md, feature/output definitions, leakage control
```

## Installation

Python ≥ 3.10. A GPU is **not** required (the PINN is small; everything was run on CPU).

```bash
git clone https://github.com/mjchen-cs/GroundedPCR-MLLM.git
cd GroundedPCR-MLLM
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`torch` CPU wheels are sufficient. `torchvision` is only needed for the legacy visual-branch ablation (step 08).

## Quick check without data or API keys

The test below builds a small synthetic DCE cohort, runs every step of the study with the deterministic offline `mock` model and checks that all tables and figures are written. It needs no download, no key and no money (a few minutes on a laptop).

```bash
bash tests/test_mllm_pipeline.sh          # ends with: MLLM PIPELINE TEST PASSED
```

The numbers it produces are meaningless; it only verifies the installation.

## Included data

| Starting point | What you need | Command |
|---|---|---|
| **A. Tables, figures and numbers of the paper** | this repository only | `OUT_DIR=data PAPER_DIR=paper bash scripts/run_study.sh evaluate` |
| **B. Re-train B1 and the analyses from the shipped ROI curves** | this repository only | copy `data/metadata` and `data/ispy2_roi` to a new `OUT_DIR`, then `run_study b1` → (`export` needs images) |
| **C. Re-query MLLMs with the same prompts** | `data/mllm/cases.jsonl` + image panels from the GitHub Release asset + API keys | `scripts/13_run_mllm.py` |
| **D. Everything from the raw images** | BreastDCEDL I-SPY2 images (54 GB) + API keys | stages `features` → `evaluate` below |

Details, provenance and terms of use: [`data/README.md`](data/README.md). The raw NIfTI images and masks are not in the repository (size and TCIA terms); they are only needed for D.

On Windows, run A with `cmd /c "set OUT_DIR=%CD%\data&& set PAPER_DIR=%CD%\paper&& scripts\run_study.bat evaluate"`.

## Data

The study uses only the **I-SPY2 subset of BreastDCEDL** (982 patients; the BreastDCEDL `test` flag gives 784 development and 198 held-out patients). Two downloads are needed:

| What | Where | Used for |
|---|---|---|
| BreastDCEDL GitHub repository (metadata CSVs) | <https://github.com/naomifridman/BreastDCEDL> | `BreastDCEDL_metadata.csv` (voxel spacing, phase indices, labels, split) and `ISPY2/BreastDCEDL_spy2_tcia_metadata.csv` (MammaPrint, treatment arm) |
| BreastDCEDL I-SPY2 image package (NIfTI DCE + tumor masks, ≈ 54 GB) | The Cancer Imaging Archive, linked from the BreastDCEDL README (download with IBM Aspera / the TCIA Data Retriever) | images and masks |

After download the image folder must look like this (the layout of the TCIA package):

```
<DATA_DIR>/                       e.g. .../PKG - BreastDCEDL_ISPY2/BreastDCEDL_ISPY2
  ACRIN-6698-102212/
    dce/ACRIN-6698-102212_spy2_vis1_dce_aqc_0.nii.gz … _aqc_7.nii.gz
    mask/ACRIN-6698-102212_spy2_vis1_mask.nii.gz
  ACRIN-6698-.../
<BREASTDCEDL_REPO>/
  BreastDCEDL_metadata.csv
  ISPY2/BreastDCEDL_spy2_tcia_metadata.csv
```

Notes: the NIfTI headers of BreastDCEDL report 1 mm isotropic voxels, so the true spacing is taken from the metadata (`--spacing_source metadata`); axial slices are on axis 0 (`--slice_axis 0`). The tumor masks are those distributed with BreastDCEDL; no segmentation network is run.

## Reproducing the study

All commands are run from the repository root. The study is split into stages; each stage skips work whose outputs already exist, so an interrupted run can simply be restarted.

### 1. Set paths

**Linux / macOS**

```bash
export DATA_DIR="/data/PKG - BreastDCEDL_ISPY2/BreastDCEDL_ISPY2"
export BREASTDCEDL_REPO=/data/BreastDCEDL
export OUT_DIR=results            # all intermediate and final results
export PAPER_DIR=results/paper    # LaTeX tables, macros and figures for the manuscript
```

**Windows** — edit the four `set ...` lines at the top of `scripts\run_study.bat`, or set the same variables in the terminal. In PowerShell use `$env:DATA_DIR="D:\data\BreastDCEDL_ISPY2"`; the cmd syntax `set X=...` does **not** work in PowerShell.

### 2. Features and reference model B1 (no API, CPU only)

```bash
bash scripts/run_study.sh features   # steps 00–01: metadata, ROIs, curves, heterogeneity  (~1–2 h, 8 processes)
bash scripts/run_study.sh b1         # steps 03, 11, 07: 7 feature sets by 5-fold CV, B1 selection,
                                     # one held-out evaluation, SHAP, sensitivity runs   (~1 h)
```

Windows: `cmd /c scripts\run_study.bat features`, then `... b1`.

Step 01 chooses the number of worker processes from the available memory (about 2 GB per worker, at most 8). B1 is the feature set with the highest development fold-mean AUROC among the sets that contain the Tofts-PINN features (`main`, `main_mp`, `main_het`, `main_mp_het`); in the paper this was `main_mp` (13 inputs). The held-out set is used once, for this model only.

### 3. MLLM inputs

```bash
bash scripts/run_study.sh export     # step 12: cases.jsonl + one 2×2 DCE panel per patient (~20–30 min)
```

### 4. API keys and connectivity check

Keys are read from environment variables only and are never written to disk.

```bash
export ZHIPU_API_KEY=...            # PowerShell: $env:ZHIPU_API_KEY="..."
export SILICONFLOW_API_KEY=...
python scripts/check_api.py --provider zhipu --model glm-4.5v --disable_thinking
python scripts/check_api.py --provider siliconflow --model Qwen/Qwen3-VL-32B-Instruct
python scripts/check_api.py --provider siliconflow --list_models   # vision models available to your key
```

Both must print `HTTP 200`. Model availability changes; override the model ids with `ZHIPU_MODEL` / `SF_MODEL` if needed (the paper used `glm-4.5v` and `Qwen/Qwen3-VL-32B-Instruct`, queried in October 2026).

### 5. Pilot and full queries (paid)

```bash
bash scripts/run_study.sh pilot      # 20 development cases × 5 configurations × 2 providers, 1 repeat
bash scripts/run_study.sh query      # 982 cases × 5 configurations × 2 repeats × 2 providers = 19,640 requests
```

Check `results/mllm/runs/*/*.csv` after the pilot (`parse_ok` should be 1, `prompt_tokens` about 400–2000). The full run took about 3–4 h for Zhipu and 8–10 h for SiliconFlow with 8 parallel workers and used about 14 million tokens per provider (≈ 11.5 M input, 2.7 M output); at October 2026 prices this was roughly CNY 20–30 per provider. Every reply is cached under `results/mllm/cache/`, so re-running never pays twice and an interrupted run resumes where it stopped. If the provider returns HTTP 429, lower `WORKERS`.

### 6. Evaluation, tables and figures (offline)

```bash
bash scripts/run_study.sh evaluate   # steps 14–18 (~10 min)
```

This writes into `PAPER_DIR`:

```
results_macros.tex                \res... macros used in the text (e.g. \resBOneTestAUROC)
tables/table_*_rows.tex           rows of Tables I–IX
figures/fig1_workflow.pdf         Fig. 1   figures/fig_results.pdf   Fig. 4
figures/fig_calib_dca.pdf         Fig. 5   figures/fig_shap_citation.pdf Fig. 6
analysis/*.csv                    every number behind the secondary analyses
```

and `results/mllm/evaluation/*.csv` (prediction and faithfulness metrics, per-reply faithfulness). Put the manuscript `main_mllm.tex` in `PAPER_DIR` (or copy the files next to it) and compile with `pdflatex → bibtex → pdflatex ×2`.

`bash scripts/run_study.sh all` runs stages 2–6 in one go (without the pilot).

## Reproducing the tables and figures from released results

The derived data are shipped in [`data/`](data/README.md) (starting point A above). The GitHub Release additionally provides `results_with_images.zip` with all 982 input panels and the API reply cache, which is needed only to re-query the models (starting point C). That archive is built with

```bash
python scripts/19_make_release_bundle.py --out_dir results --bundle release/results_with_images.zip --all_images --include_cache
```

## What each step produces

| Step | Script | Main outputs |
|---|---|---|
| 00 | `00_prepare_breastdcedl.py` | `metadata/ISPY2/metadata.csv` (labels, split, spacing, phases, MammaPrint, arm) |
| 01 | `01_extract_roi_tcc.py` | `ispy2_roi/`: `tcc_long.csv` (core/rim/tumor curves), `volume_summary.csv`, `dynamics_feature_summary.csv` (incl. `het_*`) |
| 03 | `03_train_xgboost.py` | per run: fold-wise Tofts-PINN features, `xgboost_predictions.csv`, `xgboost_metrics.json`, OOF/held-out inputs and TreeSHAP, `final_model/` |
| 07 | `07_shap_analysis.py` | `b1/shap/shap_values.csv`, `shap_importance.csv` |
| 11 | `11_compare_cv.py` | `cv/cv_comparison.csv`, `cv/best_preset.txt`, `tables/table_cv_rows.tex` |
| 12 | `12_export_mllm_inputs.py` | `mllm/cases.jsonl`, `mllm/images/<pid>.png` |
| 13 | `13_run_mllm.py` | `mllm/runs/<provider>__<model>/<config>_<split>.csv`, `mllm/cache/*.json` |
| 14 | `14_evaluate_mllm.py` | `mllm/evaluation/*.csv`, `results_macros.tex`, prediction and faithfulness tables |
| 15 | `15_plot_results.py` | `figures/fig_results.pdf` |
| 16 | `16_plot_workflow.py` | `figures/fig1_workflow.pdf` (one real held-out case) |
| 17 | `17_extra_analyses.py` | cohort, subtype, error-taxonomy and image-claim tables; calibration/DCA and SHAP/citation figures; `analysis/*.csv` |
| 18 | `18_b1_sensitivity.py` | B1 sensitivity table (recalibration, logistic regression, no weighting, depth 3, + arm) |
| 19 | `19_make_release_bundle.py` | zip of derived results |

## Paper ↔ code

See [`docs/paper_to_code_map.md`](docs/paper_to_code_map.md) for the mapping of every section, equation, table and figure to the code and output file that produces it.

## Configuration notes

| Setting | Value in the paper | Where |
|---|---|---|
| Core / rim | inner 15 %; rim = 10th–28th distance percentile | `01 --core_fraction`, `src/roi_utils.py` |
| Signal → concentration | SPGR, TR 4 ms, flip 15°, T10 1.6 s, r1 4.5; Δt 90 s if timing missing | `src/tofts_pinn_utils.py` |
| Tofts-PINN | MLP 2Nt+1→32→32→2, Softplus, Parker AIF, ≤ 250 epochs, Adam 1e-3 | `03 --pinn_*` |
| XGBoost | depth 2, lr 0.02, subsample 0.85, colsample 0.75, λ 6, α 0.5, γ 0.1, max_delta_step 1, scale_pos_weight = n_neg/n_pos, early stopping 100 (log-loss, inner 15 %) | `03` defaults |
| Prompt values | 3 significant digits; 8 largest-\|SHAP\| values; 6 balanced reference cases (M4) from other folds | `src/mllm_prompts.py` |
| Query | temperature 0.01, max 1200 tokens, 2 repeats, GLM "thinking" disabled | `13` defaults, `--disable_thinking` |
| Faithfulness | 1 % or half-unit tolerance; thousands separators removed; approximations (±10 %, mm³→cm³) counted separately; SHAP threshold 0.02 | `14_evaluate_mllm.py` |
| Bootstrap | 2000 case-level resamples; paired DeLong test | `14 --n_bootstrap` |

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: Set the environment variable ZHIPU_API_KEY` in PowerShell after `set ZHIPU_API_KEY=...` | `set` is not an environment variable in PowerShell; use `$env:ZHIPU_API_KEY="..."` in the same window. |
| `HTTP 401` from Zhipu | Key incomplete (Zhipu keys contain a dot, `<id>.<secret>`) or expired. |
| `HTTP 403 {"code":30003,"message":"Model disabled."}` from SiliconFlow | The model is not available to your account; pick another with `check_api.py --list_models` and set `SF_MODEL`. |
| `HTTP 429` | Rate limit or balance; lower `WORKERS` and re-run (cached replies are reused). |
| Step 01 uses too much memory / the IDE closes | Set `--n_jobs` lower in `run_study.*` (default auto: ~2 GB per worker, max 8). |
| Nothing printed for a long time in step 13 | Requests are retried with back-off on time-outs; progress lines appear every 10 requests. Check the network or proxy with `check_api.py`. |
| Numbers differ in the 3rd decimal from the paper | PINN training on another platform; a re-run in a second environment changed the B1 fold-mean AUROC by 0.002. MLLM replies differ slightly between calls even at temperature 0.01; use the released replies for exact numbers. |

## Legacy pipeline of the original manuscript

Steps 05 (external validation), 06 (cohort statistics, calibration, DCA), 08–10 (visual-branch ablation, sensitivity) and `scripts/run_full_pipeline.sh` belong to the earlier physics-informed XGBoost manuscript that used I-SPY1 and Duke as external cohorts. They are kept and tested (`tests/smoke_test.sh`) but are **not** used in the MLLM study; their documentation is in `docs/legacy/`.

## Data use, privacy and API providers

- This repository contains derived, de-identified patient-level data in `data/` (see [`data/README.md`](data/README.md) for provenance and terms) but no raw images or masks. `.gitignore` excludes your own `results/`, NIfTI files and the API cache.
- BreastDCEDL / I-SPY2 data are subject to the TCIA data-use terms; cite the original collections.
- Step 13 sends de-identified 2×2 image panels and tabular features to third-party APIs. Check that the data-use terms of your data source and your institution permit this, and review the providers' data-retention settings, before running it.
- Never commit API keys. Use environment variables or a local `.env` file (template: `.env.example`, ignored by git).

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)) and the data sources:

- N. Fridman *et al.*, "BreastDCEDL: A standardized deep learning-ready breast DCE-MRI dataset of 2,070 patients," *Scientific Data*, 2026.
- W. Li *et al.*, "I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2)," The Cancer Imaging Archive, 2022, doi:10.7937/TCIA.D8Z0-9T85.
- K. Clark *et al.*, "The Cancer Imaging Archive (TCIA)," *J. Digit. Imaging*, 2013.

## License

MIT, see [`LICENSE`](LICENSE). The license covers the code only, not the data.
