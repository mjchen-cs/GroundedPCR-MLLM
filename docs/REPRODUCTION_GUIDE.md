# Step-by-step reproduction guide

This guide walks through reproducing every result of *Physics-Grounded Multimodal Large Language Models for Pretreatment pCR Prediction from Breast DCE-MRI* from public data. The repository overview is in the root `README.md`.

## Fastest route (no image download, no API)

The `data/` folder already contains all derived data used in the paper: enhancement curves, B1 inputs, predictions and SHAP values, and all 19,640 MLLM replies. After installing the dependencies, one command regenerates every table, figure and number of the paper:

```bash
pip install -r requirements.txt
OUT_DIR=data PAPER_DIR=paper bash scripts/run_study.sh evaluate      # about 10 minutes
```

Windows (cmd):

```bat
cmd /c "set OUT_DIR=%CD%\data&& set PAPER_DIR=%CD%\paper&& scripts\run_study.bat evaluate"
```

The output is written to `paper/`. `data/README.md` describes each file and the step that produces it. Sections 0–6 below cover a full reproduction starting from the raw images.

## 0. Requirements

| Item | Notes |
|---|---|
| Computer | Windows, Linux or macOS; **no GPU needed**; ≥ 16 GB RAM recommended (step 01 chooses the number of worker processes at about 2 GB each) |
| Disk | about 54 GB for the images and about 2 GB for the results |
| Python | 3.10 or later |
| API access (step 5 only) | Zhipu AI open platform and SiliconFlow accounts, each with about CNY 30 of credit (October 2026 prices) |

## 1. Installation

```bash
git clone https://github.com/mjchen-cs/GroundedPCR-MLLM.git
cd GroundedPCR-MLLM
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
```

**Run the offline self-test first.** It needs no data, no API key and no money, and takes a few minutes:

```bash
bash tests/test_mllm_pipeline.sh
```

The environment is ready when it ends with `MLLM PIPELINE TEST PASSED`. On Windows, run this command in Git Bash.

## 2. Download the data

1. **BreastDCEDL metadata:** clone <https://github.com/naomifridman/BreastDCEDL>. You need `BreastDCEDL_metadata.csv` and `ISPY2/BreastDCEDL_spy2_tcia_metadata.csv`.
2. **I-SPY2 image package:** download `BreastDCEDL_ISPY2` from TCIA via the link in the BreastDCEDL README (about 54 GB; use IBM Aspera or the TCIA Data Retriever). Only this subset is needed; I-SPY1 and Duke are not used.

After download the folder should look like this:

```
BreastDCEDL_ISPY2/
  ACRIN-6698-102212/
    dce/ACRIN-6698-102212_spy2_vis1_dce_aqc_0.nii.gz ... _aqc_7.nii.gz
    mask/ACRIN-6698-102212_spy2_vis1_mask.nii.gz
  ...
```

## 3. Set the paths

**Windows:** open `scripts\run_study.bat` in a text editor and edit these lines near the top:

```bat
if "%DATA_DIR%"=="" set DATA_DIR=D:\data\BreastDCEDL_ISPY2
if "%BREASTDCEDL_REPO%"=="" set BREASTDCEDL_REPO=D:\data\BreastDCEDL
```

Results are written to `results\` inside the repository by default.

**Linux / macOS:**

```bash
export DATA_DIR="/data/BreastDCEDL_ISPY2"
export BREASTDCEDL_REPO=/data/BreastDCEDL
export OUT_DIR=results
export PAPER_DIR=results/paper
```

## 4. Features and reference model B1 (no API)

| Command (Windows) | Command (Linux/macOS) | What it does | Time |
|---|---|---|---|
| `cmd /c scripts\run_study.bat features` | `bash scripts/run_study.sh features` | prepares the metadata, builds the core and rim, extracts enhancement curves and heterogeneity features | 1–2 h |
| `cmd /c scripts\run_study.bat b1` | `bash scripts/run_study.sh b1` | 5-fold CV of 7 feature sets, selects B1, evaluates the held-out set once, computes SHAP, runs the sensitivity variants | about 1 h |
| `cmd /c scripts\run_study.bat export` | `bash scripts/run_study.sh export` | builds the MLLM inputs: `cases.jsonl` and a 2×2 image panel per patient | 20–30 min |

Every stage skips work that is already done, so an interrupted stage can simply be run again.

## 5. Querying the MLLMs (paid)

**Step 1 — set the API keys.** Keep them in environment variables only; never write them into a file.

The PyCharm terminal is PowerShell by default, which needs the `$env:` syntax:

```powershell
$env:ZHIPU_API_KEY="your-zhipu-key"            # a complete Zhipu key contains a "."
$env:SILICONFLOW_API_KEY="your-siliconflow-key"
```

`set ZHIPU_API_KEY=...` has **no effect** in PowerShell. Keys set this way are valid only in the current window.

**Step 2 — check connectivity.** Both commands should print `HTTP 200`:

```bash
python scripts/check_api.py --provider zhipu --model glm-4.5v --disable_thinking
python scripts/check_api.py --provider siliconflow --model Qwen/Qwen3-VL-32B-Instruct
```

If SiliconFlow returns `Model disabled`, list the vision models available to your key with `python scripts/check_api.py --provider siliconflow --list_models`, then set `$env:SF_MODEL="<model id>"`.

**Step 3 — pilot run** (20 cases, a few minutes, less than CNY 1):

```
cmd /c scripts\run_study.bat pilot        (Linux: bash scripts/run_study.sh pilot)
```

Open `results\mllm\runs\*\*.csv` and check that `parse_ok` is 1 everywhere.

**Step 4 — full run:**

```
cmd /c scripts\run_study.bat query
```

- 19,640 requests in total: 982 cases × 5 configurations × 2 repeats × 2 providers.
- About 14 million tokens per provider, roughly CNY 20–30 each at October 2026 prices.
- About 3–4 h for Zhipu and 8–10 h for SiliconFlow; run it overnight and disable sleep mode.
- If the run stops, start it again: completed replies are read from the cache in `results\mllm\cache\` and are **not charged twice**.
- If many HTTP 429 errors (too many requests) appear, lower the concurrency: `$env:WORKERS="4"`.

## 6. Tables and figures of the paper (offline, about 10 minutes)

```
cmd /c scripts\run_study.bat evaluate
```

Output in `results\paper\`:

- `results_macros.tex`: every number cited in the text;
- `tables\*.tex`: the contents of Tables I–IX;
- `figures\*.pdf`: Figs. 1, 4, 5 and 6;
- `analysis\*.csv`: the data behind the secondary analyses.

Put the manuscript `main_mllm.tex` in this folder (or copy these files next to it) and compile with `pdflatex → bibtex → pdflatex → pdflatex`.

## 7. Starting points

| Goal | What you need | How |
|---|---|---|
| Regenerate the tables, figures and numbers of the paper | this repository only | `OUT_DIR=data PAPER_DIR=paper bash scripts/run_study.sh evaluate` |
| Re-train B1 from the shipped curves | this repository only | copy `data/metadata` and `data/ispy2_roi` into a new results folder, then run the `b1` stage |
| Test other MLLMs with the same prompts | `data/mllm/cases.jsonl` + the image-panel archive from the GitHub Release + API keys | run `scripts/13_run_mllm.py` |
| Full reproduction from the raw images | the 54 GB images + API keys | sections 2–6 above |

The raw NIfTI images and masks are not on GitHub: they are too large (54 GB) and are governed by the TCIA data-use terms, so they must be downloaded from TCIA.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Set the environment variable ZHIPU_API_KEY` | `set` was used in PowerShell; use `$env:ZHIPU_API_KEY="..."` and run in the same window |
| Zhipu returns HTTP 401 | the key is incomplete (the part after the "." is missing) or has expired |
| SiliconFlow returns 403 / code 30003 | the model is not available to your account; choose another model |
| The IDE (e.g. PyCharm) closes during step 01 | out of memory; in the run_study script change `--n_jobs 0` to a smaller number such as `--n_jobs 4` |
| No output for a long time in step 13 | timed-out requests are being retried; a progress line is printed every 10 requests; check the network with `check_api.py` |
| The third decimal differs slightly from the paper | PINN training differs slightly between machines (AUROC difference of 0.002 in our test), and repeated MLLM calls also vary slightly; use the shipped `data/` folder for identical numbers |

## Notes

- `data/` contains de-identified derived data (provenance and terms of use in `data/README.md`) but no raw images or masks. `.gitignore` excludes your own `results/`, NIfTI files and the API cache.
- Step 13 sends de-identified image panels and features to third-party APIs. Before running it, make sure the TCIA/BreastDCEDL data-use terms and your institution's rules permit this.
- Never commit API keys to GitHub.
