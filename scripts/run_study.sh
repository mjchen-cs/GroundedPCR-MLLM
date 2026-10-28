#!/usr/bin/env bash
# =============================================================================
# Reproduce the I-SPY2 MLLM study end to end (Linux / macOS).
#
#   Usage:  bash scripts/run_study.sh <stage>
#
#   Stages (run in this order; each one skips work whose outputs already exist):
#     features   steps 00-01  metadata + ROI / enhancement curves / heterogeneity (no GPU needed)
#     b1         steps 03,07,11  feature-set CV, locked B1 + single held-out evaluation, SHAP
#     export     step 12      cases.jsonl + 2x2 image panels for the MLLMs
#     pilot      step 13      20 development cases, 1 repeat, both providers (cost / format check)
#     query      step 13      all cases, both splits, both providers (needs API keys; paid)
#     followup   steps 21-22  counterfactual reference test + verify-and-revise loop (paid, ~1-2k requests)
#     evaluate   steps 14-18, 20, 23  metrics, tables, figures, secondary analyses (offline)
#     all        features + b1 + export + query + evaluate
#
#   Paths and models are read from environment variables (defaults below), e.g.
#     export DATA_DIR=/data/BreastDCEDL_ISPY2           # <pid>/dce, <pid>/mask  (TCIA package)
#     export BREASTDCEDL_REPO=/data/BreastDCEDL         # clone of github.com/naomifridman/BreastDCEDL
#     export OUT_DIR=results  PAPER_DIR=paper
#     export ZHIPU_API_KEY=...  SILICONFLOW_API_KEY=...
# =============================================================================
set -euo pipefail
PY=${PYTHON:-python}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGE=${1:-all}

DATA_DIR=${DATA_DIR:-}
BREASTDCEDL_REPO=${BREASTDCEDL_REPO:-}
need() { [ -n "${!1}" ] || { echo "Set $1 ($2) for stage '$STAGE'"; exit 2; }; }
OUT_DIR=${OUT_DIR:-results}
PAPER_DIR=${PAPER_DIR:-$OUT_DIR/paper}
ZHIPU_MODEL=${ZHIPU_MODEL:-glm-4.5v}
SF_MODEL=${SF_MODEL:-Qwen/Qwen3-VL-32B-Instruct}
REPEATS=${REPEATS:-2}
WORKERS=${WORKERS:-8}
PINN_DEVICE=${PINN_DEVICE:-cpu}
B1_CANDIDATES=${B1_CANDIDATES:-"main main_mp main_het main_mp_het"}
CV_PRESETS=${CV_PRESETS:-"clinical_mp clinical_mp_volume clinical_mp_volume_het main main_mp main_het main_mp_het"}
# Panel shown in Fig. 1 (any held-out case works; the paper used this one)
FIG1_PID=${FIG1_PID:-ACRIN-6698-102212}
FIG1_EXCERPT=${FIG1_EXCERPT:-"…triple-negative breast cancer with a large tumor volume of over 30,000 mm³ … estimated probability of a pathologic complete response is approximately 29%, which is below the threshold for likely pCR."}

META=$OUT_DIR/metadata/ISPY2/metadata.csv
R=$OUT_DIR/ispy2_roi
M=$OUT_DIR/mllm
B="--feature_csv $R/dynamics_feature_summary.csv --volume_csv $R/volume_summary.csv --tcc_csv $R/tcc_long.csv --metadata_csv $META --pinn_device $PINN_DEVICE"
cd "$REPO"

stage_features() {
  need DATA_DIR "BreastDCEDL_ISPY2 image folder"; need BREASTDCEDL_REPO "BreastDCEDL GitHub clone with the metadata CSVs"
  [ -s "$META" ] || $PY scripts/00_prepare_breastdcedl.py \
      --breastdcedl_metadata "$BREASTDCEDL_REPO/BreastDCEDL_metadata.csv" \
      --ispy2_clinical "$BREASTDCEDL_REPO/ISPY2/BreastDCEDL_spy2_tcia_metadata.csv" \
      --output_dir "$OUT_DIR/metadata"
  [ -s "$R/dynamics_feature_summary.csv" ] || $PY scripts/01_extract_roi_tcc.py \
      --full_root "$DATA_DIR" --mask_dir "$DATA_DIR" --metadata_csv "$META" --output_dir "$R" \
      --spacing_source metadata --slice_axis 0 --no-save_qc_plots --n_jobs 0
}

stage_b1() {
  for P in $CV_PRESETS; do
    [ -s "$OUT_DIR/cv/$P/xgboost_metrics.json" ] || \
      $PY scripts/03_train_xgboost.py $B --feature_mode "$P" --n_bootstrap 0 --output_dir "$OUT_DIR/cv/$P"
  done
  $PY scripts/11_compare_cv.py --runs_dir "$OUT_DIR/cv" --candidates $B1_CANDIDATES --paper_dir "$PAPER_DIR"
  BEST=$(cat "$OUT_DIR/cv/best_preset.txt")
  echo "B1 preset selected on development CV: $BEST"
  if [ ! -s "$OUT_DIR/b1/heldout_test_shap.csv" ]; then
    $PY scripts/03_train_xgboost.py $B --feature_mode "$BEST" --run_final_test --output_dir "$OUT_DIR/b1"
    $PY scripts/07_shap_analysis.py --model_dir "$OUT_DIR/b1/final_model" \
        --model_inputs_csv "$OUT_DIR/b1/final_development_model_inputs.csv" --output_dir "$OUT_DIR/b1/shap"
  fi
  # sensitivity runs for step 18 (development CV only)
  [ -s "$OUT_DIR/sens/spw1/xgboost_metrics.json" ]   || $PY scripts/03_train_xgboost.py $B --feature_mode "$BEST" --scale_pos_weight 1 --n_bootstrap 0 --output_dir "$OUT_DIR/sens/spw1"
  [ -s "$OUT_DIR/sens/depth3/xgboost_metrics.json" ] || $PY scripts/03_train_xgboost.py $B --feature_mode "$BEST" --max_depth 3 --n_bootstrap 0 --output_dir "$OUT_DIR/sens/depth3"
  [ -s "$OUT_DIR/sens/arm/xgboost_metrics.json" ]    || $PY scripts/03_train_xgboost.py $B --feature_mode main_mp_arm --n_bootstrap 0 --output_dir "$OUT_DIR/sens/arm"
}

stage_export() {
  need DATA_DIR "BreastDCEDL_ISPY2 image folder"
  [ -s "$M/cases.jsonl" ] || $PY scripts/12_export_mllm_inputs.py --data_root "$DATA_DIR" --metadata_csv "$META" \
      --xgb_dir "$OUT_DIR/b1" --output_dir "$M" --slice_axis 0
}

query() {  # $1 split, extra args
  local S=$1; shift
  $PY scripts/13_run_mllm.py --mllm_dir "$M" --provider zhipu --model "$ZHIPU_MODEL" --disable_thinking --split "$S" --workers "$WORKERS" "$@"
  $PY scripts/13_run_mllm.py --mllm_dir "$M" --provider siliconflow --model "$SF_MODEL" --split "$S" --workers "$WORKERS" "$@"
}
stage_followup() {
  $PY scripts/21_counterfactual.py --mllm_dir "$M" --provider zhipu --model "$ZHIPU_MODEL" --disable_thinking --workers "$WORKERS"
  $PY scripts/21_counterfactual.py --mllm_dir "$M" --provider siliconflow --model "$SF_MODEL" --workers "$WORKERS"
  $PY scripts/22_verify_revise.py --mllm_dir "$M" --provider zhipu --model "$ZHIPU_MODEL" --disable_thinking --workers "$WORKERS"
  $PY scripts/22_verify_revise.py --mllm_dir "$M" --provider siliconflow --model "$SF_MODEL" --workers "$WORKERS"
}
stage_pilot() { query development --limit 20 --repeats 1; echo "Pilot done: check $M/runs/*/*.csv (parse_ok, tokens)."; }
stage_query() { for S in development heldout_test; do query "$S" --repeats "$REPEATS"; done; }

stage_evaluate() {
  BEST=$(cat "$OUT_DIR/cv/best_preset.txt")
  $PY scripts/14_evaluate_mllm.py --mllm_dir "$M" --paper_dir "$PAPER_DIR" --b1_preset "$BEST"
  $PY scripts/15_plot_results.py "$M/evaluation" "$M/cases.jsonl" "$M/runs" "$PAPER_DIR/figures/fig_results"
  ZDIR=$(ls -d "$M"/runs/zhipu__* | head -1)
  $PY scripts/16_plot_workflow.py "$M/cases.jsonl" "$M/images/$FIG1_PID.png" "$R/tcc_long.csv" \
      "$ZDIR/M3_heldout_test.csv" "$FIG1_PID" "$PAPER_DIR/figures/fig1_workflow" --eval_dir "$M/evaluation" \
      --n_repeats "$REPEATS" --report_excerpt "$FIG1_EXCERPT"
  $PY scripts/17_extra_analyses.py --mllm_dir "$M" --b1_dir "$OUT_DIR/b1" --roi_dir "$R" --out_dir "$PAPER_DIR"
  SENS=("XGBoost without class weighting=$OUT_DIR/sens/spw1" "XGBoost, tree depth 3=$OUT_DIR/sens/depth3"
        "XGBoost + treatment arm (24 inputs)=$OUT_DIR/sens/arm")
  # optional: B1 re-trained in a second computing environment (shipped in data/sens/repro)
  [ -s "$OUT_DIR/sens/repro/xgboost_predictions.csv" ] && SENS+=("Re-run of B1 in a second environment=$OUT_DIR/sens/repro")
  $PY scripts/18_b1_sensitivity.py --b1_dir "$OUT_DIR/b1" --out_dir "$PAPER_DIR" --runs "${SENS[@]}"
  $PY scripts/20_agreement_analysis.py --mllm_dir "$M" --out_dir "$PAPER_DIR"
  if [ -d "$M/runs_cf" ] || [ -d "$M/runs_revise" ]; then
    $PY scripts/23_evaluate_followups.py --mllm_dir "$M" --out_dir "$PAPER_DIR"
  fi
  echo "Done. Tables, macros and figures are in $PAPER_DIR"
}

case "$STAGE" in
  features) stage_features ;;
  b1)       stage_b1 ;;
  export)   stage_export ;;
  pilot)    stage_pilot ;;
  query)    stage_query ;;
  followup) stage_followup ;;
  evaluate) stage_evaluate ;;
  all)      stage_features; stage_b1; stage_export; stage_query; stage_followup; stage_evaluate ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
