#!/usr/bin/env bash
# Offline end-to-end check of the MLLM study (steps 1, 3, 7, 11-18) on a small SYNTHETIC
# cohort with the deterministic "mock" model: no data download, no API key, no cost.
# All numbers it produces are meaningless; it only verifies that every step runs and
# writes its outputs. Takes a few minutes on a laptop CPU.
set -euo pipefail
PY=${PYTHON:-python}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TMP=${TMP_DIR:-$(mktemp -d)}
echo "working in $TMP"

# synthetic cohort: 90 cases, 30 of them held out
$PY "$REPO/tests/make_synthetic_cohort.py" "$TMP/ISPY2" 90 7 6 9 30
mv "$TMP/ISPY2/masks" "$TMP/masks"; mv "$TMP/ISPY2/data"/* "$TMP/ISPY2/"; rmdir "$TMP/ISPY2/data"
DATA=$TMP/ISPY2; META=$DATA/metadata.csv; OUT=$TMP/out; PAPER=$TMP/paper

$PY "$REPO/scripts/01_extract_roi_tcc.py" --full_root "$DATA" --mask_dir "$TMP/masks" --metadata_csv "$META" \
    --output_dir "$OUT/roi" --no-save_qc_plots
R=$OUT/roi
B="--feature_csv $R/dynamics_feature_summary.csv --volume_csv $R/volume_summary.csv --tcc_csv $R/tcc_long.csv --metadata_csv $META --pinn_device cpu --pinn_epochs 15"

# feature-set comparison on development CV (synthetic data has no MammaPrint, so "main" stands in for B1)
for P in clinical_only main; do
  $PY "$REPO/scripts/03_train_xgboost.py" $B --feature_mode $P --n_bootstrap 0 --output_dir "$OUT/cv/$P"
done
$PY "$REPO/scripts/11_compare_cv.py" --runs_dir "$OUT/cv" --candidates main --paper_dir "$PAPER"
$PY "$REPO/scripts/03_train_xgboost.py" $B --feature_mode main --n_bootstrap 100 --run_final_test --output_dir "$OUT/b1"
$PY "$REPO/scripts/07_shap_analysis.py" --model_dir "$OUT/b1/final_model" \
    --model_inputs_csv "$OUT/b1/final_development_model_inputs.csv" --output_dir "$OUT/b1/shap"
$PY "$REPO/scripts/03_train_xgboost.py" $B --feature_mode main --scale_pos_weight 1 --n_bootstrap 0 --output_dir "$OUT/sens/spw1"

# MLLM inputs, offline queries, evaluation
$PY "$REPO/scripts/12_export_mllm_inputs.py" --data_root "$DATA" --mask_dir "$TMP/masks" --metadata_csv "$META" \
    --xgb_dir "$OUT/b1" --output_dir "$OUT/mllm" --slice_axis 2
for S in development heldout_test; do
  $PY "$REPO/scripts/13_run_mllm.py" --mllm_dir "$OUT/mllm" --provider mock --model mock-vl --split $S --repeats 2 --workers 2
done
$PY "$REPO/scripts/14_evaluate_mllm.py" --mllm_dir "$OUT/mllm" --paper_dir "$PAPER" --b1_preset main --n_bootstrap 100
$PY "$REPO/scripts/15_plot_results.py" "$OUT/mllm/evaluation" "$OUT/mllm/cases.jsonl" "$OUT/mllm/runs" "$PAPER/figures/fig_results" \
    --providers mock
PID=$(head -1 "$OUT/mllm/cases.jsonl" | $PY -c "import json,sys; print(json.loads(sys.stdin.read())['pid'])")
$PY "$REPO/scripts/16_plot_workflow.py" "$OUT/mllm/cases.jsonl" "$OUT/mllm/images/$PID.png" "$R/tcc_long.csv" \
    "$OUT/mllm/runs/mock__mock-vl/M3_development.csv" "$PID" "$PAPER/figures/fig1_workflow"
$PY "$REPO/scripts/17_extra_analyses.py" --mllm_dir "$OUT/mllm" --b1_dir "$OUT/b1" --roi_dir "$R" --out_dir "$PAPER" \
    --providers mock
$PY "$REPO/scripts/18_b1_sensitivity.py" --b1_dir "$OUT/b1" --runs "No class weighting=$OUT/sens/spw1" --out_dir "$PAPER"
$PY "$REPO/scripts/20_agreement_analysis.py" --mllm_dir "$OUT/mllm" --out_dir "$PAPER" --providers mock
$PY "$REPO/scripts/21_counterfactual.py" --mllm_dir "$OUT/mllm" --provider mock --model mock-vl --workers 2
$PY "$REPO/scripts/22_verify_revise.py" --mllm_dir "$OUT/mllm" --provider mock --model mock-vl --workers 2
$PY "$REPO/scripts/23_evaluate_followups.py" --mllm_dir "$OUT/mllm" --out_dir "$PAPER"

for f in tables/table_cv_rows.tex tables/table_prediction_rows.tex tables/table_faithfulness_rows.tex results_macros.tex \
         figures/fig_results.pdf figures/fig1_workflow.pdf figures/fig_calib_dca.pdf figures/fig_shap_citation.pdf \
         tables/table_subtype_rows.tex tables/table_b1_sensitivity_rows.tex tables/table_agreement_rows.tex \
         tables/table_cf_rows.tex tables/table_revise_rows.tex; do
  test -s "$PAPER/$f" || { echo "MISSING $f"; exit 1; }
done
echo "MLLM PIPELINE TEST PASSED ($PAPER)"
