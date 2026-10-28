#!/usr/bin/env bash
# End-to-end run of steps 1-10. Edit the paths below.
set -euo pipefail
PY=${PYTHON:-python}
HERE="$(cd "$(dirname "$0")" && pwd)"

ISPY2_ROOT=/path/to/ISPY2;  ISPY2_MASKS=/path/to/ISPY2_masks
ISPY1_ROOT=/path/to/ISPY1;  ISPY1_MASKS=/path/to/ISPY1_masks
DUKE_ROOT=/path/to/Duke;    DUKE_MASKS=/path/to/Duke_masks
RADIMAGENET_WEIGHTS=""      # optional: RadImageNet ResNet50 .pt for the visual branch
OUT=${OUT:-out}
EXTRA=${EXTRA:-}            # extra args for step 3, e.g. "--pinn_epochs 50" for a quick test

roi() {  # $1 root  $2 masks  $3 out  [extra]
  $PY "$HERE/01_extract_roi_tcc.py" --full_root "$1" --metadata_csv "$1/metadata.csv" --mask_dir "$2" \
      --output_dir "$3" --no-save_qc_plots "${@:4}"
}
roi "$ISPY2_ROOT" "$ISPY2_MASKS" "$OUT/ispy2_roi"
roi "$ISPY1_ROOT" "$ISPY1_MASKS" "$OUT/ispy1_roi"
roi "$DUKE_ROOT"  "$DUKE_MASKS"  "$OUT/duke_roi"
roi "$ISPY2_ROOT" "$ISPY2_MASKS" "$OUT/ispy2_roi_core10" --core_fraction 0.10
roi "$ISPY2_ROOT" "$ISPY2_MASKS" "$OUT/ispy2_roi_core20" --core_fraction 0.20

R=$OUT/ispy2_roi
$PY "$HERE/03_train_xgboost.py" --feature_csv $R/dynamics_feature_summary.csv --volume_csv $R/volume_summary.csv \
    --tcc_csv $R/tcc_long.csv --metadata_csv "$ISPY2_ROOT/metadata.csv" --output_dir $OUT/main \
    --feature_mode main --run_final_test $EXTRA

for C in ISPY1:ispy1_roi:$ISPY1_ROOT Duke:duke_roi:$DUKE_ROOT; do
  IFS=: read NAME DIR ROOT <<< "$C"
  $PY "$HERE/05_external_validation.py" --model_dir $OUT/main/final_model --cohort_name $NAME \
      --feature_csv $OUT/$DIR/dynamics_feature_summary.csv --volume_csv $OUT/$DIR/volume_summary.csv \
      --tcc_csv $OUT/$DIR/tcc_long.csv --metadata_csv "$ROOT/metadata.csv" --output_dir $OUT/external/$NAME
done

$PY "$HERE/06_cohort_statistics.py" --model_dir $OUT/main/final_model --output_dir $OUT/stats \
    --cohort "Development=$OUT/main/xgboost_predictions.csv::development_oof" \
    --cohort "Held-out test=$OUT/main/xgboost_predictions.csv::heldout_test" \
    --cohort "I-SPY 1=$OUT/external/ISPY1/predictions_ISPY1.csv" \
    --cohort "Duke=$OUT/external/Duke/predictions_Duke.csv" \
    --development_name Development --cv_fold_metrics_csv $OUT/main/xgboost_cv_fold_metrics.csv

$PY "$HERE/07_shap_analysis.py" --model_dir $OUT/main/final_model \
    --model_inputs_csv $OUT/main/final_development_model_inputs.csv --output_dir $OUT/shap

VIS=()
if [[ -n "$RADIMAGENET_WEIGHTS" ]]; then
  $PY "$HERE/08_extract_visual_embeddings.py" --full_root "$ISPY2_ROOT" --metadata_csv "$ISPY2_ROOT/metadata.csv" \
      --mask_dir "$ISPY2_MASKS" --weights "$RADIMAGENET_WEIGHTS" --output_csv $OUT/vis/ispy2_vis.csv
  VIS=(--visual_embedding_csv $OUT/vis/ispy2_vis.csv)
fi
$PY "$HERE/09_run_ablations.py" --feature_csv $R/dynamics_feature_summary.csv --volume_csv $R/volume_summary.csv \
    --tcc_csv $R/tcc_long.csv --metadata_csv "$ISPY2_ROOT/metadata.csv" --output_dir $OUT/ablation "${VIS[@]}" -- $EXTRA

$PY "$HERE/10_sensitivity_analysis.py" --model_dir $OUT/main/final_model --roi_dir $R \
    --metadata_csv "$ISPY2_ROOT/metadata.csv" --output_dir $OUT/sensitivity \
    --roi_variant core10=$OUT/ispy2_roi_core10 --roi_variant core20=$OUT/ispy2_roi_core20
