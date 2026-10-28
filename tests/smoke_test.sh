#!/usr/bin/env bash
# Synthetic end-to-end check of steps 1-10 (numbers are meaningless).
set -euo pipefail
PY=${PYTHON:-python}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TMP=$(mktemp -d)
echo "working in $TMP"
$PY "$REPO/tests/make_synthetic_cohort.py" "$TMP/ISPY2" 80 1 6 9 20
$PY "$REPO/tests/make_synthetic_cohort.py" "$TMP/ISPY1" 30 2 5 8 0
$PY "$REPO/tests/make_synthetic_cohort.py" "$TMP/Duke"  30 3 4 6 0
$PY -c "import torch; from torchvision.models import resnet50; torch.save(resnet50(weights=None).state_dict(), '$TMP/random_resnet50.pt')"
for C in ISPY2 ISPY1 Duke; do
  mv "$TMP/$C/masks" "$TMP/${C}_masks"; mv "$TMP/$C/data"/* "$TMP/$C/"; rmdir "$TMP/$C/data"
done
sed -e "s#^ISPY2_ROOT=.*#ISPY2_ROOT=$TMP/ISPY2;  ISPY2_MASKS=$TMP/ISPY2_masks#" \
    -e "s#^ISPY1_ROOT=.*#ISPY1_ROOT=$TMP/ISPY1;  ISPY1_MASKS=$TMP/ISPY1_masks#" \
    -e "s#^DUKE_ROOT=.*#DUKE_ROOT=$TMP/Duke;    DUKE_MASKS=$TMP/Duke_masks#" \
    -e "s#^RADIMAGENET_WEIGHTS=.*#RADIMAGENET_WEIGHTS=$TMP/random_resnet50.pt#" \
    -e "s#^HERE=.*#HERE=$REPO/scripts#" \
    "$REPO/scripts/run_full_pipeline.sh" > "$TMP/run.sh"
OUT="$TMP/out" EXTRA="--pinn_epochs 15 --n_bootstrap 200" bash "$TMP/run.sh"
for f in main/final_model/model_card.json external/Duke/metrics_Duke.json stats/table_ii_performance.csv \
         stats/table_iv_subtypes.csv stats/fig4_calibration_dca.png shap/fig5a_shap_summary.png \
         ablation/table_iii_ablation.csv sensitivity/fig5b_sensitivity.png; do
  test -s "$TMP/out/$f" || { echo "MISSING $f"; exit 1; }
done
echo "SMOKE TEST PASSED ($TMP/out)"
