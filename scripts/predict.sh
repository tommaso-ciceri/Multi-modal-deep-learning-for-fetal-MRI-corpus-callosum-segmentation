#!/usr/bin/env bash
# Stage 1: CC segmentation
# 
# Usage: 
#   ./scripts/predict.sh <t2w|fa|dual> INPUT_FOLDER OUTPUT_FOLDER

set -euo pipefail

if [[ $# -ne 3 ]]; then
    sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi

MODALITY=$1
INPUT_FOLDER=$2
OUTPUT_FOLDER=$3

case "$MODALITY" in
    t2w)  DATASET=Dataset030_CC_T2w ;;
    fa)   DATASET=Dataset031_CC_FA ;;
    dual) DATASET=Dataset032_CC_T2w_FA ;;
    *)    echo "Unknown modality: $MODALITY (expected t2w, fa or dual)" >&2; exit 1 ;;
esac

nnUNetv2_predict \
    -d "$DATASET" \
    -i "$INPUT_FOLDER" \
    -o "$OUTPUT_FOLDER" \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer \
    -c 3d_fullres \
    -p nnUNetPlans