#!/bin/bash
# =============================================================================
# Batch evaluation: FVD / SSIM / PSNR / LPIPS
# =============================================================================
# Usage:  bash scripts/run_eval_batch_metrics.sh
#
# Evaluates all experiment result sets defined below against their manifests.
# Uses the 'eval' conda environment.
# =============================================================================
set -euo pipefail

cd /root/SkyReels-V3

EVAL_SCRIPT="evaluate/evaluate_batch_metrics.py"
EXPERIMENTS_ROOT="/root/autodl-fs/experiments"

echo "=========================================="
echo " Batch Metrics Evaluation"
echo "=========================================="

python "${EVAL_SCRIPT}" \
    --manifest "${EXPERIMENTS_ROOT}/evaluation_manifest.json" \
    --gen_dir  "${EXPERIMENTS_ROOT}/baseline_results" \
    --gen_pattern "{id}_42_with_audio.mp4" \
    --output_dir "${EXPERIMENTS_ROOT}" \
    --output_prefix baseline \
    --metrics fvd ssim psnr lpips \
    --fvd_resize 224 224 \
    --fvd_method styleganv \
    --device cuda

