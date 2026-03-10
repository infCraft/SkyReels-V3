#!/bin/bash
# CKA 激活提取脚本启动器
# 用法: bash scripts/run_cka_extraction.sh

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/root/autodl-tmp/SkyReels-V3-A2V-19B"
MANIFEST_JSON="/root/autodl-fs/experiments/calibration_manifest.json"
OUTPUT_DIR="/root/autodl-fs/experiments/cka_analysis"

cd /root/SkyReels-V3-cka

python scripts/cka_extract.py \
    --model_id "$MODEL_PATH" \
    --manifest_json "$MANIFEST_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --num_subsample_tokens 8192 \
    --seed 42 \
    --num_gpus 1
