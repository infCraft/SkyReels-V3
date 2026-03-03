#!/bin/bash
# ============================================================
# Run baseline inference on the evaluation set (40 videos).
# Parameters match run_talking_avatar_single.sh:
#   sampling_steps=4, text_guide_scale=1.0, audio_guide_scale=1.0, seed=42
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/root/autodl-tmp/SkyReels-V3-A2V-19B"
MANIFEST="/root/autodl-fs/experiments/evaluation_manifest.json"
OUTPUT_DIR="/root/autodl-fs/experiments/baseline_results"

python generate_video.py \
    --task_type talking_avatar \
    --model_id "$MODEL_PATH" \
    --prompt "A high-quality video of a person talking to the camera, natural lighting, realistic." \
    --seed 42 \
    --manifest_json "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    2>&1 | tee run_baseline.log
