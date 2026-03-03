#!/bin/bash
# ============================================================
# Phase 2: Run sensitivity probe on 10 calibration videos.
# Probe mode captures per-block redundancy metrics (cos_sim,
# res_mag) for each (step, block) pair in segment 0.
# CFPS is computed offline by tools/analyze_probe.py with configurable alpha.
# Parameters match run_talking_avatar_single.sh / run_baseline.sh:
#   sampling_steps=4, text_guide_scale=1.0, audio_guide_scale=1.0, seed=42
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/root/autodl-tmp/SkyReels-V3-A2V-19B"
MANIFEST="/root/autodl-fs/experiments/calibration_manifest.json"
OUTPUT_DIR="/root/autodl-fs/experiments/probe_results_raw"

mkdir -p "$OUTPUT_DIR"

python generate_video.py \
    --task_type talking_avatar \
    --model_id "$MODEL_PATH" \
    --prompt "A high-quality video of a person talking to the camera, natural lighting, realistic." \
    --seed 42 \
    --manifest_json "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --probe_mode \
    2>&1 | tee logs/run_probe.log

echo "=========================================="
echo "Probe inference complete."
echo "Results saved to: $OUTPUT_DIR"
echo "Check logs/run_probe.log for details."
echo "=========================================="
