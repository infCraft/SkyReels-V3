export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/root/autodl-tmp/SkyReels-V3-A2V-19B"
MANIFEST_JSON="/root/autodl-fs/experiments/evaluation_manifest_2_video.json"
OUTPUT_DIR="/root/autodl-fs/experiments/sanity_check"

python generate_video.py \
    --task_type talking_avatar \
    --model_id "$MODEL_PATH" \
    --seed 42 \
    --manifest_json "$MANIFEST_JSON" \
    --output_dir "$OUTPUT_DIR"
