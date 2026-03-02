export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/root/autodl-tmp/SkyReels-V3-A2V-19B"
PROMPT="A woman is giving a speech. She is confident, poised, and joyful. Use a static shot."
INPUT_IMAGE="/root/SkyReels-V3/input_image/woman.JPEG"
INPUT_AUDIO="/root/SkyReels-V3/input_audio/woman_5s.mp3"

# 我删除了 --low_vram 参数，因为我现在使用的是A800-80GB显卡，应该不需要这个参数了。
python generate_video.py \
    --task_type talking_avatar \
    --model_id $MODEL_PATH \
    --prompt "$PROMPT" \
    --seed 42 \
    --input_image "$INPUT_IMAGE" \
    --input_audio "$INPUT_AUDIO" \
    2>&1 | tee run.log
