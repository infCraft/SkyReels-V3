export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/root/autodl-tmp/SkyReels-V3-A2V-19B"
PROMPT="A woman is giving a speech. She is confident, poised, and joyful. Use a static shot."
INPUT_IMAGE="/root/SkyReels-V3/input_image/woman.JPEG"
INPUT_AUDIO="/root/SkyReels-V3/input_audio/woman_speech.mp3"

python generate_video.py \
    --task_type talking_avatar \
    --model_id $MODEL_PATH \
    --prompt "$PROMPT" \
    --seed 42 \
    --low_vram \
    --input_image "$INPUT_IMAGE" \
    --input_audio "$INPUT_AUDIO"
