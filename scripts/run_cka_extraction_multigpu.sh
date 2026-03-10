#!/bin/bash
# CKA 多 GPU 数据并行提取脚本
# 4 卡并行处理两个标定数据集 (HDTF + CelebVHQ)
#
# 用法: bash scripts/run_cka_extraction_multigpu.sh
#
# 架构: torch.multiprocessing.spawn，每个 GPU 独立加载完整模型，
#       处理交错分配的样本 (manifest[rank::num_gpus])

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

NUM_GPUS=4
MODEL_PATH="/root/autodl-tmp/SkyReels-V3-A2V-19B"
NUM_SUBSAMPLE_TOKENS=8192

cd /root/SkyReels-V3-cka

echo "============================================"
echo "  CKA Extraction - HDTF (${NUM_GPUS} GPUs)"
echo "============================================"

python scripts/cka_extract.py \
    --model_id "$MODEL_PATH" \
    --manifest_json /root/autodl-fs/experiments/calibration_manifest_hdtf.json \
    --output_dir /root/autodl-fs/experiments/cka_analysis/hdtf \
    --num_subsample_tokens $NUM_SUBSAMPLE_TOKENS \
    --seed 42 \
    --num_gpus $NUM_GPUS

echo "============================================"
echo "  CKA Extraction - CelebVHQ (${NUM_GPUS} GPUs)"
echo "============================================"

python scripts/cka_extract.py \
    --model_id "$MODEL_PATH" \
    --manifest_json /root/autodl-fs/experiments/calibration_manifest_celebvhq.json \
    --output_dir /root/autodl-fs/experiments/cka_analysis/celebvhq \
    --num_subsample_tokens $NUM_SUBSAMPLE_TOKENS \
    --seed 42 \
    --num_gpus $NUM_GPUS

echo "============================================"
echo "  All CKA extractions complete!"
echo "============================================"
