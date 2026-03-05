#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 子模块级能量探测 – 批量推理脚本
#
# 使用 calibration_manifest.json 中的 10 个 HDTF 校准样本，
# 对 SkyReels-V3 Talking Avatar 模型执行子模块级能量探测。
#
# 输出：
#   /root/autodl-fs/experiments/submodule_probe_raw/
#     ├── hdtf_0000_submodule_probe.json
#     ├── hdtf_0001_submodule_probe.json
#     ├── ...
#     ├── hdtf_0009_submodule_probe.json
#     └── batch_timing.json
# ─────────────────────────────────────────────────────────────
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/root/autodl-tmp/SkyReels-V3-A2V-19B"
MANIFEST="/root/autodl-fs/experiments/calibration_manifest.json"
OUTPUT_DIR="/root/autodl-fs/experiments/submodule_probe_raw"
SEED=42

echo "============================================"
echo " SkyReels-V3 子模块级能量探测"
echo " Model:    ${MODEL_PATH}"
echo " Manifest: ${MANIFEST}"
echo " Output:   ${OUTPUT_DIR}"
echo " Seed:     ${SEED}"
echo "============================================"

cd /root/SkyReels-V3

conda run -n sky --no-capture-output python generate_video_submodule_probe.py \
    --model_id "${MODEL_PATH}" \
    --calibration_manifest "${MANIFEST}" \
    --probe_output_dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --resolution 720P \
    --save_video

echo ""
echo "探测完成！原始数据已保存至 ${OUTPUT_DIR}"
echo "下一步：运行 python tools/analyze_submodule_probe.py 进行分析。"
