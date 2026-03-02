#!/bin/bash
# ──────────────────────────────────────────────────────────────────
# HDTF RD 数据集下载脚本（从 YouTube 下载原始视频）
#
# 从 HDTF 官方 GitHub 仓库获取 RD (Radio) 部分的 YouTube 链接，
# 使用 yt-dlp 下载原始 720p/1080p 视频，然后根据标注时间戳切片。
#
# 前置要求：
#   - yt-dlp: pip install yt-dlp
#   - ffmpeg: 系统已安装
#   - 网络能访问 YouTube（需 VPN）
#
# 使用说明：
#   bash tools/download_hdtf.sh          # 下载全部 ~59 个视频
#   bash tools/download_hdtf.sh 5        # 仅下载前 5 个（测试）
# ──────────────────────────────────────────────────────────────────

set -euo pipefail

MAX_VIDEOS="${1:-0}"  # 第一个参数：最多下载视频数（0=全部）
OUTPUT_DIR="/root/autodl-fs/experiments/hdtf_raw"

echo "======================================"
echo "HDTF RD Dataset Downloader (YouTube)"
echo "  Output: ${OUTPUT_DIR}"
echo "  Max videos: ${MAX_VIDEOS} (0=all)"
echo "======================================"

# 检查依赖
command -v yt-dlp &>/dev/null || { echo "ERROR: yt-dlp not found. Run: pip install yt-dlp"; exit 1; }
command -v ffmpeg &>/dev/null || { echo "ERROR: ffmpeg not found."; exit 1; }

# 运行下载脚本
python tools/download_hdtf_youtube.py \
    --output_dir "${OUTPUT_DIR}" \
    --max_videos "${MAX_VIDEOS}"

echo ""
echo "Download complete!"
echo ""

# 统计下载结果
VIDEO_COUNT=$(find "${OUTPUT_DIR}/clips" -name "*.mp4" -type f 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh "${OUTPUT_DIR}" 2>/dev/null | cut -f1)

echo "======================================"
echo "Download Summary"
echo "  Clips generated: ${VIDEO_COUNT}"
echo "  Total size: ${TOTAL_SIZE}"
echo "  Location: ${OUTPUT_DIR}"
echo "======================================"
echo ""
echo "Next step: Run the preparation script:"
echo "  python tools/prepare_hdtf.py \\"
echo "      --raw_dir ${OUTPUT_DIR}/clips \\"
echo "      --output_dir /root/autodl-fs/experiments/hdtf_processed \\"
echo "      --experiment_dir /root/autodl-fs/experiments"
