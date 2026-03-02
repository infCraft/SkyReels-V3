#!/usr/bin/env python3
"""
HDTF RD 数据集下载与切片脚本

从 HDTF GitHub 仓库获取 RD (Radio) 部分的 YouTube 链接和标注时间戳，
使用 yt-dlp 下载原始视频，然后根据标注时间戳切片。

流程：
1. 从 GitHub 下载 RD_video_url.txt 和 RD_annotion_time.txt
2. 解析 YouTube 链接和标注时间戳
3. 用 yt-dlp 下载原始视频（最高画质 720p/1080p）
4. 根据标注时间戳用 ffmpeg 切片，每个片段为一个独立的 talking face clip
5. 切片后的视频存入指定目录，供 prepare_hdtf.py 进一步处理

用法：
  python tools/download_hdtf_youtube.py \\
      --output_dir /root/autodl-fs/experiments/hdtf_raw \\
      --max_videos 0

  # 仅下载前5个视频（测试）
  python tools/download_hdtf_youtube.py \\
      --output_dir /root/autodl-fs/experiments/hdtf_raw \\
      --max_videos 5
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# GitHub raw URLs
GITHUB_BASE = "https://raw.githubusercontent.com/MRzzm/HDTF/main/HDTF_dataset"
RD_VIDEO_URL = f"{GITHUB_BASE}/RD_video_url.txt"
RD_ANNOTATION_TIME = f"{GITHUB_BASE}/RD_annotion_time.txt"


# ─────────────────────────────────────────────────────────────────────
# 解析 HDTF 标注文件
# ─────────────────────────────────────────────────────────────────────

def download_text(url: str) -> str:
    """下载文本文件内容"""
    logger.info(f"Downloading: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        sys.exit(1)


def parse_video_urls(text: str) -> dict[str, str]:
    """
    解析 RD_video_url.txt

    格式不太规整，有两种模式：
    - 单行: "Radio1 https://www.youtube.com/watch?v=xxx"
    - 双行: "Radio2\nhttps://www.youtube.com/watch?v=xxx"

    返回 {video_name: youtube_url}
    """
    result = {}
    lines = text.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 尝试在当前行找到 name 和 URL
        # 模式1: "RadioXX https://..."
        match = re.match(r'^(Radio\d+)\s+(https?://\S+)', line)
        if match:
            name = match.group(1)
            url = match.group(2)
            result[name] = url
            i += 1
            continue

        # 模式2: "RadioXX" 在一行，URL 在下一行
        match_name = re.match(r'^(Radio\d+)\s*$', line)
        if match_name and i + 1 < len(lines):
            name = match_name.group(1)
            next_line = lines[i + 1].strip()
            if next_line.startswith("http"):
                result[name] = next_line
                i += 2
                continue

        i += 1

    return result


def parse_annotation_times(text: str) -> dict[str, list[tuple[str, str]]]:
    """
    解析 RD_annotion_time.txt

    格式：
    - "Radio1.mp4 00:20-01:35"
    - "Radio34.mp4 00:02-00:30 00:40-00:50 01:00-01:20 ..."（多个片段）
    - 有些在两行之间换行

    返回 {video_name: [(start, end), ...]}
    """
    result = {}
    # 先将换行合并，使得每个 RadioXX.mp4 开头的内容成为完整的一行
    text = text.strip()
    # 将非 Radio 开头的行合并到前一行
    merged_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r'^Radio\d+\.mp4', stripped):
            merged_lines.append(stripped)
        else:
            # 属于上一行的延续
            if merged_lines:
                merged_lines[-1] += " " + stripped
            else:
                merged_lines.append(stripped)

    for line in merged_lines:
        # 解析 "RadioXX.mp4 HH:MM:SS-HH:MM:SS HH:MM:SS-HH:MM:SS ..."
        parts = line.split()
        if not parts:
            continue

        video_file = parts[0]  # e.g. "Radio1.mp4"
        video_name = video_file.replace(".mp4", "")  # e.g. "Radio1"

        timestamps = []
        for part in parts[1:]:
            # 匹配 "MM:SS-MM:SS" 或 "HH:MM:SS-HH:MM:SS"
            match = re.match(r'^(\d+:\d+(?::\d+)?)-(\d+:\d+(?::\d+)?)$', part)
            if match:
                timestamps.append((match.group(1), match.group(2)))

        if timestamps:
            result[video_name] = timestamps

    return result


def time_to_seconds(t: str) -> float:
    """将 MM:SS 或 HH:MM:SS 转为秒数"""
    parts = t.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    else:
        raise ValueError(f"Invalid time format: {t}")


# ─────────────────────────────────────────────────────────────────────
# 下载与切片
# ─────────────────────────────────────────────────────────────────────

def download_youtube_video(
    url: str,
    output_path: str,
    video_name: str,
) -> bool:
    """
    使用 yt-dlp 下载 YouTube 视频。
    下载最佳画质 (720p/1080p)，输出为 mp4。
    """
    # 清理 URL（去掉 &t=6s 等参数）
    url = re.sub(r'&t=\d+s?', '', url)

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_path,
        "--no-overwrites",
        "--retries", "3",
        "--socket-timeout", "30",
        url,
    ]
    try:
        logger.info(f"  Downloading {video_name} from {url}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 分钟超时
        )
        if result.returncode != 0:
            # 检查是否已存在
            if "has already been downloaded" in result.stdout or os.path.exists(output_path):
                logger.info(f"  {video_name}: already downloaded")
                return True
            logger.warning(f"  yt-dlp failed for {video_name}: {result.stderr[:500]}")
            return False
        return os.path.exists(output_path)
    except subprocess.TimeoutExpired:
        logger.warning(f"  Download timeout for {video_name}")
        return False


def split_video_by_timestamps(
    input_path: str,
    output_dir: str,
    video_name: str,
    timestamps: list[tuple[str, str]],
) -> list[str]:
    """
    根据标注时间戳用 ffmpeg 切片。

    返回成功生成的切片文件路径列表。
    """
    output_files = []
    for clip_idx, (start, end) in enumerate(timestamps):
        start_sec = time_to_seconds(start)
        end_sec = time_to_seconds(end)
        duration = end_sec - start_sec

        if duration < 3:
            logger.warning(
                f"  Skipping {video_name} clip {clip_idx}: too short ({duration:.1f}s)"
            )
            continue

        clip_name = f"{video_name}_{clip_idx}.mp4"
        output_path = os.path.join(output_dir, clip_name)

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"  Clip already exists: {clip_name}")
            output_files.append(output_path)
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_sec),
            "-i", str(input_path),
            "-t", str(duration),
            "-c", "copy",  # 无需重编码，直接拷贝流
            "-avoid_negative_ts", "make_zero",
            "-loglevel", "error",
            output_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                output_files.append(output_path)
                logger.info(
                    f"  Clip: {clip_name} ({start}-{end}, {duration:.0f}s)"
                )
            else:
                logger.warning(
                    f"  Failed to create clip {clip_name}: {result.stderr[:200]}"
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"  Timeout creating clip {clip_name}")

    return output_files


# ─────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HDTF RD 数据集 YouTube 下载与切片",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-fs/experiments/hdtf_raw",
        help="输出目录（原始视频和切片都存在此目录下）",
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=0,
        help="最多下载的视频数量（0=全部）。用于测试",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="跳过下载步骤，仅做切片（假设原始视频已存在）",
    )
    parser.add_argument(
        "--url_file",
        type=str,
        default="",
        help="本地 RD_video_url.txt 路径（如已下载）。为空则从 GitHub 下载",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="",
        help="本地 RD_annotion_time.txt 路径（如已下载）。为空则从 GitHub 下载",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_youtube"
    clips_dir = output_dir / "clips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 获取标注文件 ──
    if args.url_file and os.path.exists(args.url_file):
        with open(args.url_file) as f:
            url_text = f.read()
    else:
        url_text = download_text(RD_VIDEO_URL)
        # 保存到本地以便后续使用
        local_url_file = output_dir / "RD_video_url.txt"
        with open(local_url_file, "w") as f:
            f.write(url_text)
        logger.info(f"Saved URL file to {local_url_file}")

    if args.annotation_file and os.path.exists(args.annotation_file):
        with open(args.annotation_file) as f:
            ann_text = f.read()
    else:
        ann_text = download_text(RD_ANNOTATION_TIME)
        local_ann_file = output_dir / "RD_annotion_time.txt"
        with open(local_ann_file, "w") as f:
            f.write(ann_text)
        logger.info(f"Saved annotation file to {local_ann_file}")

    # ── 2. 解析 ──
    video_urls = parse_video_urls(url_text)
    annotations = parse_annotation_times(ann_text)

    logger.info(f"Parsed {len(video_urls)} video URLs")
    logger.info(f"Parsed {len(annotations)} annotation entries")

    # 找出同时有 URL 和标注的视频
    common_videos = sorted(set(video_urls.keys()) & set(annotations.keys()))
    logger.info(f"Videos with both URL and annotation: {len(common_videos)}")

    if args.max_videos > 0:
        common_videos = common_videos[:args.max_videos]
        logger.info(f"Limited to first {args.max_videos} videos")

    # ── 3. 下载与切片 ──
    total_clips = 0
    failed_downloads = []
    successful_videos = []

    for idx, video_name in enumerate(common_videos):
        url = video_urls[video_name]
        timestamps = annotations[video_name]

        logger.info(
            f"\n[{idx+1}/{len(common_videos)}] {video_name}: "
            f"{len(timestamps)} clip(s), URL={url[:60]}..."
        )

        raw_video_path = raw_dir / f"{video_name}.mp4"

        # 下载
        if not args.skip_download:
            if raw_video_path.exists() and raw_video_path.stat().st_size > 0:
                logger.info(f"  Raw video already exists: {raw_video_path}")
            else:
                success = download_youtube_video(
                    url, str(raw_video_path), video_name
                )
                if not success:
                    failed_downloads.append(video_name)
                    logger.warning(f"  SKIP: Download failed for {video_name}")
                    continue

        if not raw_video_path.exists():
            logger.warning(f"  SKIP: Raw video not found: {raw_video_path}")
            failed_downloads.append(video_name)
            continue

        # 切片
        clip_files = split_video_by_timestamps(
            str(raw_video_path),
            str(clips_dir),
            video_name,
            timestamps,
        )
        total_clips += len(clip_files)
        if clip_files:
            successful_videos.append(video_name)

    # ── 4. 汇总 ──
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOAD & SPLIT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total videos attempted: {len(common_videos)}")
    logger.info(f"  Successful videos:      {len(successful_videos)}")
    logger.info(f"  Failed downloads:       {len(failed_downloads)}")
    logger.info(f"  Total clips generated:  {total_clips}")
    logger.info(f"  Raw videos dir:         {raw_dir}")
    logger.info(f"  Clips dir:              {clips_dir}")

    if failed_downloads:
        logger.info(f"  Failed videos: {', '.join(failed_downloads[:10])}")
        if len(failed_downloads) > 10:
            logger.info(f"    ... and {len(failed_downloads) - 10} more")

    # 保存下载状态
    status = {
        "total_attempted": len(common_videos),
        "successful": successful_videos,
        "failed": failed_downloads,
        "total_clips": total_clips,
        "clips_dir": str(clips_dir),
    }
    status_file = output_dir / "download_status.json"
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    logger.info(f"  Status saved to: {status_file}")

    logger.info("=" * 60)
    logger.info(
        f"\nNext step: Run prepare_hdtf.py to sample and process clips:\n"
        f"  python tools/prepare_hdtf.py \\\n"
        f"      --raw_dir {clips_dir} \\\n"
        f"      --output_dir /root/autodl-fs/experiments/hdtf_processed \\\n"
        f"      --experiment_dir /root/autodl-fs/experiments"
    )


if __name__ == "__main__":
    main()
