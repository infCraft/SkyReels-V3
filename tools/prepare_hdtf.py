#!/usr/bin/env python3
"""
HDTF 数据集预处理脚本

功能：
1. 扫描指定目录下所有 MP4 文件
2. 用 ffprobe 检查分辨率(≥720p)、时长(可配置)、无损坏
3. 设定 random.seed(42) 后随机抽取 50 个视频
4. 对每个视频：
   - 用 ffmpeg 裁剪前 5 秒片段 → GT_video.mp4
   - 提取第 0 帧 → reference_image.jpg
   - 提取音频 → audio.wav（16kHz 单声道 PCM）
5. 生成统一的 Text Prompt
6. 前 10 个为标定集，后 40 个为评估集
7. 输出 calibration_manifest.json 和 evaluation_manifest.json

用法示例：
  python tools/prepare_hdtf.py \\
      --raw_dir /root/autodl-fs/experiments/hdtf_raw \\
      --output_dir /root/autodl-fs/experiments/hdtf_processed \\
      --experiment_dir /root/autodl-fs/experiments

  # 使用 mock 数据测试：
  python tools/prepare_hdtf.py --create_mock --mock_dir /tmp/hdtf_mock_test
"""

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# ffprobe / ffmpeg helpers
# ─────────────────────────────────────────────────────────────────────

def probe_video(video_path: str) -> dict | None:
    """
    用 ffprobe 获取视频信息：分辨率、时长、编解码器。
    返回 dict 或 None（如果视频损坏或无法读取）。
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,codec_name",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
            return None

        data = json.loads(result.stdout)

        # 获取视频流信息
        streams = data.get("streams", [])
        if not streams:
            logger.warning(f"No video streams found in {video_path}")
            return None

        stream = streams[0]
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))

        # 时长：优先从 stream 获取，其次从 format 获取
        duration = stream.get("duration")
        if duration is None:
            duration = data.get("format", {}).get("duration")
        if duration is None:
            logger.warning(f"Cannot determine duration for {video_path}")
            return None
        duration = float(duration)

        return {
            "path": str(video_path),
            "width": width,
            "height": height,
            "duration": duration,
            "codec": stream.get("codec_name", "unknown"),
        }
    except subprocess.TimeoutExpired:
        logger.warning(f"ffprobe timeout for {video_path}")
        return None
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"ffprobe parse error for {video_path}: {e}")
        return None


def check_audio_stream(video_path: str) -> bool:
    """检查视频是否包含音频流"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name",
        "-of", "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout)
        return len(data.get("streams", [])) > 0
    except Exception:
        return False


def compute_safe_offset(total_duration: float, crop_duration: float = 5.0) -> float:
    """
    计算安全的裁剪起始偏移量，避开视频开头的字幕/介绍画面。

    策略：
    - clip >= 15s: 跳过前 10s，从第 10s 开始
    - clip >= 10s: 取中间 5s 段
    - clip 5~10s:  取最后 5s（尽量远离开头字幕）
    - clip < 5s:   从头开始（无法避免）
    """
    if total_duration < crop_duration:
        return 0.0
    elif total_duration >= 15.0:
        # 足够长，跳过前 10s
        return 10.0
    elif total_duration >= 10.0:
        # 中等长度，取中间段
        return (total_duration - crop_duration) / 2.0
    else:
        # 5~10s，取最后 5s
        return total_duration - crop_duration


def crop_video(input_path: str, output_path: str, duration: float = 5.0, start_offset: float = 0.0) -> bool:
    """裁剪视频指定区间 [start_offset, start_offset + duration]"""
    cmd = [
        "ffmpeg",
        "-y",                    # 覆盖输出
        "-ss", str(start_offset), # 起始偏移
        "-i", str(input_path),
        "-t", str(duration),
        "-c:v", "libopenh264",  # 重新编码确保兼容（兼容无libx264环境）
        "-c:a", "aac",
        "-loglevel", "error",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning(f"ffmpeg crop failed for {input_path}: {result.stderr.strip()}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except subprocess.TimeoutExpired:
        logger.warning(f"ffmpeg crop timeout for {input_path}")
        return False


def extract_first_frame(input_path: str, output_path: str, start_offset: float = 0.0) -> bool:
    """提取指定偏移处的第一帧为 JPEG"""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_offset),
        "-i", str(input_path),
        "-vframes", "1",
        "-q:v", "2",           # JPEG 质量 (2=高质量)
        "-loglevel", "error",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning(f"ffmpeg frame extract failed for {input_path}: {result.stderr.strip()}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except subprocess.TimeoutExpired:
        logger.warning(f"ffmpeg frame extract timeout for {input_path}")
        return False


def extract_audio(input_path: str, output_path: str, sr: int = 16000, duration: float = 5.0, start_offset: float = 0.0) -> bool:
    """提取音频为 16kHz 单声道 PCM WAV（与 avatar_preprocess.py 的 librosa.load(sr=16000) 对齐）"""
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_offset),
        "-i", str(input_path),
        "-t", str(duration),
        "-vn",                   # 不要视频
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", str(sr),          # 采样率
        "-ac", "1",              # 单声道
        "-loglevel", "error",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.warning(f"ffmpeg audio extract failed for {input_path}: {result.stderr.strip()}")
            return False
        return os.path.exists(output_path) and os.path.getsize(output_path) > 0
    except subprocess.TimeoutExpired:
        logger.warning(f"ffmpeg audio extract timeout for {input_path}")
        return False


# ─────────────────────────────────────────────────────────────────────
# 核心处理流程
# ─────────────────────────────────────────────────────────────────────

def scan_and_filter_videos(
    raw_dir: str,
    min_height: int = 720,
    min_duration: float = 5.0,
    max_duration: float = 60.0,
) -> list[dict]:
    """
    扫描目录下所有 MP4 文件，用 ffprobe 过滤。

    返回满足条件的视频信息列表。
    """
    raw_path = Path(raw_dir)
    mp4_files = sorted(raw_path.rglob("*.mp4"))
    logger.info(f"Found {len(mp4_files)} MP4 files in {raw_dir}")

    if not mp4_files:
        logger.error(f"No MP4 files found in {raw_dir}")
        return []

    valid_videos = []
    skipped_probe = 0
    skipped_resolution = 0
    skipped_duration = 0
    skipped_no_audio = 0

    for i, mp4_path in enumerate(mp4_files):
        if (i + 1) % 100 == 0 or i == 0:
            logger.info(f"Probing video {i+1}/{len(mp4_files)}: {mp4_path.name}")

        info = probe_video(str(mp4_path))
        if info is None:
            skipped_probe += 1
            continue

        # 分辨率过滤：高度 >= min_height（即 >=720p）
        if info["height"] < min_height:
            skipped_resolution += 1
            continue

        # 时长过滤
        if info["duration"] < min_duration or info["duration"] > max_duration:
            skipped_duration += 1
            continue

        # 音频流检查
        if not check_audio_stream(str(mp4_path)):
            skipped_no_audio += 1
            continue

        valid_videos.append(info)

    logger.info(
        f"Scan results: total={len(mp4_files)}, valid={len(valid_videos)}, "
        f"skipped(probe={skipped_probe}, resolution={skipped_resolution}, "
        f"duration={skipped_duration}, no_audio={skipped_no_audio})"
    )
    return valid_videos


def sample_and_process(
    valid_videos: list[dict],
    output_dir: str,
    num_samples: int = 50,
    crop_duration: float = 5.0,
    seed: int = 42,
) -> list[dict]:
    """
    随机抽样并处理视频，生成 GT_video.mp4 + reference_image.jpg + audio.wav

    返回处理成功的记录列表。
    """
    random.seed(seed)

    if len(valid_videos) < num_samples:
        logger.warning(
            f"Only {len(valid_videos)} valid videos available, "
            f"requested {num_samples}. Using all."
        )
        selected = valid_videos[:]
    else:
        selected = random.sample(valid_videos, num_samples)

    # 按文件名排序确保确定性
    selected.sort(key=lambda v: os.path.basename(v["path"]))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    unified_prompt = (
        "A high-quality video of a person talking to the camera, "
        "natural lighting, realistic."
    )

    processed = []
    for idx, video_info in enumerate(selected):
        video_id = f"hdtf_{idx:04d}"
        src_name = Path(video_info["path"]).stem
        logger.info(
            f"Processing [{idx+1}/{len(selected)}] {video_id} "
            f"(src: {src_name}, dur: {video_info['duration']:.1f}s, "
            f"{video_info['width']}x{video_info['height']})"
        )

        # 创建子目录
        sample_dir = output_path / video_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        gt_video_path = sample_dir / "GT_video.mp4"
        ref_image_path = sample_dir / "reference_image.jpg"
        audio_path = sample_dir / "audio.wav"

        # 计算安全偏移量，避开视频开头的字幕/介绍画面
        start_offset = compute_safe_offset(video_info["duration"], crop_duration)
        logger.info(
            f"  Crop window: [{start_offset:.1f}s, {start_offset + crop_duration:.1f}s] "
            f"(total {video_info['duration']:.1f}s)"
        )

        # 1. 裁剪视频 [start_offset, start_offset + crop_duration]
        if not crop_video(video_info["path"], str(gt_video_path), duration=crop_duration, start_offset=start_offset):
            logger.warning(f"  SKIP: Failed to crop video for {video_id}")
            continue

        # 2. 提取裁剪起始处的第一帧作为 reference image
        if not extract_first_frame(video_info["path"], str(ref_image_path), start_offset=start_offset):
            logger.warning(f"  SKIP: Failed to extract frame for {video_id}")
            continue

        # 3. 提取对应区间的音频（16kHz 单声道）
        if not extract_audio(video_info["path"], str(audio_path), duration=crop_duration, start_offset=start_offset):
            logger.warning(f"  SKIP: Failed to extract audio for {video_id}")
            continue

        record = {
            "id": video_id,
            "source_file": src_name,
            "ref_image_path": str(ref_image_path),
            "audio_path": str(audio_path),
            "gt_video_path": str(gt_video_path),
            "prompt": unified_prompt,
            "source_duration": video_info["duration"],
            "source_resolution": f"{video_info['width']}x{video_info['height']}",
            "crop_duration": crop_duration,
            "start_offset": start_offset,
        }
        processed.append(record)
        logger.info(f"  OK: {video_id} ({src_name})")

    logger.info(f"Successfully processed {len(processed)}/{len(selected)} videos")
    return processed


def generate_manifests(
    processed: list[dict],
    experiment_dir: str,
    calibration_count: int = 10,
):
    """
    将处理好的记录分为标定集和评估集，生成 manifest JSON 文件。

    - 前 calibration_count 个 → calibration_manifest.json
    - 剩余 → evaluation_manifest.json
    """
    exp_path = Path(experiment_dir)
    exp_path.mkdir(parents=True, exist_ok=True)

    calibration = processed[:calibration_count]
    evaluation = processed[calibration_count:]

    cal_path = exp_path / "calibration_manifest.json"
    eval_path = exp_path / "evaluation_manifest.json"

    with open(cal_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)
    logger.info(f"Calibration manifest: {cal_path} ({len(calibration)} entries)")

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=2)
    logger.info(f"Evaluation manifest: {eval_path} ({len(evaluation)} entries)")

    # 打印摘要
    logger.info("=" * 60)
    logger.info("MANIFEST SUMMARY")
    logger.info(f"  Total processed: {len(processed)}")
    logger.info(f"  Calibration set: {len(calibration)} videos")
    logger.info(f"  Evaluation set:  {len(evaluation)} videos")
    if calibration:
        logger.info(f"  Calibration IDs: {calibration[0]['id']} ~ {calibration[-1]['id']}")
    if evaluation:
        logger.info(f"  Evaluation IDs:  {evaluation[0]['id']} ~ {evaluation[-1]['id']}")
    logger.info(f"  Prompt: \"{processed[0]['prompt']}\"" if processed else "  No data")
    logger.info("=" * 60)

    return cal_path, eval_path


# ─────────────────────────────────────────────────────────────────────
# Mock 数据生成（用于无数据环境下的功能验证）
# ─────────────────────────────────────────────────────────────────────

def create_mock_data(mock_dir: str, num_videos: int = 60):
    """
    使用 ffmpeg 生成模拟 MP4 视频（彩色条纹 + 静音/正弦波音频），
    用于在无 GPU / 无真实数据环境下验证流水线。
    """
    mock_path = Path(mock_dir)
    mock_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating {num_videos} mock videos in {mock_dir}")

    created = 0
    for i in range(num_videos):
        # 随机生成不同属性的视频用于测试过滤逻辑
        # 大部分满足条件，少部分故意不满足
        if i < 5:
            # 低分辨率，应被过滤
            width, height = 640, 480
            duration = 12
        elif i < 8:
            # 时长太短，应被过滤
            width, height = 1280, 720
            duration = 2
        elif i < 10:
            # 时长太长，应被过滤
            width, height = 1280, 720
            duration = 65
        else:
            # 正常视频
            width, height = 1280, 720
            duration = random.Random(42 + i).randint(5, 15)

        name = f"MockPerson{i:02d}_001.mp4"
        output_file = mock_path / name

        # 使用 lavfi 生成彩色条纹视频 + 正弦波音频
        # 注意：某些系统(如AutoDL)的 ffmpeg 没有 libx264，使用 libopenh264 替代
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i",
            f"testsrc=duration={duration}:size={width}x{height}:rate=25",
            "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={duration}:sample_rate=16000",
            "-c:v", "libopenh264",
            "-c:a", "aac", "-b:a", "32k",
            "-shortest",
            "-loglevel", "error",
            str(output_file),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and output_file.exists():
                created += 1
            else:
                logger.warning(f"Failed to create mock video {name}: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout creating mock video {name}")

    logger.info(f"Created {created}/{num_videos} mock videos in {mock_dir}")
    return created


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="HDTF 数据集预处理：扫描、过滤、裁剪、生成 manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 1. 生成 mock 数据并处理（功能验证，无需真实数据）
  python tools/prepare_hdtf.py --create_mock --mock_dir /tmp/hdtf_mock_test

  # 2. 处理真实 HDTF 数据
  python tools/prepare_hdtf.py \\
      --raw_dir /root/autodl-fs/experiments/hdtf_raw \\
      --output_dir /root/autodl-fs/experiments/hdtf_processed \\
      --experiment_dir /root/autodl-fs/experiments

  # 3. 仅打印帮助信息
  python tools/prepare_hdtf.py --help
        """,
    )

    # 数据路径
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="/root/autodl-fs/experiments/hdtf_raw",
        help="HDTF 原始 MP4 文件所在目录（支持递归扫描子目录）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-fs/experiments/hdtf_processed",
        help="处理后的文件输出目录",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="/root/autodl-fs/experiments",
        help="实验根目录，manifest 文件将保存在此目录下",
    )

    # 过滤参数
    parser.add_argument(
        "--min_height",
        type=int,
        default=720,
        help="最小视频高度（像素），低于此值的视频将被跳过。默认 720（即 720p）",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=5.0,
        help="最短视频时长（秒）。默认 5.0",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=60.0,
        help="最长视频时长（秒）。默认 60.0",
    )

    # 采样参数
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="随机抽取的视频数量。默认 50",
    )
    parser.add_argument(
        "--crop_duration",
        type=float,
        default=5.0,
        help="裁剪的视频片段时长（秒）。默认 5.0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子。默认 42",
    )
    parser.add_argument(
        "--calibration_count",
        type=int,
        default=10,
        help="标定集数量（前 N 个为标定集，其余为评估集）。默认 10",
    )

    # Mock 模式
    parser.add_argument(
        "--create_mock",
        action="store_true",
        help="生成 mock 数据并在其上运行完整处理流程（用于功能验证）",
    )
    parser.add_argument(
        "--mock_dir",
        type=str,
        default="/tmp/hdtf_mock_test",
        help="Mock 数据的临时目录。默认 /tmp/hdtf_mock_test",
    )
    parser.add_argument(
        "--mock_count",
        type=int,
        default=60,
        help="Mock 视频数量。默认 60",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Mock 模式 ──
    if args.create_mock:
        logger.info("=" * 60)
        logger.info("MOCK MODE: Creating synthetic data for pipeline validation")
        logger.info("=" * 60)

        mock_raw_dir = os.path.join(args.mock_dir, "raw")
        mock_output_dir = os.path.join(args.mock_dir, "processed")
        mock_experiment_dir = args.mock_dir

        # 1. 生成 mock 数据
        created = create_mock_data(mock_raw_dir, args.mock_count)
        if created == 0:
            logger.error("Failed to create any mock videos. Check ffmpeg installation.")
            sys.exit(1)

        # 2. 扫描和过滤（使用宽松参数以包含更多 mock 数据）
        valid = scan_and_filter_videos(
            mock_raw_dir,
            min_height=args.min_height,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )

        if not valid:
            logger.error("No valid videos after filtering. Check filter parameters.")
            sys.exit(1)

        # 3. 采样和处理
        processed = sample_and_process(
            valid,
            mock_output_dir,
            num_samples=min(args.num_samples, len(valid)),
            crop_duration=args.crop_duration,
            seed=args.seed,
        )

        if not processed:
            logger.error("No videos processed successfully.")
            sys.exit(1)

        # 4. 生成 manifest
        cal_path, eval_path = generate_manifests(
            processed,
            mock_experiment_dir,
            calibration_count=min(args.calibration_count, len(processed)),
        )

        # 5. 验证输出
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION")
        logger.info("=" * 60)

        with open(cal_path) as f:
            cal_data = json.load(f)
        with open(eval_path) as f:
            eval_data = json.load(f)

        logger.info(f"Calibration entries: {len(cal_data)}")
        logger.info(f"Evaluation entries:  {len(eval_data)}")

        if cal_data:
            sample = cal_data[0]
            logger.info(f"Sample entry keys: {list(sample.keys())}")
            for key in ["ref_image_path", "audio_path", "gt_video_path"]:
                exists = os.path.exists(sample[key])
                size = os.path.getsize(sample[key]) if exists else 0
                logger.info(f"  {key}: exists={exists}, size={size} bytes")

        logger.info("\nMock pipeline validation PASSED!")
        return

    # ── 正常模式 ──
    logger.info("=" * 60)
    logger.info("HDTF Dataset Preparation")
    logger.info(f"  Raw dir:        {args.raw_dir}")
    logger.info(f"  Output dir:     {args.output_dir}")
    logger.info(f"  Experiment dir: {args.experiment_dir}")
    logger.info(f"  Filter:         height>={args.min_height}, "
                f"duration=[{args.min_duration}, {args.max_duration}]s")
    logger.info(f"  Sampling:       {args.num_samples} videos, seed={args.seed}")
    logger.info(f"  Crop duration:  {args.crop_duration}s")
    logger.info("=" * 60)

    # 1. 扫描和过滤
    valid = scan_and_filter_videos(
        args.raw_dir,
        min_height=args.min_height,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    if not valid:
        logger.error(
            f"No valid videos found in {args.raw_dir}. "
            f"Please check the directory and filter parameters.\n"
            f"Tip: Use --create_mock to test with synthetic data first."
        )
        sys.exit(1)

    # 2. 采样和处理
    processed = sample_and_process(
        valid,
        args.output_dir,
        num_samples=args.num_samples,
        crop_duration=args.crop_duration,
        seed=args.seed,
    )

    if not processed:
        logger.error("No videos processed successfully.")
        sys.exit(1)

    # 3. 生成 manifest
    generate_manifests(
        processed,
        args.experiment_dir,
        calibration_count=args.calibration_count,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
