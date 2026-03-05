#!/usr/bin/env python3
"""
子模块级能量探测 – 推理入口脚本。

基于原始 generate_video.py 的 talking_avatar manifest 批量模式，
在推理过程中对 segment 0 的每一步去噪执行子模块级能量探测，
将 640 维（4 steps × 40 blocks × 4 sub-modules）能量图谱以 JSON 保存。

不修改原始仓库的任何代码，通过导入 SubmoduleProbeWrapper 在运行时
monkey-patch WanAttentionBlock.forward()。

用法：
    python generate_video_submodule_probe.py \
        --model_id /root/autodl-tmp/SkyReels-V3-A2V-19B \
        --calibration_manifest /root/autodl-fs/experiments/calibration_manifest.json \
        --probe_output_dir /root/autodl-fs/experiments/submodule_probe_raw \
        --seed 42

输出格式（每个样本一个 JSON 文件）：
    {
      "0_0_SA":  {"cos_sim": ..., "res_mag": ..., "energy": ..., ...},
      "0_0_CCA": {...},
      ...
      "3_39_FFN": {...}
    }
"""

import argparse
import json
import logging
import os
import sys
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - probe - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler()],
)

import imageio
import subprocess
import torch

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from skyreels_v3.configs import WAN_CONFIGS
from skyreels_v3.modules import download_model
from skyreels_v3.pipelines import TalkingAvatarPipeline
from skyreels_v3.utils.avatar_preprocess import preprocess_audio
from skyreels_v3.utils.profiler import profiler

# 导入子模块探测核心
from tools.submodule_probe import SubmoduleProbeWrapper


class StepTracker:
    """
    通过 model.forward() 调用计数来追踪当前去噪步索引。

    设计原理：
    - pipeline 的 denoise loop 每步调用 model 1 次（CFG=1.0 时）
    - segment 0 共 4 步 = 4 次 forward 调用
    - segment 1+ 的 forward 调用因 enabled=False 而不被追踪
    """

    def __init__(self):
        self.call_count = 0
        self.calls_per_step = 1   # CFG=1.0 → 每步 1 次调用
        self.enabled = False
        self.max_steps = 4        # 仅追踪 segment 0 的 4 步

    @property
    def current_step(self):
        return self.call_count // self.calls_per_step

    def reset(self):
        self.call_count = 0


def install_step_tracker(model, probe_wrapper, tracker):
    """
    Wrap model.forward() 使每次调用时自动设置 probe_wrapper.current_step。

    segment 0 的 4 步完成后自动禁用探测（tracker.enabled → False），
    后续 segment 的 forward 调用不受影响。
    """
    original_forward = model.forward

    def tracked_forward(*args, **kwargs):
        if tracker.enabled:
            step = tracker.current_step
            probe_wrapper.current_step = step
            result = original_forward(*args, **kwargs)
            tracker.call_count += 1
            # 超过 max_steps 后自动禁用，避免 segment 1+ 数据污染
            if tracker.current_step >= tracker.max_steps:
                tracker.enabled = False
                logging.info(
                    f"Step tracker auto-disabled after {tracker.call_count} calls "
                    f"({tracker.max_steps} steps)."
                )
            return result
        else:
            return original_forward(*args, **kwargs)

    model.forward = tracked_forward
    model._original_forward_for_probe = original_forward


def uninstall_step_tracker(model):
    """恢复 model.forward。"""
    if hasattr(model, "_original_forward_for_probe"):
        model.forward = model._original_forward_for_probe
        del model._original_forward_for_probe


def main():
    parser = argparse.ArgumentParser(
        description="SkyReels-V3 子模块级能量探测推理"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="模型路径（如 /root/autodl-tmp/SkyReels-V3-A2V-19B）")
    parser.add_argument("--calibration_manifest", type=str, required=True,
                        help="校准数据集 manifest JSON 路径")
    parser.add_argument("--probe_output_dir", type=str, required=True,
                        help="探测结果输出目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=str, default="720P",
                        choices=["480P", "540P", "720P"])
    parser.add_argument("--save_video", action="store_true",
                        help="同时保存生成的视频（用于质量对照）")
    parser.add_argument("--low_vram", action="store_true",
                        help="启用低显存模式（FP8 + block offload）")
    args = parser.parse_args()

    # ── 加载校准数据集声明 ──
    with open(args.calibration_manifest, "r") as f:
        manifest_items = json.load(f)
    logging.info(f"Loaded {len(manifest_items)} calibration samples from {args.calibration_manifest}")

    os.makedirs(args.probe_output_dir, exist_ok=True)

    # ── 初始化 pipeline ──
    args.model_id = download_model(args.model_id)
    config = WAN_CONFIGS["talking-avatar-19B"]
    profiler.start("Pipeline Init")
    pipe = TalkingAvatarPipeline(
        config=config,
        model_path=args.model_id,
        device_id=0,
        rank=0,
        use_usp=False,
        offload=False,
        low_vram=args.low_vram,
    )
    profiler.end("Pipeline Init")

    # ── 安装子模块探测 ──
    probe_wrapper = SubmoduleProbeWrapper()
    probe_wrapper.install(pipe.model)

    # 安装 step 追踪
    tracker = StepTracker()
    install_step_tracker(pipe.model, probe_wrapper, tracker)

    timing = {}

    # ── 遍历校准样本 ──
    for item_idx, item in enumerate(manifest_items):
        video_id = item.get("id", f"sample_{item_idx:04d}")
        logging.info(f"\n{'='*60}")
        logging.info(f"[{item_idx + 1}/{len(manifest_items)}] Processing {video_id}")
        logging.info(f"{'='*60}")

        input_image = item["ref_image_path"]
        input_audio = item["audio_path"]
        prompt = item.get("prompt", "A high-quality video of a person talking to the camera, natural lighting, realistic.")

        input_data = {
            "prompt": prompt,
            "cond_image": input_image,
            "cond_audio": {"person1": input_audio},
        }

        profiler.start(f"Audio Preprocess ({video_id})")
        input_data, _ = preprocess_audio(args.model_id, input_data, "processed_audio")
        profiler.end(f"Audio Preprocess ({video_id})")

        kwargs = {
            "input_data": input_data,
            "size_buckget": args.resolution,
            "motion_frame": 5,
            "frame_num": 81,
            "drop_frame": 12,
            "shift": 11,
            "text_guide_scale": 1.0,    # 不做 CFG，每步仅 1 次前向
            "audio_guide_scale": 1.0,
            "seed": args.seed,
            "sampling_steps": 4,
            "max_frames_num": 5000,
        }

        # 重置追踪器和探测数据
        probe_wrapper.clear()
        tracker.reset()
        tracker.calls_per_step = 1  # CFG=1.0 → 每步 1 次调用
        tracker.max_steps = kwargs["sampling_steps"]
        tracker.enabled = True

        t0 = time.time()
        profiler.start(f"Video Generation ({video_id})")
        video_out = pipe.generate(**kwargs)
        profiler.end(f"Video Generation ({video_id})")
        elapsed = time.time() - t0

        # tracker 在 segment 0 完成后已自动禁用
        timing[video_id] = elapsed

        # ── 保存探测数据 ──
        serialized = SubmoduleProbeWrapper.serialize(probe_wrapper.collect())
        probe_out_path = os.path.join(args.probe_output_dir, f"{video_id}_submodule_probe.json")
        with open(probe_out_path, "w") as f:
            json.dump(serialized, f, indent=2)
        logging.info(f"Saved {len(serialized)} submodule entries to {probe_out_path}")

        # ── 可选：保存视频 ──
        if args.save_video and video_out is not None:
            video_path = os.path.join(args.probe_output_dir, f"{video_id}_{args.seed}.mp4")
            imageio.mimwrite(video_path, video_out, fps=25, quality=8,
                             output_params=["-loglevel", "error"])
            if "video_audio" in input_data:
                video_with_audio_path = os.path.join(
                    args.probe_output_dir,
                    f"{video_id}_{args.seed}_with_audio.mp4"
                )
                cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", input_data["video_audio"],
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-shortest",
                    video_with_audio_path,
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    logging.info(f"Saved video with audio: {video_with_audio_path}")
                    os.remove(video_path)
                except subprocess.CalledProcessError as e:
                    logging.warning(f"ffmpeg failed for {video_id}: {e.stderr}")

        # 检验数据完整性
        num_steps = 4
        num_blocks = 40
        expected = num_steps * num_blocks * 4  # 4 sub-modules
        actual = len(serialized)
        if actual != expected:
            logging.warning(
                f"Data integrity check: expected {expected} entries, got {actual}. "
                f"This may happen because the pipeline runs multiple segments and "
                f"only segment 0 is tracked. Check the probe data carefully."
            )
        else:
            logging.info(f"Data integrity OK: {actual} entries (4 steps × 40 blocks × 4 sub-modules)")

    # ── 保存计时数据 ──
    timing_path = os.path.join(args.probe_output_dir, "batch_timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2)
    logging.info(f"Saved timing data to {timing_path}")

    # ── 清理 ──
    uninstall_step_tracker(pipe.model)
    probe_wrapper.remove(pipe.model)
    logging.info("All calibration samples processed. Probe complete.")


if __name__ == "__main__":
    main()
