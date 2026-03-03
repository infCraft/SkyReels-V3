import argparse
import json
import logging
import os
import random
import time

# 配置日志格式和级别，实现实时终端打印
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - skyreels_v3 - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler()],  # 显式指定输出到终端
)
import subprocess

import imageio
import torch
# torch.cuda.set_per_process_memory_fraction(0.75)
import torch.distributed as dist
import wget
from diffusers.utils import load_image

from skyreels_v3.configs import WAN_CONFIGS
from skyreels_v3.modules import download_model
from skyreels_v3.pipelines import (
    ReferenceToVideoPipeline,
    ShotSwitchingExtensionPipeline,
    SingleShotExtensionPipeline,
    TalkingAvatarPipeline,
)
from skyreels_v3.utils.avatar_preprocess import preprocess_audio
from skyreels_v3.utils.profiler import profiler


def maybe_download(path_or_url: str, save_dir: str) -> str:
    """
    If `path_or_url` is already a local path, return it.
    Otherwise, download it into `save_dir` and return the downloaded local path.
    """
    if os.path.exists(path_or_url):
        return path_or_url

    url = path_or_url
    filename = url.split("/")[-1]
    local_path = os.path.join(save_dir, filename)
    logging.info(f"downloading input: {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        logging.info(f"input already exists: {local_path}")
        return local_path

    wget.download(url, local_path)
    assert os.path.exists(local_path), f"Failed to download input: {url}"
    logging.info(f"finished downloading input: {local_path}")
    return local_path


def prepare_and_broadcast_inputs(args, local_rank: int):
    """
    Prepare (download) inputs on rank0, and broadcast resolved local paths to all ranks.
    This keeps multi-process inference consistent (every process sees the same args.input_*).
    """
    is_dist = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    is_rank0 = (dist.get_rank() == 0) if is_dist else (local_rank == 0)

    obj_list = [None]
    if is_rank0:
        updates = {
            "input_video": args.input_video,
            "input_audio": args.input_audio,
            "input_image": args.input_image,
            "ref_imgs": args.ref_imgs,
        }

        if args.task_type in ["single_shot_extension", "shot_switching_extension"]:
            updates["input_video"] = maybe_download(args.input_video, "input_video")

        if args.task_type == "talking_avatar":
            updates["input_audio"] = maybe_download(args.input_audio, "input_audio")
            updates["input_image"] = maybe_download(args.input_image, "input_image")

        if args.task_type == "reference_to_video":
            # Normalize to list[str] and resolve URLs to local paths on rank0.
            ref_imgs = args.ref_imgs
            if isinstance(ref_imgs, str):
                ref_imgs = [p.strip() for p in ref_imgs.split(",") if p.strip()]
            assert isinstance(ref_imgs, list) and len(ref_imgs) > 0, "ref_imgs must be a list of images"
            updates["ref_imgs"] = [maybe_download(p, "ref_imgs") for p in ref_imgs]

        obj_list[0] = updates
        print("prepare input data done")

    if is_dist:
        dist.broadcast_object_list(obj_list, src=0)
        dist.barrier()

    updates = obj_list[0]
    if updates:
        args.input_video = updates.get("input_video", args.input_video)
        args.input_audio = updates.get("input_audio", args.input_audio)
        args.input_image = updates.get("input_image", args.input_image)
        args.ref_imgs = updates.get("ref_imgs", args.ref_imgs)

    # For reference_to_video, load images on every rank after we agree on local paths.
    if args.task_type == "reference_to_video":
        ref_imgs = args.ref_imgs
        if isinstance(ref_imgs, str):
            ref_imgs = [p.strip() for p in ref_imgs.split(",") if p.strip()]
        if isinstance(ref_imgs, list) and (len(ref_imgs) == 0 or isinstance(ref_imgs[0], str)):
            ref_imgs = [load_image(p) for p in ref_imgs]
        args.ref_imgs = ref_imgs
        assert isinstance(args.ref_imgs, list) and len(args.ref_imgs) > 0, "ref_imgs must be a list of images"

    return args


MODEL_ID_CONFIG = {
    "single_shot_extension": "Skywork/SkyReels-V3-V2V-14B",
    "shot_switching_extension": "Skywork/SkyReels-V3-V2V-14B",
    "reference_to_video": "Skywork/SkyReels-V3-R2V-14B",
    "talking_avatar": "Skywork/SkyReels-V3-A2V-19B",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SkyReels V3: Multimodal Video Generation Model")

    # ==================== Task Selection ====================
    parser.add_argument(
        "--task_type",
        type=str,
        choices=[
            "single_shot_extension",  # Single-shot video extension (5s to 30s)
            "shot_switching_extension",  # Shot switching extension with cinematic transitions (Cut-In, Cut-Out, etc.), limited to 5s
            "reference_to_video",  # Generate video from 1-4 reference images with text prompt
            "talking_avatar",  # Generate talking avatar from portrait image and audio (up to 200s)
        ],
        help="Type of video generation task to perform.",
    )

    # ==================== Model Configuration ====================
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model path or HuggingFace model ID. If not specified, will auto-select based on task_type. "
        "Supports: Skywork/SkyReels-V3-R2V-14B, Skywork/SkyReels-V3-V2V-14B, Skywork/SkyReels-V3-A2V-19B",
    )

    # ==================== Generation Parameters ====================
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Output video duration in seconds. "
        "For single_shot_extension: 5-30s; for shot_switching_extension: max 5s; for reference_to_video: recommended 5s.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A man is making his way forward slowly, leaning on a white cane to prop himself up.",
        help="Text prompt describing the desired video content. For shot_switching_extension, use prefixes like [ZOOM_IN_CUT], [ZOOM_OUT_CUT], etc.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720P",
        choices=["480P", "540P", "720P"],
        help="Output video resolution. Lower resolution (540P/480P) recommended for low VRAM GPUs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation. Required when using --use_usp mode.",
    )

    # ==================== Performance & Memory Options ====================
    parser.add_argument(
        "--use_usp",
        action="store_true",
        help="Enable multi-GPU parallel inference using xDiT USP (Unified Sequence Parallelism). "
        "Use with torchrun --nproc_per_node=N. Cannot be used with --low_vram.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable model offloading to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Enable low VRAM mode with FP8 weight-only quantization and block offload. "
        "Recommended for GPUs with <24GB VRAM. Cannot be used with --use_usp.",
    )

    # ==================== Video Extension Parameters ====================
    parser.add_argument(
        "--input_video",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4",
        help="[single_shot_extension/shot_switching_extension] Input video path or URL to extend.",
    )

    # ==================== Reference to Video Parameters ====================
    parser.add_argument(
        "--ref_imgs",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/subject_reference/0_0.png",
        help="[reference_to_video] Reference images (1-4) for video generation. "
        "Supports character portraits, objects, and backgrounds. "
        "Multiple images should be comma-separated (e.g., 'img1.png,img2.png').",
    )

    # ==================== Talking Avatar Parameters ====================
    parser.add_argument(
        "--input_image",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single1.png",
        help="[talking_avatar] Portrait image path or URL for avatar generation. "
        "Supports jpg/jpeg, png, gif, bmp formats. Works with real people, anime, animals, and stylized characters.",
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single_actor/huahai_5s.mp3",
        help="[talking_avatar] Driving audio path or URL. Supports mp3, wav formats. "
        "Audio duration must be <= 200 seconds. Supports multiple languages.",
    )

    # ==================== Batch & Probe Parameters ====================
    parser.add_argument(
        "--manifest_json",
        type=str,
        default=None,
        help="Path to a manifest JSON file for batch inference. Each entry should contain "
        "{id, ref_image_path, audio_path, gt_video_path, prompt}.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for batch inference results.",
    )
    parser.add_argument(
        "--probe_mode",
        action="store_true",
        help="Enable sensitivity probe mode (Phase 2). Records per-block redundancy metrics (cos_sim, res_mag).",
    )
    parser.add_argument(
        "--skip_list_json",
        type=str,
        default=None,
        help="Path to a skip-list JSON file for block pruning (Phase 3).",
    )

    args = parser.parse_args()

    # 自动补全model_id参数，如果用户没有指定，则根据task_type选择默认模型。
    if args.model_id is None:
        args.model_id = MODEL_ID_CONFIG[args.task_type]
    # init multi gpu environment
    local_rank = 0
    if args.use_usp:
        # torchrun 多卡环境初始化
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        dist.init_process_group("nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(dist.get_rank())

        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
    device = f"cuda:{local_rank}"
    assert not(args.use_usp and args.low_vram), "usp mode and low_vram mode cannot be used together"

    # 多卡环境下，仅rank0下载模型文件，然后广播路径到其他rank，避免重复下载浪费时间和带宽。
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        obj_list = [None]
        if dist.get_rank() == 0:
            obj_list[0] = download_model(args.model_id)
        dist.broadcast_object_list(obj_list, src=0)
        args.model_id = obj_list[0]
        dist.barrier()
    else:
        args.model_id = download_model(args.model_id)

    print(f"args.model_id: {args.model_id}")

    assert (args.use_usp and args.seed is not None) or (not args.use_usp), "usp mode need seed"
    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    logging.info(f"input params: {args}")

    # 把输入准备好（下载视频/图片/音频等），并在多卡环境下广播给所有rank，确保每个rank都使用相同的输入路径和数据。对于reference_to_video任务，还会在每个rank上加载参考图片。
    args = prepare_and_broadcast_inputs(args, local_rank)

    video_out = None

    # 根据不同的任务类型，初始化对应的Pipeline，并调用生成函数。每个Pipeline内部会根据传入的参数进行视频生成。
    if args.task_type == "single_shot_extension":
        pipe = SingleShotExtensionPipeline(model_path=args.model_id, use_usp=args.use_usp, offload=args.offload, low_vram=args.low_vram)
        video_out = pipe.extend_video(args.input_video, args.prompt, args.duration, args.seed, resolution=args.resolution)
    elif args.task_type == "shot_switching_extension":
        pipe = ShotSwitchingExtensionPipeline(model_path=args.model_id, use_usp=args.use_usp, offload=args.offload, low_vram=args.low_vram)
        video_out = pipe.extend_video(args.input_video, args.prompt, args.duration, args.seed, resolution=args.resolution)
    elif args.task_type == "reference_to_video":
        pipe = ReferenceToVideoPipeline(model_path=args.model_id, use_usp=args.use_usp, offload=args.offload, low_vram=args.low_vram)
        video_out = pipe.generate_video(args.ref_imgs, args.prompt, args.duration, args.seed, resolution=args.resolution)
    elif args.task_type == "talking_avatar":
        config = WAN_CONFIGS["talking-avatar-19B"]
        profiler.start("Pipeline Init (total)")
        pipe = TalkingAvatarPipeline(
            config=config,
            model_path=args.model_id,
            device_id=local_rank,
            rank=local_rank,
            use_usp=args.use_usp,
            offload=args.offload,
            low_vram=args.low_vram,
        )
        profiler.end("Pipeline Init (total)")

        # ---- Configure probe / skip attributes on the DiT model ----
        if args.probe_mode:
            pipe.model.probe_mode = True
            logging.info("Probe mode enabled. Raw metrics (cos_sim, res_mag) will be recorded.")

        if args.skip_list_json:
            with open(args.skip_list_json, "r") as f:
                skip_list = json.load(f)
            pipe.model.skip_set = set(tuple(pair) for pair in skip_list)
            logging.info(f"Loaded skip_set with {len(pipe.model.skip_set)} (step, block) pairs")

        # ---- Build list of input_data items (single or batch from manifest) ----
        manifest_items = []
        if args.manifest_json:
            with open(args.manifest_json, "r") as f:
                manifest_items = json.load(f)
            logging.info(f"Loaded manifest with {len(manifest_items)} items from {args.manifest_json}")
        else:
            # Single-item mode: wrap the CLI args into a pseudo-manifest entry
            manifest_items = [{
                "id": "single",
                "ref_image_path": args.input_image,
                "audio_path": args.input_audio,
                "gt_video_path": None,
                "prompt": args.prompt,
            }]

        batch_output_dir = args.output_dir or os.path.join("result", args.task_type)
        os.makedirs(batch_output_dir, exist_ok=True)
        batch_timing = {}  # video_id -> wall_clock_seconds

        for item_idx, item in enumerate(manifest_items):
            video_id = item["id"]
            logging.info(f"[{item_idx+1}/{len(manifest_items)}] Processing video_id={video_id}")

            # 构造输入数据字典
            input_data = {
                "prompt": item.get("prompt", args.prompt),
                "cond_image": item["ref_image_path"],
                "cond_audio": {"person1": item["audio_path"]},
            }
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                obj_list = [None]
                if dist.get_rank() == 0:
                    input_data, _ = preprocess_audio(args.model_id, input_data, "processed_audio")
                    obj_list[0] = input_data
                dist.broadcast_object_list(obj_list, src=0)
                input_data = obj_list[0]
                dist.barrier()
            else:
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
                "text_guide_scale": 1.0,
                "audio_guide_scale": 1.0,
                "seed": args.seed,
                "sampling_steps": 4,
                "max_frames_num": 5000,
                "probe_mode": args.probe_mode,
            }
            logging.info(f"generate video kwargs: {kwargs}")

            t_start = time.time()
            profiler.start(f"Video Generation ({video_id})")
            video_out = pipe.generate(**kwargs)
            profiler.end(f"Video Generation ({video_id})")
            wall_time = time.time() - t_start
            batch_timing[video_id] = wall_time
            logging.info(f"Video {video_id} generated in {wall_time:.2f}s")

            # ---- Save probe results if in probe mode ----
            if args.probe_mode and hasattr(pipe.model, "probe_results") and pipe.model.probe_results:
                probe_out_path = os.path.join(batch_output_dir, f"{video_id}_probe.json")
                # Convert tuple keys to string keys for JSON serialization
                probe_serializable = {}
                for k, v in pipe.model.probe_results.items():
                    key_str = f"{k[0]}_{k[1]}"  # "step_blockidx"
                    probe_serializable[key_str] = v
                with open(probe_out_path, "w") as f:
                    json.dump(probe_serializable, f, indent=2)
                logging.info(f"Saved probe results to {probe_out_path} ({len(probe_serializable)} entries)")
                pipe.model.probe_results = {}  # Clear for next video

            # ---- Save generated video ----
            if local_rank == 0 and video_out is not None:
                video_out_file = f"{video_id}_{args.seed}.mp4"
                output_path = os.path.join(batch_output_dir, video_out_file)
                imageio.mimwrite(
                    output_path,
                    video_out,
                    fps=25,
                    quality=8,
                    output_params=["-loglevel", "error"],
                )
                # Merge audio
                if "video_audio" in input_data:
                    video_with_audio_path = os.path.join(batch_output_dir, f"{video_id}_{args.seed}_with_audio.mp4")
                    audio_path = input_data["video_audio"]
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', f'"{os.path.abspath(output_path)}"',
                        '-i', f'"{os.path.abspath(audio_path)}"',
                        '-map', '0:v', '-map', '1:a',
                        '-c:v', 'copy', '-shortest',
                        f'"{os.path.abspath(video_with_audio_path)}"'
                    ]
                    try:
                        subprocess.run(" ".join(cmd), shell=True, check=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                        logging.info(f"Saved video with audio: {video_with_audio_path}")
                        os.remove(output_path)
                    except subprocess.CalledProcessError as e:
                        logging.warning(f"ffmpeg failed for {video_id}: {e.stdout}")

        # ---- Save batch timing summary ----
        if len(manifest_items) > 1:
            timing_path = os.path.join(batch_output_dir, "batch_timing.json")
            with open(timing_path, "w") as f:
                json.dump(batch_timing, f, indent=2)
            logging.info(f"Saved batch timing to {timing_path}")

        # Skip the default save logic below for talking_avatar (already handled)
        video_out = None
    else:
        raise ValueError(f"Invalid task type: {args.task_type}")

    save_dir = os.path.join("result", args.task_type)
    os.makedirs(save_dir, exist_ok=True)

    if local_rank == 0 and video_out is not None:
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        video_out_file = f"{args.seed}_{current_time}.mp4"
        output_path = os.path.join(save_dir, video_out_file)
        fps = 25 if args.task_type == "talking_avatar" else 24
        profiler.start("Video Save (imageio)")
        imageio.mimwrite(
            output_path,
            video_out,
            fps=fps,
            quality=8,
            output_params=["-loglevel", "error"],
        )
        profiler.end("Video Save (imageio)")
        if args.task_type == "talking_avatar":
            video_with_audio_path = os.path.join(save_dir, video_out_file.replace(".mp4", "_with_audio.mp4"))
            audio_path = kwargs["input_data"]["video_audio"]
            video_in = os.path.abspath(output_path)
            audio_in = os.path.abspath(audio_path)
            video_out_with_audio = os.path.abspath(video_with_audio_path)
            print(f"video_in: {video_in}, audio_in: {audio_in}, video_out_with_audio: {video_out_with_audio}")
            # fmt: off
            cmd = [
                'ffmpeg',
                '-y',
                '-i', f'"{video_in}"',
                '-i', f'"{audio_in}"',
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'copy',
                '-shortest',
                f'"{video_out_with_audio}"'
            ]
            # fmt: on

            try:
                profiler.start("FFmpeg Audio Merge")
                subprocess.run(
                    " ".join(cmd),
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                profiler.end("FFmpeg Audio Merge")
                print(f"Video with audio generated successfully: {video_with_audio_path}")
                os.remove(video_in) # remove the original video
            except subprocess.CalledProcessError as e:
                profiler.end("FFmpeg Audio Merge")
                print(f"ffmpeg failed (exit={e.returncode}). Output:\n{e.stdout}")

    profiler.summary()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
