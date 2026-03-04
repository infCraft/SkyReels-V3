#!/usr/bin/env python3
"""
Batch video quality evaluation script for SkyReels-V3 talking avatar experiments.

Computes FVD, LPIPS, PSNR, SSIM between GT videos and generated videos specified
via a manifest JSON file. Handles resolution mismatch by resizing GT to generated
video resolution, and frame count mismatch by truncating to min(T_gt, T_gen).

Usage example:
    python evaluate/batch_evaluate.py \
        --manifest /root/autodl-fs/experiments/evaluation_manifest.json \
        --gen_dir /root/autodl-fs/experiments/baseline_results \
        --seed 42 \
        --output_dir /root/autodl-fs/experiments/baseline_metrics_v2
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Add the common_metrics_on_video_quality directory to sys.path so that the
# existing calculate_* modules (and their internal relative imports like
# `from fvd.styleganv.fvd import ...`) work correctly.
# ---------------------------------------------------------------------------
_METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "common_metrics_on_video_quality")
if _METRICS_DIR not in sys.path:
    sys.path.insert(0, _METRICS_DIR)

warnings.filterwarnings("ignore")


# ========================  Video I/O Utilities  ============================

def load_video_frames(video_path: str, target_hw: tuple = None,
                      max_frames: int = None) -> torch.Tensor:
    """
    Read a video file and return a float32 tensor of shape [T, C, H, W] in [0, 1].

    Parameters
    ----------
    video_path : str
        Path to .mp4 video file.
    target_hw : tuple (H, W) or None
        If provided and the video resolution differs, resize every frame to
        this (H, W) using bilinear interpolation.
    max_frames : int or None
        If provided, truncate to at most this many frames.

    Returns
    -------
    torch.Tensor [T, C, H, W], float32, values in [0, 1]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize if needed
        if target_hw is not None:
            h_target, w_target = target_hw
            h_cur, w_cur = frame.shape[:2]
            if (h_cur, w_cur) != (h_target, w_target):
                frame = cv2.resize(frame, (w_target, h_target),
                                   interpolation=cv2.INTER_LINEAR)
        # HWC uint8 -> CHW float32 [0,1]
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        frames.append(frame_tensor)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    return torch.stack(frames)  # [T, C, H, W]


def get_video_resolution(video_path: str) -> tuple:
    """Return (H, W) of the first frame of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    h, w = frame.shape[:2]
    return (h, w)


# =====================  Video Pair Loading  ================================

def load_manifest(manifest_path: str) -> list:
    """Load manifest JSON (list of dicts or JSONL)."""
    with open(manifest_path, "r") as f:
        content = f.read().strip()

    # Try standard JSON list first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        else:
            return [data]
    except json.JSONDecodeError:
        pass

    # Fallback: JSONL (one JSON object per line)
    items = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def build_gen_video_path(gen_dir: str, gen_pattern: str,
                         item_id: str, seed: int) -> str:
    """Construct the generated video filename from pattern."""
    filename = gen_pattern.format(id=item_id, seed=seed)
    return os.path.join(gen_dir, filename)


# ======================  Per-Video Pixel Metrics  ==========================

def compute_per_video_pixel_metrics(
    gt_tensor: torch.Tensor,
    gen_tensor: torch.Tensor,
    device: torch.device,
    metrics: list,
) -> dict:
    """
    Compute per-frame pixel-level metrics for a single video pair.

    Parameters
    ----------
    gt_tensor : [T, C, H, W] float32 in [0, 1]
    gen_tensor : [T, C, H, W] float32 in [0, 1]
    device : torch device
    metrics : list of str, subset of ['lpips', 'psnr', 'ssim']

    Returns
    -------
    dict with keys like 'ssim', 'ssim_frame_std', 'psnr', 'psnr_frame_std',
    'lpips', 'lpips_frame_std'.
    """
    result = {}
    # Reshape to [1, T, C, H, W] for the calculate_* functions
    v_gt = gt_tensor.unsqueeze(0)   # [1, T, C, H, W]
    v_gen = gen_tensor.unsqueeze(0)  # [1, T, C, H, W]

    if "ssim" in metrics:
        from calculate_ssim import calculate_ssim
        # only_final=False returns per-timestamp values for a single video
        # With B=1, the per-timestamp list has T entries, each being the mean
        # across the batch (which is just 1 video).
        ssim_out = calculate_ssim(v_gt, v_gen, only_final=False)
        per_frame = np.array(ssim_out["value"])  # [T]
        result["ssim"] = float(np.mean(per_frame))
        result["ssim_frame_std"] = float(np.std(per_frame))

    if "psnr" in metrics:
        from calculate_psnr import calculate_psnr
        psnr_out = calculate_psnr(v_gt, v_gen, only_final=False)
        per_frame = np.array(psnr_out["value"])
        result["psnr"] = float(np.mean(per_frame))
        result["psnr_frame_std"] = float(np.std(per_frame))

    if "lpips" in metrics:
        from calculate_lpips import calculate_lpips
        lpips_out = calculate_lpips(v_gt, v_gen, device, only_final=False)
        per_frame = np.array(lpips_out["value"])
        result["lpips"] = float(np.mean(per_frame))
        result["lpips_frame_std"] = float(np.std(per_frame))

    return result


def downsample_video_for_fvd(video_tensor: torch.Tensor,
                             target_hw: tuple = (224, 224)) -> torch.Tensor:
    """Downsample [T, C, H, W] video to target size for lighter FVD memory usage."""
    if video_tensor.shape[-2:] == target_hw:
        return video_tensor
    return F.interpolate(
        video_tensor,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )


# ============================  FVD  ========================================

def compute_fvd(
    gt_videos: list,
    gen_videos: list,
    device: torch.device,
    fvd_method: str = "styleganv",
) -> float:
    """
    Compute FVD between two sets of videos.

    Because FVD is a distribution-level metric, we pad/truncate all videos to
    the same frame count and stack them into a single batch tensor.

    Parameters
    ----------
    gt_videos : list of [T_i, C, H, W] tensors
    gen_videos : list of [T_i, C, H, W] tensors (same length as gt_videos)
    device : torch device
    fvd_method : 'styleganv' or 'videogpt'

    Returns
    -------
    float : FVD score
    """
    from calculate_fvd import calculate_fvd

    assert len(gt_videos) == len(gen_videos), "Mismatch in number of videos"

    # Determine uniform frame count: use the minimum across all videos
    # (all pairs should already be truncated to min(T_gt, T_gen), but
    #  different pairs may have slightly different T)
    min_frames = min(
        min(v.shape[0] for v in gt_videos),
        min(v.shape[0] for v in gen_videos),
    )
    # FVD requires >= 10 frames
    assert min_frames >= 10, f"FVD requires >= 10 frames, got {min_frames}"

    # Stack into [B, T, C, H, W]
    gt_batch = torch.stack([v[:min_frames] for v in gt_videos])
    gen_batch = torch.stack([v[:min_frames] for v in gen_videos])

    result = calculate_fvd(gt_batch, gen_batch, device,
                           method=fvd_method, only_final=True)
    return result["value"][0]


# =========================  Summary Stats  =================================

def summarize_per_video(per_video: dict, metric_key: str):
    """Compute mean/std/min/max/count for a metric across all per-video results."""
    values = []
    for vid_id, vid_metrics in per_video.items():
        if metric_key in vid_metrics:
            values.append(vid_metrics[metric_key])
    if len(values) == 0:
        return {}
    arr = np.array(values)
    return {
        metric_key: float(np.mean(arr)),
        f"{metric_key}_std": float(np.std(arr)),
        f"{metric_key}_min": float(np.min(arr)),
        f"{metric_key}_max": float(np.max(arr)),
        f"{metric_key}_count": len(values),
    }


# ========================  Output Formatting  ==============================

def write_json_results(output_path: str, config: dict, summary: dict,
                       per_video: dict):
    """Write results in the same format as baseline_metrics.json."""
    result = {
        "config": config,
        "summary": summary,
        "per_video": per_video,
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] JSON results saved to: {output_path}")


def write_csv_results(output_path: str, per_video: dict):
    """Write per-video results as CSV, compatible with baseline_per_video.csv."""
    fieldnames = ["id", "lpips", "lpips_frame_std", "psnr", "psnr_frame_std",
                  "ssim", "ssim_frame_std"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for vid_id, vid_metrics in per_video.items():
            row = {"id": vid_id}
            for key in fieldnames[1:]:
                if key in vid_metrics:
                    row[key] = vid_metrics[key]
                else:
                    row[key] = ""
            writer.writerow(row)
    print(f"[INFO] CSV results saved to: {output_path}")


def print_summary_table(summary: dict, elapsed: float):
    """Print a nice summary table to the terminal."""
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    metric_keys = ["fvd", "lpips", "psnr", "ssim"]
    for mk in metric_keys:
        if mk in summary:
            count_key = f"{mk}_count"
            std_key = f"{mk}_std"
            count = summary.get(count_key, "N/A")
            std = summary.get(std_key, None)
            std_str = f" (std={std:.4f})" if std is not None else ""
            print(f"  {mk.upper():>6s} = {summary[mk]:.4f}{std_str}  [n={count}]")
    print(f"  {'TIME':>6s} = {elapsed:.1f}s")
    print("=" * 60 + "\n")


# ============================  Main  =======================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch evaluate video quality metrics (FVD/LPIPS/PSNR/SSIM)")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to evaluation manifest JSON file")
    parser.add_argument("--gen_dir", type=str, required=True,
                        help="Directory containing generated videos")
    parser.add_argument("--gen_pattern", type=str,
                        default="{id}_{seed}_with_audio.mp4",
                        help="Filename pattern for generated videos. "
                             "Supports {id} and {seed} placeholders. "
                             "Default: '{id}_{seed}_with_audio.mp4'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed value used in gen_pattern (default: 42)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results. "
                             "Default: {gen_dir}_metrics/")
    parser.add_argument("--metrics", type=str, default="fvd,lpips,psnr,ssim",
                        help="Comma-separated list of metrics to compute "
                             "(default: fvd,lpips,psnr,ssim)")
    parser.add_argument("--fvd_method", type=str, default="styleganv",
                        choices=["styleganv", "videogpt"],
                        help="FVD computation method (default: styleganv)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Truncate videos to at most this many frames "
                             "(default: None = use all frames)")
    parser.add_argument("--skip_ids", type=str, default=None,
                        help="Comma-separated list of IDs to skip")
    return parser.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    # ---- Parse arguments ----
    metrics = [m.strip().lower() for m in args.metrics.split(",")]
    skip_ids = set()
    if args.skip_ids:
        skip_ids = set(s.strip() for s in args.skip_ids.split(","))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.gen_dir.rstrip("/") + "_metrics"
    os.makedirs(output_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no GPU is available.")
    device = torch.device("cuda")

    # ---- Load manifest ----
    manifest_items = load_manifest(args.manifest)
    print(f"[INFO] Loaded manifest with {len(manifest_items)} items from: {args.manifest}")

    # ---- Phase 1: Stream load video pairs + per-video metrics ----
    # To avoid OOM, we compute pixel metrics per-video immediately and only keep
    # 224x224 copies for FVD.
    print("\n[Phase 1] Loading/preprocessing pairs + per-video metrics...")

    pixel_metrics = [m for m in metrics if m in ("lpips", "psnr", "ssim")]
    need_fvd = "fvd" in metrics

    per_video_results = OrderedDict()  # id -> metric dict
    gt_video_list = []    # list of downsampled [T, C, H, W] tensors (for FVD)
    gen_video_list = []   # list of downsampled [T, C, H, W] tensors (for FVD)
    valid_ids = []        # IDs of successfully loaded pairs
    num_skipped = 0
    resolution_info = {}  # id -> {gt_res, gen_res, aligned_res}

    for item in tqdm(manifest_items, desc="Loading videos"):
        vid_id = item["id"]
        per_video_results[vid_id] = {"id": vid_id}

        # Skip if requested
        if vid_id in skip_ids:
            print(f"  [SKIP] {vid_id} (in skip_ids)")
            num_skipped += 1
            continue

        gt_path = item["gt_video_path"]
        gen_path = build_gen_video_path(args.gen_dir, args.gen_pattern,
                                        vid_id, args.seed)

        # Check existence
        if not os.path.exists(gt_path):
            print(f"  [WARN] GT video missing for {vid_id}: {gt_path}")
            num_skipped += 1
            continue
        if not os.path.exists(gen_path):
            print(f"  [WARN] Generated video missing for {vid_id}: {gen_path}")
            num_skipped += 1
            continue

        try:
            # Determine generated video resolution (this is our target)
            gen_hw = get_video_resolution(gen_path)

            # Load generated video (no resize needed – it's the reference resolution)
            gen_frames = load_video_frames(gen_path, target_hw=None,
                                           max_frames=args.max_frames)

            # Load GT video and resize to generated video's resolution
            gt_frames = load_video_frames(gt_path, target_hw=gen_hw,
                                          max_frames=args.max_frames)

            # Frame count alignment: truncate to min
            T_gt, T_gen = gt_frames.shape[0], gen_frames.shape[0]
            T_min = min(T_gt, T_gen)
            gt_frames = gt_frames[:T_min]
            gen_frames = gen_frames[:T_min]

            gt_hw = get_video_resolution(gt_path)
            resolution_info[vid_id] = {
                "gt_original": f"{gt_hw[1]}x{gt_hw[0]}",
                "gen": f"{gen_hw[1]}x{gen_hw[0]}",
                "aligned_to": f"{gen_hw[1]}x{gen_hw[0]}",
                "gt_frames": T_gt,
                "gen_frames": T_gen,
                "used_frames": T_min,
            }

            if pixel_metrics:
                vid_result = compute_per_video_pixel_metrics(
                    gt_frames,
                    gen_frames,
                    device,
                    pixel_metrics,
                )
                per_video_results[vid_id].update(vid_result)

            if need_fvd:
                gt_video_list.append(downsample_video_for_fvd(gt_frames, (224, 224)).cpu())
                gen_video_list.append(downsample_video_for_fvd(gen_frames, (224, 224)).cpu())

            valid_ids.append(vid_id)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [ERROR] Failed to load {vid_id}: {e}")
            num_skipped += 1
            continue

    print(f"\n[INFO] Successfully loaded {len(valid_ids)} video pairs, "
          f"skipped {num_skipped}")

    if len(valid_ids) == 0:
        print("[ERROR] No valid video pairs found. Exiting.")
        sys.exit(1)

    # ---- Phase 2: Pixel-level metrics status ----
    if pixel_metrics:
        print(f"\n[Phase 2] Pixel-level metrics already computed during loading: {pixel_metrics}")
    else:
        print("\n[Phase 2] Skipped (no pixel-level metrics requested)")

    # ---- Phase 3: Compute FVD ----
    fvd_value = None
    if need_fvd and len(valid_ids) >= 2:
        print(f"\n[Phase 3] Computing FVD ({args.fvd_method})...")
        try:
            fvd_value = compute_fvd(
                gt_video_list, gen_video_list,
                device, fvd_method=args.fvd_method,
            )
            print(f"  FVD = {fvd_value:.4f}")
        except Exception as e:
            print(f"  [ERROR] FVD computation failed: {e}")
            import traceback
            traceback.print_exc()
    elif need_fvd:
        print("\n[Phase 3] Skipped FVD (need at least 2 valid video pairs)")

    # ---- Phase 4: Summarize and write results ----
    print("\n[Phase 4] Summarizing results...")
    elapsed = time.time() - t_start

    summary = {}
    if fvd_value is not None:
        summary["fvd"] = fvd_value
        summary["fvd_count"] = len(valid_ids)

    for mk in ["lpips", "psnr", "ssim"]:
        if mk in pixel_metrics:
            stats = summarize_per_video(per_video_results, mk)
            summary.update(stats)

    summary["elapsed_seconds"] = round(elapsed, 2)

    # Config record
    config = {
        "manifest": os.path.abspath(args.manifest),
        "gen_dir": os.path.abspath(args.gen_dir),
        "gen_pattern": args.gen_pattern,
        "seed": args.seed,
        "metrics": metrics,
        "fvd_method": args.fvd_method,
        "max_frames": args.max_frames,
        "device": str(device),
        "num_valid_pairs": len(valid_ids),
        "num_skipped": num_skipped,
        "resolution_alignment": "GT resized to generated video resolution",
        "frame_alignment": "truncated to min(T_gt, T_gen)",
    }

    # Write outputs
    json_path = os.path.join(output_dir, "metrics.json")
    csv_path = os.path.join(output_dir, "per_video.csv")

    write_json_results(json_path, config, summary, per_video_results)
    write_csv_results(csv_path, per_video_results)

    # Save resolution info for debugging
    res_info_path = os.path.join(output_dir, "resolution_info.json")
    with open(res_info_path, "w") as f:
        json.dump(resolution_info, f, indent=2)
    print(f"[INFO] Resolution info saved to: {res_info_path}")

    # Print summary
    print_summary_table(summary, elapsed)

    print("[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
