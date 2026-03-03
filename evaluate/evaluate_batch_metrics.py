#!/usr/bin/env python3
"""
Batch evaluation script for SkyReels-V3 talking avatar experiments.

Computes FVD, SSIM, PSNR, LPIPS between generated videos and Ground Truth (GT)
videos using the common_metrics_on_video_quality library.

Features:
  - Reads manifest JSON files to locate GT and generated video pairs
  - Memory-efficient: processes SSIM/PSNR/LPIPS one video pair at a time
  - FVD computed on all videos at configurable resolution (default 224x224)
  - Per-video metrics + overall summary
  - Supports selecting a subset of metrics
  - Graceful handling of missing videos, resolution mismatches, frame count diffs

Outputs:
  1. {prefix}_metrics.json  – full results (config, summary, per-video detail)
  2. {prefix}_per_video.csv – per-video metrics table (if pandas available)

Usage:
  python evaluate/evaluate_batch_metrics.py \
      --manifest /root/autodl-fs/experiments/evaluation_manifest.json \
      --gen_dir  /root/autodl-fs/experiments/baseline_results \
      --gen_pattern "{id}_42_with_audio.mp4" \
      --output_dir /root/autodl-fs/experiments \
      --output_prefix baseline \
      --metrics fvd ssim psnr lpips

Notes:
  - Pixel values are normalised to [0, 1] as required by the library.
  - For FVD, the I3D model internally resizes to 224x224, so we load videos at
    that resolution by default to save memory.
  - SSIM/PSNR/LPIPS are computed at the original video resolution (or a custom
    resolution via --pixel_metrics_resize).
  - FVD is a distributional metric and requires all videos at once.
  - CUDA_VISIBLE_DEVICES=0 is recommended on multi-GPU machines.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Put common_metrics_on_video_quality on sys.path so its calculate_* modules
# and the bundled lpips package can be imported.
# ---------------------------------------------------------------------------
METRICS_LIB_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "common_metrics_on_video_quality",
)
if METRICS_LIB_DIR not in sys.path:
    sys.path.insert(0, METRICS_LIB_DIR)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# Video loading
# =====================================================================

def load_video_as_tensor(
    path: str,
    resize: Optional[Tuple[int, int]] = None,
    max_frames: Optional[int] = None,
) -> torch.Tensor:
    """Load a video file and return a float32 tensor [T, C, H, W] in [0, 1].

    Parameters
    ----------
    path : str
        Path to the video file (mp4, avi, etc.).
    resize : (H, W) or None
        If given, resize every frame to this (height, width).
    max_frames : int or None
        If given, read at most this many frames.

    Returns
    -------
    torch.Tensor  –  shape [T, C, H, W], dtype float32, values in [0, 1].
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize is not None:
            # cv2.resize expects (width, height)
            frame = cv2.resize(frame, (resize[1], resize[0]), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Decoded 0 frames from: {path}")

    # Stack: (T, H, W, C)  ->  transpose to (T, C, H, W)  ->  float [0,1]
    video = np.stack(frames, axis=0)               # (T, H, W, C) uint8
    video = video.transpose(0, 3, 1, 2)            # (T, C, H, W)
    tensor = torch.from_numpy(video).float() / 255.0
    return tensor


def get_video_info(path: str) -> Dict:
    """Return basic video metadata without decoding all frames."""
    cap = cv2.VideoCapture(path)
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


# =====================================================================
# Manifest loading & pair building
# =====================================================================

def load_manifest(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a JSON array: {path}")
    return data


def build_pairs(
    manifest: list,
    gen_dir: str,
    gen_pattern: str,
) -> List[Dict]:
    """Build (gt_path, gen_path) pairs from manifest.

    Returns list of dicts:  {id, gt_path, gen_path, valid, skip_reason}
    """
    pairs = []
    for item in manifest:
        vid_id = item.get("id")
        gt_path = item.get("gt_video_path")
        if not vid_id or not gt_path:
            logger.warning("Skipping manifest entry missing 'id' or 'gt_video_path': %s", item)
            continue

        gen_filename = gen_pattern.replace("{id}", vid_id)
        gen_path = os.path.join(gen_dir, gen_filename)

        pair = {"id": vid_id, "gt_path": gt_path, "gen_path": gen_path}

        if not os.path.isfile(gt_path):
            pair["valid"] = False
            pair["skip_reason"] = f"GT not found: {gt_path}"
        elif not os.path.isfile(gen_path):
            pair["valid"] = False
            pair["skip_reason"] = f"Gen not found: {gen_path}"
        else:
            pair["valid"] = True
            pair["skip_reason"] = None

        pairs.append(pair)
    return pairs


# =====================================================================
# Metric computation helpers
# =====================================================================

def compute_pixel_metrics_single(
    gt_tensor: torch.Tensor,
    gen_tensor: torch.Tensor,
    metrics: set,
    device: torch.device,
) -> Dict:
    """Compute per-video SSIM/PSNR/LPIPS for one (GT, Gen) pair.

    gt_tensor, gen_tensor: [T, C, H, W] float tensors in [0, 1].
    Returns dict of metric_name -> value.
    """
    # Reshape to [1, T, C, H, W] (batch of 1)
    gt_batch = gt_tensor.unsqueeze(0)
    gen_batch = gen_tensor.unsqueeze(0)

    result = {}

    if "ssim" in metrics:
        from calculate_ssim import calculate_ssim
        r = calculate_ssim(gt_batch, gen_batch, only_final=True)
        result["ssim"] = float(r["value"][0])
        result["ssim_frame_std"] = float(r["value_std"][0])

    if "psnr" in metrics:
        from calculate_psnr import calculate_psnr
        r = calculate_psnr(gt_batch, gen_batch, only_final=True)
        result["psnr"] = float(r["value"][0])
        result["psnr_frame_std"] = float(r["value_std"][0])

    if "lpips" in metrics:
        from calculate_lpips import calculate_lpips
        r = calculate_lpips(gt_batch, gen_batch, device, only_final=True)
        result["lpips"] = float(r["value"][0])
        result["lpips_frame_std"] = float(r["value_std"][0])

    return result


def compute_fvd_batch(
    gt_tensors: List[torch.Tensor],
    gen_tensors: List[torch.Tensor],
    device: torch.device,
    method: str = "styleganv",
) -> float:
    """Compute FVD on batches of videos.

    gt_tensors, gen_tensors: lists of [T, C, H, W] tensors (same T for all).
    Returns the FVD value (scalar).
    """
    from calculate_fvd import calculate_fvd

    # Align to global minimum frame count
    min_t = min(v.shape[0] for v in gt_tensors + gen_tensors)
    if min_t < 10:
        raise ValueError(
            f"FVD requires >= 10 frames per video, but global minimum is {min_t}."
        )

    gt_batch = torch.stack([v[:min_t] for v in gt_tensors])   # [B, T, C, H, W]
    gen_batch = torch.stack([v[:min_t] for v in gen_tensors])  # [B, T, C, H, W]

    logger.info(
        "FVD batch shapes: GT=%s  Gen=%s  (%.1f MB each)",
        list(gt_batch.shape),
        list(gen_batch.shape),
        gt_batch.nelement() * 4 / 1e6,
    )

    r = calculate_fvd(gt_batch, gen_batch, device, method=method, only_final=True)
    return float(r["value"][0])


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch metrics evaluation (FVD / SSIM / PSNR / LPIPS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest", required=True,
        help="Path to manifest JSON file",
    )
    p.add_argument(
        "--gen_dir", required=True,
        help="Directory containing generated videos",
    )
    p.add_argument(
        "--gen_pattern", default="{id}_42_with_audio.mp4",
        help="Filename pattern, {id} will be replaced by manifest id "
             "(default: {id}_42_with_audio.mp4)",
    )
    p.add_argument(
        "--output_dir", required=True,
        help="Directory for output files",
    )
    p.add_argument(
        "--output_prefix", default="metrics",
        help="Prefix for output filenames (default: metrics)",
    )
    p.add_argument(
        "--metrics", nargs="+",
        default=["fvd", "ssim", "psnr", "lpips"],
        choices=["fvd", "ssim", "psnr", "lpips"],
        help="Metrics to compute (default: fvd ssim psnr lpips)",
    )
    p.add_argument(
        "--fvd_resize", type=int, nargs=2, default=[224, 224],
        metavar=("H", "W"),
        help="Resize resolution (H W) for FVD computation (default: 224 224). "
             "I3D internally uses 224x224 so this is safe to keep small.",
    )
    p.add_argument(
        "--pixel_metrics_resize", type=int, nargs=2, default=None,
        metavar=("H", "W"),
        help="Resize resolution (H W) for SSIM/PSNR/LPIPS. "
             "Default: use original video resolution.",
    )
    p.add_argument(
        "--fvd_method", default="styleganv",
        choices=["styleganv", "videogpt"],
        help="FVD implementation (default: styleganv)",
    )
    p.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to read per video (default: all)",
    )
    p.add_argument(
        "--device", default="auto",
        help="Torch device: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    # ---- Device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # ---- Metrics selection ----
    metrics_to_compute = set(args.metrics)
    pixel_metrics = metrics_to_compute & {"ssim", "psnr", "lpips"}
    do_fvd = "fvd" in metrics_to_compute
    logger.info("Metrics to compute: %s", sorted(metrics_to_compute))

    # ---- Load manifest & build pairs ----
    manifest = load_manifest(args.manifest)
    pairs = build_pairs(manifest, args.gen_dir, args.gen_pattern)
    valid_pairs = [p for p in pairs if p["valid"]]
    skipped = [p for p in pairs if not p["valid"]]

    logger.info(
        "Manifest: %d entries | Valid pairs: %d | Skipped: %d",
        len(manifest), len(valid_pairs), len(skipped),
    )
    for s in skipped:
        logger.warning("  SKIP [%s]: %s", s["id"], s["skip_reason"])

    if not valid_pairs:
        logger.error("No valid video pairs found. Exiting.")
        sys.exit(1)

    # ---- Prepare result structure ----
    results = {
        "config": {
            "manifest": args.manifest,
            "gen_dir": args.gen_dir,
            "gen_pattern": args.gen_pattern,
            "metrics": sorted(metrics_to_compute),
            "fvd_resize": args.fvd_resize if do_fvd else None,
            "pixel_metrics_resize": args.pixel_metrics_resize,
            "fvd_method": args.fvd_method if do_fvd else None,
            "max_frames": args.max_frames,
            "device": str(device),
            "num_valid_pairs": len(valid_pairs),
            "num_skipped": len(skipped),
        },
        "summary": {},
        "per_video": {},
    }

    # ---- FVD: accumulate resized videos during pixel-metrics pass ----
    fvd_gt_list: List[torch.Tensor] = []
    fvd_gen_list: List[torch.Tensor] = []
    fvd_resize = tuple(args.fvd_resize) if do_fvd else None

    # ==================================================================
    # Pass 1: Per-video pixel metrics (SSIM / PSNR / LPIPS)
    #         + accumulate FVD-resolution videos
    # ==================================================================
    logger.info("=" * 60)
    logger.info("Pass 1: Per-video metrics & FVD video loading")
    logger.info("=" * 60)

    pixel_resize = tuple(args.pixel_metrics_resize) if args.pixel_metrics_resize else None

    for pair in tqdm(valid_pairs, desc="Processing videos"):
        vid_id = pair["id"]
        per_video = {"id": vid_id}

        try:
            # ---- Load for pixel metrics (original or custom resolution) ----
            if pixel_metrics:
                gt = load_video_as_tensor(pair["gt_path"], resize=pixel_resize, max_frames=args.max_frames)
                gen = load_video_as_tensor(pair["gen_path"], resize=pixel_resize, max_frames=args.max_frames)

                # Align frame counts
                min_t = min(gt.shape[0], gen.shape[0])
                if gt.shape[0] != gen.shape[0]:
                    logger.warning(
                        "[%s] Frame count mismatch: GT=%d, Gen=%d -> using %d",
                        vid_id, gt.shape[0], gen.shape[0], min_t,
                    )
                gt = gt[:min_t]
                gen = gen[:min_t]

                # Check resolution match
                if gt.shape != gen.shape:
                    logger.error(
                        "[%s] Shape mismatch after alignment: GT=%s, Gen=%s. Skipping pixel metrics.",
                        vid_id, list(gt.shape), list(gen.shape),
                    )
                else:
                    pv = compute_pixel_metrics_single(gt, gen, pixel_metrics, device)
                    per_video.update(pv)

                del gt, gen  # free memory

            # ---- Load for FVD (at FVD resolution) ----
            if do_fvd:
                gt_fvd = load_video_as_tensor(pair["gt_path"], resize=fvd_resize, max_frames=args.max_frames)
                gen_fvd = load_video_as_tensor(pair["gen_path"], resize=fvd_resize, max_frames=args.max_frames)
                min_t_fvd = min(gt_fvd.shape[0], gen_fvd.shape[0])
                fvd_gt_list.append(gt_fvd[:min_t_fvd])
                fvd_gen_list.append(gen_fvd[:min_t_fvd])
                del gt_fvd, gen_fvd

        except Exception as e:
            logger.error("[%s] Error: %s", vid_id, e)
            per_video["error"] = str(e)

        results["per_video"][vid_id] = per_video

    # ==================================================================
    # Pass 2: FVD (distributional metric — needs all videos)
    # ==================================================================
    if do_fvd and fvd_gt_list:
        logger.info("=" * 60)
        logger.info("Pass 2: Computing FVD (%s)", args.fvd_method)
        logger.info("=" * 60)

        try:
            fvd_value = compute_fvd_batch(
                fvd_gt_list, fvd_gen_list, device, method=args.fvd_method,
            )
            results["summary"]["fvd"] = fvd_value
            logger.info("FVD (%s) = %.4f", args.fvd_method, fvd_value)
        except Exception as e:
            logger.error("FVD computation failed: %s", e)
            results["summary"]["fvd_error"] = str(e)

        del fvd_gt_list, fvd_gen_list

    # ==================================================================
    # Aggregate per-video pixel metrics into summary
    # ==================================================================
    for metric in sorted(pixel_metrics):
        values = [
            v[metric]
            for v in results["per_video"].values()
            if metric in v and not isinstance(v[metric], str)
        ]
        if values:
            arr = np.array(values, dtype=np.float64)
            results["summary"][metric] = float(np.mean(arr))
            results["summary"][f"{metric}_std"] = float(np.std(arr))
            results["summary"][f"{metric}_min"] = float(np.min(arr))
            results["summary"][f"{metric}_max"] = float(np.max(arr))
            results["summary"][f"{metric}_count"] = len(values)

    # ---- Timing ----
    elapsed = time.time() - t_start
    results["summary"]["elapsed_seconds"] = round(elapsed, 2)

    # ==================================================================
    # Save results
    # ==================================================================
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) JSON
    json_path = os.path.join(args.output_dir, f"{args.output_prefix}_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON results to: %s", json_path)

    # 2) CSV (per-video)
    csv_path = os.path.join(args.output_dir, f"{args.output_prefix}_per_video.csv")
    try:
        fieldnames = ["id"]
        # Collect all possible per-video keys
        all_keys = set()
        for v in results["per_video"].values():
            all_keys.update(k for k in v.keys() if k != "id")
        fieldnames.extend(sorted(all_keys))

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for vid_id in sorted(results["per_video"].keys()):
                writer.writerow(results["per_video"][vid_id])
        logger.info("Saved CSV results to: %s", csv_path)
    except Exception as e:
        logger.warning("Could not write CSV: %s", e)

    # ==================================================================
    # Print summary
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"  Evaluation Summary  ({len(valid_pairs)} videos, {elapsed:.1f}s)")
    print("=" * 60)
    for k, v in sorted(results["summary"].items()):
        if isinstance(v, float):
            print(f"  {k:20s} = {v:.6f}")
        else:
            print(f"  {k:20s} = {v}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
