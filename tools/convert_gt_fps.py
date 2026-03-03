#!/usr/bin/env python3
"""
Batch-convert GT videos listed in manifest JSON files to target FPS (default: 25).

Design:
1) Read one or more manifest files.
2) Collect unique gt_video_path entries.
3) For each video, use ffmpeg fps filter to resample while keeping playback speed.
4) Replace original file atomically via a temp output in the same directory.
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_manifest(manifest_path: Path):
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Manifest must be a list: {manifest_path}")
    return data


def collect_gt_paths(manifest_paths):
    gt_paths = []
    missing_key = 0
    for manifest_path in manifest_paths:
        records = load_manifest(manifest_path)
        for item in records:
            gt_path = item.get("gt_video_path")
            if not gt_path:
                missing_key += 1
                continue
            gt_paths.append(Path(gt_path))

    # Keep stable order while removing duplicates
    seen = set()
    unique_paths = []
    for p in gt_paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(p)

    if missing_key:
        logger.warning("%d manifest entries do not contain gt_video_path", missing_key)
    return unique_paths


def ffprobe_fps(video_path: Path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            return float(num) / float(den)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    try:
        return float(text)
    except ValueError:
        return None


def convert_video_inplace(video_path: Path, target_fps: float):
    tmp_path = video_path.with_name(f"{video_path.stem}.tmp_fps{int(target_fps)}{video_path.suffix}")
    codecs_to_try = ["libx264", "libopenh264", "mpeg4"]
    last_error = ""

    for codec in codecs_to_try:
        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-filter:v", f"fps=fps={target_fps}", "-c:v", codec]
        if codec in {"libx264", "libopenh264"}:
            cmd.extend(["-preset", "veryfast", "-crf", "18"])
        elif codec == "mpeg4":
            cmd.extend(["-q:v", "2"])
        cmd.extend(["-c:a", "copy", str(tmp_path)])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            break
        last_error = result.stderr.strip()
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    else:
        return False, last_error

    tmp_fps = ffprobe_fps(tmp_path)
    if tmp_fps is None:
        tmp_path.unlink(missing_ok=True)
        return False, "ffprobe failed on temporary output"

    # Atomic replace
    tmp_path.replace(video_path)
    return True, f"converted to {tmp_fps:.6f} fps"


def main():
    parser = argparse.ArgumentParser(description="Convert GT videos in manifests to a unified FPS.")
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        help="Path to a manifest JSON file. Can be specified multiple times.",
    )
    parser.add_argument("--target_fps", type=float, default=25.0, help="Target FPS (default: 25.0)")
    parser.add_argument(
        "--skip_if_close",
        action="store_true",
        help="Skip conversion if current FPS is already close to target (abs diff <= --fps_tolerance).",
    )
    parser.add_argument("--fps_tolerance", type=float, default=0.05, help="Tolerance for --skip_if_close")
    parser.add_argument("--dry_run", action="store_true", help="Only print actions without converting")
    args = parser.parse_args()

    manifest_paths = [Path(p) for p in args.manifest]
    for m in manifest_paths:
        if not m.exists():
            raise FileNotFoundError(f"Manifest not found: {m}")

    gt_paths = collect_gt_paths(manifest_paths)
    logger.info("Found %d unique GT videos from %d manifest(s)", len(gt_paths), len(manifest_paths))

    success = 0
    skipped = 0
    failed = 0

    for idx, video_path in enumerate(gt_paths, start=1):
        if not video_path.exists():
            failed += 1
            logger.error("[%d/%d] Missing file: %s", idx, len(gt_paths), video_path)
            continue

        src_fps = ffprobe_fps(video_path)
        logger.info("[%d/%d] %s | src_fps=%s", idx, len(gt_paths), video_path, f"{src_fps:.6f}" if src_fps else "unknown")

        if args.skip_if_close and src_fps is not None and abs(src_fps - args.target_fps) <= args.fps_tolerance:
            skipped += 1
            logger.info("  -> skip (already close to %.3f fps)", args.target_fps)
            continue

        if args.dry_run:
            skipped += 1
            logger.info("  -> dry-run skip (would convert to %.3f fps)", args.target_fps)
            continue

        ok, message = convert_video_inplace(video_path, args.target_fps)
        if ok:
            success += 1
            logger.info("  -> success: %s", message)
        else:
            failed += 1
            logger.error("  -> failed: %s", message)

    logger.info("Done. success=%d, skipped=%d, failed=%d, total=%d", success, skipped, failed, len(gt_paths))
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
