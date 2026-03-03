#!/usr/bin/env python3
"""
Phase 2 – Step 13: Aggregate and analyze probe results.

Reads per-video probe JSON files (from run_probe.sh), computes mean
redundancy metrics across calibration videos, produces:
  1. global_ranking.json       – all (step, block) entries sorted by descending CFPS
  2. heatmap_cfps.png          – 4×N_blocks heatmap of CFPS values
  3. heatmap_cos_sim.png       – 4×N_blocks heatmap of cos_sim values
  4. heatmap_res_mag.png       – 4×N_blocks heatmap of res_mag values
  5. skip_list_XX.json         – pruning skip-lists at 10%/20%/30%/40%/50% ratios

Usage:
  python tools/analyze_probe.py \
      --input_dir /root/autodl-fs/experiments/probe_results \
      --output_dir /root/autodl-fs/experiments/probe_results \
      --num_steps 4 --num_blocks 40 --probe_alpha 0.75
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load_probe_files(input_dir: str):
    """Load all *_probe.json files from input_dir.
    
    Each file has keys like "0_5" -> {"cos_sim": ..., "res_mag": ...}
    where "0_5" means step=0, block_idx=5.
    
    Also supports legacy format with "cfps" field (will be ignored; cfps is
    recomputed offline with the user-specified alpha).
    
    Returns:
        all_results: list of dicts, each dict mapping (step, block_idx) -> metrics dict
        per_sample_results: list of tuples (sample_id, parsed_dict)
    """
    pattern = os.path.join(input_dir, "*_probe.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No *_probe.json files found in {input_dir}")
        sys.exit(1)
    
    all_results = []
    per_sample_results = []
    for fp in files:
        with open(fp, "r") as f:
            data = json.load(f)
        # Parse string keys "step_blockidx" back to (step, block_idx) tuples
        parsed = {}
        for key_str, metrics in data.items():
            parts = key_str.split("_")
            step = int(parts[0])
            block_idx = int(parts[1])
            parsed[(step, block_idx)] = metrics
        sample_id = os.path.basename(fp).replace("_probe.json", "")
        all_results.append(parsed)
        per_sample_results.append((sample_id, parsed))
        print(f"  Loaded {fp}: {len(parsed)} entries")
    
    print(f"Loaded {len(all_results)} probe files total.")
    return all_results, per_sample_results


def aggregate_results(all_results: list, num_steps: int, num_blocks: int, probe_alpha: float):
    """Compute per-(step, block) mean of metrics across all videos.
    
    CFPS is computed here from raw cos_sim and res_mag using the given alpha:
        cfps = alpha * cos_sim - (1 - alpha) * res_mag
    
    Returns:
        dict mapping (step, block_idx) -> {"cos_sim": float, "res_mag": float, "cfps": float,
                                            "cos_sim_std": float, "res_mag_std": float, "cfps_std": float}
    """
    # Accumulate raw values per position
    accum = defaultdict(lambda: {"cos_sim": [], "res_mag": []})
    
    for result in all_results:
        for (step, block_idx), metrics in result.items():
            accum[(step, block_idx)]["cos_sim"].append(metrics["cos_sim"])
            accum[(step, block_idx)]["res_mag"].append(metrics["res_mag"])
    
    # Compute means, stds, and CFPS from raw metrics
    aggregated = {}
    for step in range(num_steps):
        for block_idx in range(num_blocks):
            key = (step, block_idx)
            if key in accum:
                vals = accum[key]
                cos_arr = np.array(vals["cos_sim"])
                res_arr = np.array(vals["res_mag"])
                cfps_arr = probe_alpha * cos_arr - (1 - probe_alpha) * res_arr
                aggregated[key] = {
                    "cos_sim": float(np.mean(cos_arr)),
                    "res_mag": float(np.mean(res_arr)),
                    "cfps": float(np.mean(cfps_arr)),
                    "cos_sim_std": float(np.std(cos_arr)),
                    "res_mag_std": float(np.std(res_arr)),
                    "cfps_std": float(np.std(cfps_arr)),
                    "n_samples": len(cos_arr),
                }
            else:
                print(f"  WARNING: no data for step={step}, block={block_idx}")
                aggregated[key] = {
                    "cos_sim": 0.0, "res_mag": 0.0, "cfps": 0.0,
                    "cos_sim_std": 0.0, "res_mag_std": 0.0, "cfps_std": 0.0,
                    "n_samples": 0,
                }
    
    return aggregated


def save_global_ranking(aggregated: dict, output_dir: str):
    """Sort by CFPS descending and save global_ranking.json."""
    ranking = []
    for (step, block_idx), metrics in aggregated.items():
        entry = {
            "step": step,
            "block_idx": block_idx,
            "cos_sim": metrics["cos_sim"],
            "res_mag": metrics["res_mag"],
            "cfps": metrics["cfps"],
            "cos_sim_std": metrics["cos_sim_std"],
            "res_mag_std": metrics["res_mag_std"],
            "cfps_std": metrics["cfps_std"],
            "n_samples": metrics["n_samples"],
        }
        ranking.append(entry)
    
    # Sort by CFPS descending (higher CFPS = more redundant = safer to skip)
    ranking.sort(key=lambda x: x["cfps"], reverse=True)
    
    out_path = os.path.join(output_dir, "global_ranking.json")
    with open(out_path, "w") as f:
        json.dump(ranking, f, indent=2)
    print(f"Saved global ranking ({len(ranking)} entries) to {out_path}")
    
    # Print top-10 and bottom-10 for quick inspection
    print("\n=== Top-10 most redundant (step, block) pairs (safest to skip) ===")
    print(f"{'Rank':>4}  {'Step':>4}  {'Block':>5}  {'CFPS':>8}  {'CosSim':>8}  {'ResMag':>8}")
    for i, entry in enumerate(ranking[:10]):
        print(f"{i+1:>4}  {entry['step']:>4}  {entry['block_idx']:>5}  "
              f"{entry['cfps']:>8.5f}  {entry['cos_sim']:>8.5f}  {entry['res_mag']:>8.5f}")
    
    print("\n=== Bottom-10 least redundant (step, block) pairs (critical, do NOT skip) ===")
    for i, entry in enumerate(ranking[-10:]):
        print(f"{len(ranking)-9+i:>4}  {entry['step']:>4}  {entry['block_idx']:>5}  "
              f"{entry['cfps']:>8.5f}  {entry['cos_sim']:>8.5f}  {entry['res_mag']:>8.5f}")
    
    return ranking


def generate_heatmaps(aggregated: dict, num_steps: int, num_blocks: int, output_dir: str):
    """Generate heatmap PNGs for CFPS, cos_sim, and res_mag."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("WARNING: matplotlib/seaborn not available. Skipping heatmap generation.")
        print("Install with: pip install matplotlib seaborn")
        return
    
    metrics_to_plot = [
        ("cfps", "CFPS (Composite Redundancy Score)", "RdYlGn"),
        ("cos_sim", "Cosine Similarity (S_cos)", "YlOrRd"),
        ("res_mag", "Relative Residual Magnitude (M_res)", "YlOrRd_r"),
    ]
    
    for metric_key, title, cmap in metrics_to_plot:
        # Build matrix: rows=steps, cols=blocks
        matrix = np.zeros((num_steps, num_blocks))
        for step in range(num_steps):
            for block_idx in range(num_blocks):
                matrix[step, block_idx] = aggregated[(step, block_idx)][metric_key]
        
        fig, ax = plt.subplots(figsize=(max(16, num_blocks * 0.45), max(3, num_steps * 0.8 + 1.5)))
        
        sns.heatmap(
            matrix,
            ax=ax,
            annot=True if num_blocks <= 40 else False,
            fmt=".3f" if num_blocks <= 40 else "",
            cmap=cmap,
            xticklabels=[str(i) for i in range(num_blocks)],
            yticklabels=[f"Step {i}" for i in range(num_steps)],
            linewidths=0.5,
            annot_kws={"size": 6} if num_blocks <= 40 else {},
        )
        ax.set_xlabel("Block Index")
        ax.set_ylabel("Denoise Step")
        ax.set_title(f"{title}\n(Mean over calibration videos)")
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"heatmap_{metric_key}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved heatmap: {out_path}")


def generate_per_sample_heatmaps(per_sample_results: list, num_steps: int, num_blocks: int, output_dir: str):
    """Generate per-sample heatmaps for cos_sim and res_mag.

    Outputs are saved to: output_dir/per_sample_heatmaps/
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("WARNING: matplotlib/seaborn not available. Skipping per-sample heatmap generation.")
        print("Install with: pip install matplotlib seaborn")
        return

    per_sample_dir = os.path.join(output_dir, "per_sample_heatmaps")
    os.makedirs(per_sample_dir, exist_ok=True)

    metrics_to_plot = [
        ("cos_sim", "Cosine Similarity (S_cos)", "YlOrRd"),
        ("res_mag", "Relative Residual Magnitude (M_res)", "YlOrRd_r"),
    ]

    print(f"\n=== Generating per-sample heatmaps to {per_sample_dir} ===")
    for sample_id, sample_data in per_sample_results:
        for metric_key, title, cmap in metrics_to_plot:
            matrix = np.zeros((num_steps, num_blocks))
            for step in range(num_steps):
                for block_idx in range(num_blocks):
                    key = (step, block_idx)
                    if key in sample_data:
                        matrix[step, block_idx] = sample_data[key][metric_key]

            fig, ax = plt.subplots(figsize=(max(16, num_blocks * 0.45), max(3, num_steps * 0.8 + 1.5)))

            sns.heatmap(
                matrix,
                ax=ax,
                annot=True if num_blocks <= 40 else False,
                fmt=".3f" if num_blocks <= 40 else "",
                cmap=cmap,
                xticklabels=[str(i) for i in range(num_blocks)],
                yticklabels=[f"Step {i}" for i in range(num_steps)],
                linewidths=0.5,
                annot_kws={"size": 6} if num_blocks <= 40 else {},
            )
            ax.set_xlabel("Block Index")
            ax.set_ylabel("Denoise Step")
            ax.set_title(f"{title}\n(Sample: {sample_id})")
            plt.tight_layout()

            out_path = os.path.join(per_sample_dir, f"{sample_id}_{metric_key}.png")
            fig.savefig(out_path, dpi=150)
            plt.close(fig)

    print(f"Generated {len(per_sample_results) * 2} per-sample heatmaps.")


def generate_skip_lists(ranking: list, num_total: int, output_dir: str):
    """Generate skip-list JSONs for different pruning ratios.
    
    Each skip-list is a list of [step, block_idx] pairs to skip.
    Entries are selected from the top of the ranking (highest CFPS = most redundant).
    """
    ratios = [0.10, 0.20, 0.30, 0.40, 0.50]
    
    print(f"\n=== Generating skip lists (total positions: {num_total}) ===")
    for ratio in ratios:
        n_skip = int(round(num_total * ratio))
        skip_entries = ranking[:n_skip]
        skip_list = [[entry["step"], entry["block_idx"]] for entry in skip_entries]
        
        pct = int(ratio * 100)
        out_path = os.path.join(output_dir, f"skip_list_{pct}pct.json")
        with open(out_path, "w") as f:
            json.dump(skip_list, f, indent=2)
        
        if skip_entries:
            min_cfps = skip_entries[-1]["cfps"]
            max_cfps = skip_entries[0]["cfps"]
        else:
            min_cfps = max_cfps = 0.0
        
        print(f"  {pct}%: {n_skip} pairs, CFPS range [{min_cfps:.5f}, {max_cfps:.5f}] -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze probe results from Phase 2 calibration runs."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing *_probe.json files from probe inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save analysis outputs. Defaults to --input_dir.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="Number of denoising steps (default: 4).",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=40,
        help="Number of DiT blocks (default: 40 for 19B model).",
    )
    parser.add_argument(
        "--probe_alpha",
        type=float,
        default=0.75,
        help="Alpha coefficient for CFPS: cfps = alpha * cos_sim - (1-alpha) * res_mag. "
             "Adjustable without re-running inference (default: 0.75).",
    )
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Input dir:    {args.input_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"probe_alpha:  {args.probe_alpha}")
    print(f"CFPS formula: cfps = {args.probe_alpha} * cos_sim - {1 - args.probe_alpha} * res_mag")
    print(f"Expected grid: {args.num_steps} steps x {args.num_blocks} blocks = "
          f"{args.num_steps * args.num_blocks} positions\n")
    
    # Step 1: Load all probe files
    all_results, per_sample_results = load_probe_files(args.input_dir)
    
    # Step 2: Aggregate across videos and compute CFPS offline
    aggregated = aggregate_results(all_results, args.num_steps, args.num_blocks, args.probe_alpha)
    
    # Step 3: Save global ranking
    ranking = save_global_ranking(aggregated, args.output_dir)
    
    # Step 4: Generate heatmaps
    generate_heatmaps(aggregated, args.num_steps, args.num_blocks, args.output_dir)

    # Step 4.5: Generate per-sample heatmaps (cos_sim/res_mag)
    generate_per_sample_heatmaps(per_sample_results, args.num_steps, args.num_blocks, args.output_dir)
    
    # Step 5: Generate skip lists for different pruning ratios
    num_total = args.num_steps * args.num_blocks
    generate_skip_lists(ranking, num_total, args.output_dir)
    
    print("\n=== Analysis complete ===")


if __name__ == "__main__":
    main()
