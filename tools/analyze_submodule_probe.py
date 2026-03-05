#!/usr/bin/env python3
"""
子模块级能量探测 – 聚合分析与可视化。

读取 submodule_probe_raw/ 下的 per-sample JSON 文件，聚合统计后生成：
  1. global_ranking.json          – 640 个 (step, block, sub) entry，按 energy 升序
  2. 热力图 (per-step, per-metric) – 4 steps × 3 metrics = 12 张 + 3 平均
  3. per-sample 热力图             – 每个样本独立生成
  4. cross_sample_analysis.json   – 跨样本方差分析 (真废块 vs 条件激活块)
  5. low_energy_zones.json        – 低能量暗区标记 (可配阈值)

用法：
    python tools/analyze_submodule_probe.py \
        --input_dir /root/autodl-fs/experiments/submodule_probe_raw \
        --output_dir /root/autodl-fs/experiments/submodule_probe_processed \
        --num_steps 4 --num_blocks 40 \
        --energy_threshold 0.01
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# Matplotlib 后端设置（无 GUI 支持）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

# ─────────────────────────────────────────
# 常量
# ─────────────────────────────────────────
SUBMODULE_NAMES = ("SA", "CCA", "ACA", "FFN")
METRICS = ("cos_sim", "res_mag", "energy")
METRIC_LABELS = {
    "cos_sim": "Cosine Similarity",
    "res_mag": "Relative L2 Magnitude",
    "energy": "Integrated Energy E=(1−cos)×M",
}


# ─────────────────────────────────────────
# 加载与解析
# ─────────────────────────────────────────

def load_probe_files(input_dir: str):
    """
    加载所有 *_submodule_probe.json 文件。

    Returns
    -------
    all_results : list[dict]
        每个元素是 {(step, block, sub): {metrics...}}
    sample_ids : list[str]
        对应的 sample ID
    """
    all_results = []
    sample_ids = []

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith("_submodule_probe.json")
    ])

    if not files:
        print(f"ERROR: No *_submodule_probe.json files found in {input_dir}")
        sys.exit(1)

    for fname in files:
        sample_id = fname.replace("_submodule_probe.json", "")
        filepath = os.path.join(input_dir, fname)
        with open(filepath, "r") as f:
            raw = json.load(f)

        parsed = {}
        for key, metrics in raw.items():
            parts = key.split("_", 2)
            step, block, sub = int(parts[0]), int(parts[1]), parts[2]
            parsed[(step, block, sub)] = metrics

        all_results.append(parsed)
        sample_ids.append(sample_id)

    print(f"Loaded {len(all_results)} probe files: {sample_ids}")
    return all_results, sample_ids


# ─────────────────────────────────────────
# 聚合
# ─────────────────────────────────────────

def aggregate_results(all_results, num_steps, num_blocks):
    """
    跨样本聚合，计算每个 (step, block, sub) 的均值和标准差。

    Returns
    -------
    aggregated : dict
        {(step, block, sub): {metric: mean, metric_std: std, ...}}
    """
    collector = defaultdict(lambda: defaultdict(list))

    for sample_data in all_results:
        for (step, block, sub), metrics in sample_data.items():
            for k, v in metrics.items():
                collector[(step, block, sub)][k].append(v)

    aggregated = {}
    for key, metric_lists in collector.items():
        entry = {"n_samples": len(all_results)}
        for metric_name, values in metric_lists.items():
            arr = np.array(values, dtype=np.float64)
            entry[metric_name] = float(np.mean(arr))
            entry[f"{metric_name}_std"] = float(np.std(arr))
        aggregated[key] = entry

    # 填充缺失的 key（防御性）
    for step in range(num_steps):
        for block in range(num_blocks):
            for sub in SUBMODULE_NAMES:
                key = (step, block, sub)
                if key not in aggregated:
                    aggregated[key] = {
                        "cos_sim": 1.0, "res_mag": 0.0, "energy": 0.0,
                        "cos_sim_std": 0.0, "res_mag_std": 0.0, "energy_std": 0.0,
                        "n_samples": 0,
                    }

    return aggregated


# ─────────────────────────────────────────
# 排名
# ─────────────────────────────────────────

def generate_ranking(aggregated):
    """
    按 energy 升序排列（最冗余在前）。

    Returns
    -------
    ranking : list[dict]
    """
    ranking = []
    for (step, block, sub), metrics in aggregated.items():
        entry = {
            "step": step,
            "block_idx": block,
            "submodule": sub,
        }
        entry.update(metrics)
        # 跨样本 energy 方差（区分真废块 vs 条件激活块）
        entry["cross_sample_energy_var"] = metrics.get("energy_std", 0.0) ** 2
        # 分类
        if metrics.get("energy", 0) < 0.01 and entry["cross_sample_energy_var"] < 1e-5:
            entry["classification"] = "truly_redundant"
        elif metrics.get("energy", 0) < 0.01:
            entry["classification"] = "conditionally_active"
        else:
            entry["classification"] = "active"
        ranking.append(entry)

    ranking.sort(key=lambda x: x["energy"])
    return ranking


# ─────────────────────────────────────────
# 热力图生成
# ─────────────────────────────────────────

def _build_matrix(data_dict, step, num_blocks, metric):
    """
    构建 [num_blocks, 4] 矩阵用于热力图。

    轴：行 = block_idx (0..39)，列 = SA/CCA/ACA/FFN
    """
    mat = np.full((num_blocks, len(SUBMODULE_NAMES)), np.nan)
    for b in range(num_blocks):
        for s_idx, sub in enumerate(SUBMODULE_NAMES):
            key = (step, b, sub)
            if key in data_dict:
                mat[b, s_idx] = data_dict[key].get(metric, np.nan)
    return mat


def _build_avg_matrix(data_dict, num_steps, num_blocks, metric):
    """构建跨 step 平均的矩阵。"""
    mats = []
    for step in range(num_steps):
        mats.append(_build_matrix(data_dict, step, num_blocks, metric))
    return np.nanmean(mats, axis=0)


def plot_heatmap(matrix, title, xlabel, ylabel, xtick_labels, output_path,
                 cmap="RdYlGn", vmin=None, vmax=None, figsize=(6, 16)):
    """通用热力图绘制。"""
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")

    ax.set_xticks(range(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # y 轴每行标注 block index
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels([str(i) for i in range(matrix.shape[0])], fontsize=7)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # 数值标注
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > (vmax or 1) * 0.7 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=5.5, color=text_color)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_heatmaps(aggregated, num_steps, num_blocks, output_dir):
    """生成 per-step 和 average 热力图。"""
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    # 配色方案
    cmap_configs = {
        "cos_sim": ("RdYlGn", None, None),        # 高=绿(好), 低=红(冗余?)
        "res_mag": ("YlOrRd", 0, None),            # 高=红, 低=黄
        "energy":  ("YlOrRd", 0, None),            # 高=红, 低=黄(冗余)
    }

    for metric in METRICS:
        cmap, vmin, vmax = cmap_configs[metric]

        # Per-step 热力图
        all_vals = []
        for step in range(num_steps):
            mat = _build_matrix(aggregated, step, num_blocks, metric)
            all_vals.extend(mat[~np.isnan(mat)].tolist())

        # 统一色彩范围
        if all_vals:
            if vmin is None:
                vmin_use = np.percentile(all_vals, 2)
            else:
                vmin_use = vmin
            if vmax is None:
                vmax_use = np.percentile(all_vals, 98)
            else:
                vmax_use = vmax
        else:
            vmin_use, vmax_use = 0, 1

        for step in range(num_steps):
            mat = _build_matrix(aggregated, step, num_blocks, metric)
            plot_heatmap(
                mat,
                title=f"{METRIC_LABELS[metric]}\nStep {step}",
                xlabel="Sub-module",
                ylabel="Block Index",
                xtick_labels=list(SUBMODULE_NAMES),
                output_path=os.path.join(heatmap_dir, f"step{step}_{metric}.png"),
                cmap=cmap, vmin=vmin_use, vmax=vmax_use,
            )

        # Average 热力图
        avg_mat = _build_avg_matrix(aggregated, num_steps, num_blocks, metric)
        plot_heatmap(
            avg_mat,
            title=f"{METRIC_LABELS[metric]}\nAverage across {num_steps} steps",
            xlabel="Sub-module",
            ylabel="Block Index",
            xtick_labels=list(SUBMODULE_NAMES),
            output_path=os.path.join(heatmap_dir, f"avg_{metric}.png"),
            cmap=cmap, vmin=vmin_use, vmax=vmax_use,
        )


def generate_per_sample_heatmaps(all_results, sample_ids, num_steps, num_blocks,
                                 output_dir):
    """为每个样本生成独立热力图。"""
    per_sample_dir = os.path.join(output_dir, "per_sample_heatmaps")
    os.makedirs(per_sample_dir, exist_ok=True)

    for sample_data, sample_id in zip(all_results, sample_ids):
        for metric in ("cos_sim", "res_mag", "energy"):
            for step in range(num_steps):
                mat = _build_matrix(sample_data, step, num_blocks, metric)
                plot_heatmap(
                    mat,
                    title=f"{sample_id} – {METRIC_LABELS[metric]} – Step {step}",
                    xlabel="Sub-module",
                    ylabel="Block Index",
                    xtick_labels=list(SUBMODULE_NAMES),
                    output_path=os.path.join(
                        per_sample_dir,
                        f"{sample_id}_step{step}_{metric}.png"
                    ),
                    cmap="YlOrRd" if metric != "cos_sim" else "RdYlGn",
                )


# ─────────────────────────────────────────
# 跨样本方差分析
# ─────────────────────────────────────────

def cross_sample_analysis(aggregated, energy_threshold, output_dir):
    """
    分类模块：truly_redundant / conditionally_active / active。
    """
    analysis = {
        "truly_redundant": [],
        "conditionally_active": [],
        "active_count": 0,
        "energy_threshold": energy_threshold,
    }

    for (step, block, sub), metrics in aggregated.items():
        energy = metrics.get("energy", 0)
        energy_std = metrics.get("energy_std", 0)

        if energy < energy_threshold and energy_std < energy_threshold * 0.1:
            analysis["truly_redundant"].append({
                "step": step, "block_idx": block, "submodule": sub,
                "energy": energy, "energy_std": energy_std,
            })
        elif energy < energy_threshold:
            analysis["conditionally_active"].append({
                "step": step, "block_idx": block, "submodule": sub,
                "energy": energy, "energy_std": energy_std,
            })
        else:
            analysis["active_count"] += 1

    analysis["truly_redundant"].sort(key=lambda x: x["energy"])
    analysis["conditionally_active"].sort(key=lambda x: x["energy"])

    out_path = os.path.join(output_dir, "cross_sample_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved cross-sample analysis to {out_path}")
    print(f"  Truly redundant:      {len(analysis['truly_redundant'])}")
    print(f"  Conditionally active: {len(analysis['conditionally_active'])}")
    print(f"  Active:               {analysis['active_count']}")

    return analysis


# ─────────────────────────────────────────
# 低能量暗区
# ─────────────────────────────────────────

def identify_low_energy_zones(ranking, energy_threshold, output_dir):
    """
    标记 energy < threshold 的 entry 为低能量暗区。
    """
    zones = [
        entry for entry in ranking
        if entry["energy"] < energy_threshold
    ]
    out_path = os.path.join(output_dir, "low_energy_zones.json")
    with open(out_path, "w") as f:
        json.dump(zones, f, indent=2)
    print(f"Saved {len(zones)} low-energy zones to {out_path}")
    return zones


# ─────────────────────────────────────────
# Modulation scale 分析
# ─────────────────────────────────────────

def analyze_modulation_scales(aggregated, num_steps, num_blocks, output_dir):
    """
    提取 SA/FFN 的 mod_scale 均值，辅助判断网络主动抑制 vs 子模块本身低能量。
    """
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    for sub in ("SA", "FFN"):
        for step in range(num_steps):
            mat = np.full((num_blocks, 1), np.nan)
            for b in range(num_blocks):
                key = (step, b, sub)
                if key in aggregated and "mod_scale" in aggregated[key]:
                    mat[b, 0] = aggregated[key]["mod_scale"]

            fig, ax = plt.subplots(figsize=(2.5, 16))
            im = ax.imshow(mat, aspect="auto", cmap="Blues", interpolation="nearest")
            ax.set_xticks([0])
            ax.set_xticklabels([f"|e| ({sub})"], fontsize=9)
            ax.set_ylabel("Block Index", fontsize=11)
            ax.set_title(f"AdaLN Mod Scale – {sub}\nStep {step}", fontsize=11,
                         fontweight="bold")
            ax.set_yticks(range(num_blocks))
            ax.set_yticklabels([str(i) for i in range(num_blocks)], fontsize=7)
            for i in range(num_blocks):
                if not np.isnan(mat[i, 0]):
                    ax.text(0, i, f"{mat[i,0]:.4f}", ha="center", va="center",
                            fontsize=6)
            plt.colorbar(im, ax=ax, fraction=0.08, pad=0.04)
            plt.tight_layout()
            out_path = os.path.join(heatmap_dir, f"step{step}_mod_scale_{sub}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ─────────────────────────────────────────
# Summary 统计
# ─────────────────────────────────────────

def print_summary(ranking, num_steps, num_blocks):
    """打印前 20 个最冗余和最活跃的 entry。"""
    total = num_steps * num_blocks * len(SUBMODULE_NAMES)
    print(f"\n{'='*70}")
    print(f"Global Ranking Summary ({len(ranking)}/{total} entries)")
    print(f"{'='*70}")

    print(f"\n{'─'*70}")
    print("Top 20 MOST REDUNDANT (lowest energy):")
    print(f"{'─'*70}")
    print(f"{'Rank':>4}  {'Step':>4}  {'Block':>5}  {'Sub':>4}  "
          f"{'cos_sim':>8}  {'res_mag':>8}  {'energy':>10}  {'e_std':>8}  {'class':>20}")
    for i, e in enumerate(ranking[:20]):
        print(f"{i+1:>4}  {e['step']:>4}  {e['block_idx']:>5}  {e['submodule']:>4}  "
              f"{e['cos_sim']:>8.4f}  {e['res_mag']:>8.4f}  {e['energy']:>10.6f}  "
              f"{e.get('energy_std', 0):>8.6f}  {e.get('classification', ''):>20}")

    print(f"\n{'─'*70}")
    print("Top 20 MOST ACTIVE (highest energy):")
    print(f"{'─'*70}")
    for i, e in enumerate(reversed(ranking[-20:])):
        print(f"{i+1:>4}  {e['step']:>4}  {e['block_idx']:>5}  {e['submodule']:>4}  "
              f"{e['cos_sim']:>8.4f}  {e['res_mag']:>8.4f}  {e['energy']:>10.6f}  "
              f"{e.get('energy_std', 0):>8.6f}  {e.get('classification', ''):>20}")

    # Per-submodule 统计
    print(f"\n{'─'*70}")
    print("Per-submodule average energy (across all steps and blocks):")
    print(f"{'─'*70}")
    for sub in SUBMODULE_NAMES:
        energies = [e["energy"] for e in ranking if e["submodule"] == sub]
        if energies:
            print(f"  {sub:>4}:  mean={np.mean(energies):.6f}  "
                  f"std={np.std(energies):.6f}  "
                  f"min={np.min(energies):.6f}  "
                  f"max={np.max(energies):.6f}")

    # Per-step 统计
    print(f"\n{'─'*70}")
    print("Per-step average energy (across all blocks and sub-modules):")
    print(f"{'─'*70}")
    for step in range(max(e["step"] for e in ranking) + 1):
        energies = [e["energy"] for e in ranking if e["step"] == step]
        if energies:
            print(f"  Step {step}:  mean={np.mean(energies):.6f}  "
                  f"std={np.std(energies):.6f}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="子模块级能量探测 – 聚合分析与可视化"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="包含 *_submodule_probe.json 文件的目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="分析结果输出目录")
    parser.add_argument("--num_steps", type=int, default=4,
                        help="去噪步数")
    parser.add_argument("--num_blocks", type=int, default=40,
                        help="Transformer block 数量")
    parser.add_argument("--energy_threshold", type=float, default=0.01,
                        help="低能量暗区阈值")
    parser.add_argument("--skip_per_sample", action="store_true",
                        help="跳过 per-sample 热力图生成（节省时间）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 加载 ──
    all_results, sample_ids = load_probe_files(args.input_dir)

    # ── 聚合 ──
    print("\nAggregating results across samples...")
    aggregated = aggregate_results(all_results, args.num_steps, args.num_blocks)

    # ── 排名 ──
    print("Generating global ranking...")
    ranking = generate_ranking(aggregated)
    ranking_path = os.path.join(args.output_dir, "global_ranking.json")
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2)
    print(f"Saved global ranking to {ranking_path} ({len(ranking)} entries)")

    # ── 热力图 ──
    print("\nGenerating aggregated heatmaps...")
    generate_heatmaps(aggregated, args.num_steps, args.num_blocks, args.output_dir)

    # ── Per-sample 热力图 ──
    if not args.skip_per_sample:
        print("\nGenerating per-sample heatmaps...")
        generate_per_sample_heatmaps(
            all_results, sample_ids,
            args.num_steps, args.num_blocks, args.output_dir
        )

    # ── Modulation scale 分析 ──
    print("\nAnalyzing modulation scales (e[2], e[5])...")
    analyze_modulation_scales(aggregated, args.num_steps, args.num_blocks,
                              args.output_dir)

    # ── 跨样本方差分析 ──
    print("\nRunning cross-sample analysis...")
    cross_sample_analysis(aggregated, args.energy_threshold, args.output_dir)

    # ── 低能量暗区 ──
    print("\nIdentifying low-energy zones...")
    identify_low_energy_zones(ranking, args.energy_threshold, args.output_dir)

    # ── 摘要 ──
    print_summary(ranking, args.num_steps, args.num_blocks)

    print(f"\n{'='*70}")
    print(f"Analysis complete. All outputs saved to: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
