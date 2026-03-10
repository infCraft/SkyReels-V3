#!/usr/bin/env python3
"""
Block 任务可视化脚本

读取 cka_extract.py 产出的空间统计数据，生成以下可视化：
1. 残差 L2 范数空间热力图 (Residual Energy Map)
2. 残差 PCA 主成分 RGB 映射 (Residual PCA)
3. Block 空间集中度分析 (Gini 系数)

用法:
    python scripts/block_task_visualize.py \
        --input_dir /root/autodl-fs/experiments/cka_analysis/spatial_stats \
        --output_dir /root/autodl-fs/experiments/cka_analysis/visualizations \
        --sample_id hdtf_0000
    或者不指定 --sample_id 则对所有样本取平均（仅对 res_norm 有效，PCA 保留第一个样本）
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

NUM_BLOCKS = 40
NUM_STEPS = 4


def load_spatial_stats(input_dir, sample_id=None):
    """
    加载空间统计数据。

    返回:
        stats_dict: {(step, block): {'res_norm': Tensor, 'pca_proj': Tensor, 'fft_mag': Tensor}}
    """
    if sample_id:
        sample_dirs = [os.path.join(input_dir, sample_id)]
    else:
        sample_dirs = sorted([
            os.path.join(input_dir, d) for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d)) and not d.startswith(".")
        ])

    if not sample_dirs:
        raise FileNotFoundError(f"No sample directories found in {input_dir}")

    print(f"Loading from {len(sample_dirs)} sample(s)...")

    # 累加和计数
    sum_stats = {}
    count = 0

    for sample_dir in sample_dirs:
        if not os.path.isdir(sample_dir):
            print(f"Warning: {sample_dir} not found, skipping")
            continue

        for step in range(NUM_STEPS):
            for blk in range(NUM_BLOCKS):
                fpath = os.path.join(sample_dir, f"step{step}_block{blk}.pt")
                if not os.path.exists(fpath):
                    print(f"Warning: missing {fpath}")
                    continue
                data = torch.load(fpath, map_location="cpu", weights_only=True)
                key = (step, blk)
                if key not in sum_stats:
                    sum_stats[key] = {
                        'res_norm': data['res_norm'].float(),
                        'pca_proj': data['pca_proj'].float(),
                    }
                else:
                    sum_stats[key]['res_norm'] += data['res_norm'].float()
                    # pca_proj 不累加: 不同样本的 PCA 主成分方向不同，跨样本平均投影值无物理意义
                    # 保留第一个样本的 pca_proj
        count += 1

    if count > 1:
        print(f"WARNING: PCA projections are from the first sample only "
              f"(cross-sample PCA averaging is mathematically invalid).")
        print(f"         For PCA visualization, use --sample_id to specify a single sample.")
        for key in sum_stats:
            sum_stats[key]['res_norm'] /= count
            # pca_proj 不做平均（保留第一个样本的值）

    print(f"Loaded {len(sum_stats)} entries from {count} sample(s)")
    return sum_stats


def plot_residual_energy_map(stats, output_dir):
    """
    图表1: 残差 L2 范数空间热力图。

    取中间帧 (frame T//2)，绘制 40(blocks) × 4(steps) 的子图矩阵。
    亮区 = block 在此空间位置改变特征最多。
    """
    # 获取网格尺寸
    sample_key = next(iter(stats))
    T, H, W = stats[sample_key]['res_norm'].shape
    mid_frame = T // 2

    fig, axes = plt.subplots(NUM_BLOCKS, NUM_STEPS, figsize=(NUM_STEPS * 3, NUM_BLOCKS * 1.2))

    # 全局 vmax 用于统一色阶
    all_vals = []
    for (step, blk), data in stats.items():
        all_vals.append(data['res_norm'][mid_frame].numpy())
    global_vmax = np.percentile(np.concatenate([v.flatten() for v in all_vals]), 98)
    global_vmin = 0.0

    for blk in range(NUM_BLOCKS):
        for step in range(NUM_STEPS):
            ax = axes[blk, step]
            key = (step, blk)
            if key in stats:
                frame_data = stats[key]['res_norm'][mid_frame].numpy()
                ax.imshow(frame_data, cmap="hot", vmin=global_vmin, vmax=global_vmax,
                          interpolation="bilinear", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if step == 0:
                ax.set_ylabel(f"B{blk}", fontsize=6, rotation=0, labelpad=15)
            if blk == 0:
                ax.set_title(f"Step {step}", fontsize=9)

    fig.suptitle(f"Residual L2 Norm Spatial Map (Frame {mid_frame}/{T})\n"
                 f"Bright = High residual energy (block modifies features here)",
                 fontsize=12, y=1.001)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "block_residual_energy_map.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_residual_energy_map_compact(stats, output_dir):
    """
    图表1b: 紧凑版残差能量图 — 选取代表性 blocks (每隔4个)。
    """
    sample_key = next(iter(stats))
    T, H, W = stats[sample_key]['res_norm'].shape
    mid_frame = T // 2

    selected_blocks = list(range(0, NUM_BLOCKS, 4))  # 0, 4, 8, ..., 36
    n_sel = len(selected_blocks)

    fig, axes = plt.subplots(n_sel, NUM_STEPS, figsize=(NUM_STEPS * 3.5, n_sel * 2.2))

    all_vals = []
    for (step, blk), data in stats.items():
        if blk in selected_blocks:
            all_vals.append(data['res_norm'][mid_frame].numpy())
    global_vmax = np.percentile(np.concatenate([v.flatten() for v in all_vals]), 98)

    for row_idx, blk in enumerate(selected_blocks):
        for step in range(NUM_STEPS):
            ax = axes[row_idx, step]
            key = (step, blk)
            if key in stats:
                frame_data = stats[key]['res_norm'][mid_frame].numpy()
                im = ax.imshow(frame_data, cmap="hot", vmin=0, vmax=global_vmax,
                               interpolation="bilinear", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if step == 0:
                ax.set_ylabel(f"Block {blk}", fontsize=10, rotation=0, labelpad=40)
            if row_idx == 0:
                ax.set_title(f"Step {step}", fontsize=11, fontweight="bold")

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Residual L2 Norm")

    fig.suptitle(f"Residual Energy Map — Selected Blocks (Frame {mid_frame}/{T})",
                 fontsize=13, y=1.005)
    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    save_path = os.path.join(output_dir, "block_residual_energy_compact.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_residual_pca(stats, output_dir):
    """
    图表2: 残差 PCA 主成分 RGB 映射。

    选取代表性 blocks (0, 10, 20, 30, 39) × 4 steps。
    top-3 PCA 成分映射到 RGB 通道。
    """
    selected_blocks = [0, 10, 20, 30, 39]
    n_sel = len(selected_blocks)

    sample_key = next(iter(stats))
    T, H, W = stats[sample_key]['res_norm'].shape
    mid_frame = T // 2

    fig, axes = plt.subplots(n_sel, NUM_STEPS, figsize=(NUM_STEPS * 3.5, n_sel * 2.5))

    for row_idx, blk in enumerate(selected_blocks):
        for step in range(NUM_STEPS):
            ax = axes[row_idx, step]
            key = (step, blk)
            if key in stats:
                pca = stats[key]['pca_proj']  # [3, T, H, W]
                # 取中间帧
                rgb = pca[:, mid_frame, :, :].numpy()  # [3, H, W]

                # 对每个通道独立归一化到 [0, 1]
                for ch in range(3):
                    ch_min = rgb[ch].min()
                    ch_max = rgb[ch].max()
                    if ch_max - ch_min > 1e-8:
                        rgb[ch] = (rgb[ch] - ch_min) / (ch_max - ch_min)
                    else:
                        rgb[ch] = 0.5

                rgb = np.transpose(rgb, (1, 2, 0))  # [H, W, 3]
                ax.imshow(rgb, interpolation="bilinear", aspect="auto")
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")

            ax.set_xticks([])
            ax.set_yticks([])
            if step == 0:
                ax.set_ylabel(f"Block {blk}", fontsize=10, rotation=0, labelpad=40)
            if row_idx == 0:
                ax.set_title(f"Step {step}", fontsize=11, fontweight="bold")

    fig.suptitle(f"Residual PCA → RGB (Top-3 Components, Frame {mid_frame}/{T})\n"
                 f"Different colors = semantically distinct residual patterns",
                 fontsize=12, y=1.005)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "block_residual_pca_rgb.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_block_role_summary(stats, output_dir):
    """
    附加图表: Block 空间集中度分析。

    基于残差能量的空间分布（基尼系数），为每个 block 生成空间集中度标签。
    基尼系数: Gini = 1 − 2·L + 1/n，其中 L = ∑cumsum(sorted_x) / (n · total)
    Gini 接近 1 表示能量集中在少数像素，接近 0 表示均匀分布。
    """
    import json

    sample_key = next(iter(stats))
    T, H, W = stats[sample_key]['res_norm'].shape

    roles = {}

    for step in range(NUM_STEPS):
        for blk in range(NUM_BLOCKS):
            key = (step, blk)
            if key not in stats:
                continue

            res_norm = stats[key]['res_norm']  # [T, H, W]

            # 空间集中度分析（基尼系数）
            flat = res_norm.flatten().numpy()
            total_energy = float(flat.sum())
            sorted_flat = np.sort(flat)
            cumsum = np.cumsum(sorted_flat)
            n_pixels = len(flat)
            if total_energy > 1e-8:
                lorenz_area = cumsum.sum() / (n_pixels * total_energy)
                gini = 1 - 2 * lorenz_area + 1 / n_pixels
            else:
                gini = 0.0

            if gini > 0.6:
                spatial_focus = "highly_localized"
            elif gini > 0.35:
                spatial_focus = "moderately_localized"
            else:
                spatial_focus = "globally_distributed"

            roles[f"step{step}_block{blk}"] = {
                "spatial_focus": spatial_focus,
                "total_energy": float(total_energy),
                "gini_coefficient": float(gini),
            }

    save_path = os.path.join(output_dir, "block_role_classification.json")
    with open(save_path, "w") as f:
        json.dump(roles, f, indent=2)
    print(f"Saved: {save_path}")

    # 可视化空间集中度矩阵
    spatial_map = {"globally_distributed": 0, "moderately_localized": 1, "highly_localized": 2}

    fig, ax = plt.subplots(figsize=(8, 10))

    spatial_matrix = np.zeros((NUM_BLOCKS, NUM_STEPS))
    for step in range(NUM_STEPS):
        for blk in range(NUM_BLOCKS):
            role = roles.get(f"step{step}_block{blk}", {})
            spatial_matrix[blk, step] = spatial_map.get(role.get("spatial_focus", ""), 0)

    spatial_cmap = mcolors.ListedColormap(["#4CAF50", "#FFC107", "#F44336"])
    im = ax.imshow(spatial_matrix, cmap=spatial_cmap, aspect="auto", interpolation="nearest",
                   vmin=-0.5, vmax=2.5)
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Block Index", fontsize=11)
    ax.set_title("Spatial Focus Pattern (Gini Coefficient)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(NUM_STEPS))
    ax.set_xticklabels([f"Step {s}" for s in range(NUM_STEPS)])
    ax.set_yticks(range(0, NUM_BLOCKS, 5))
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(["Global", "Moderate", "Localized"])

    fig.suptitle("Block Spatial Concentration Analysis\n"
                 "Localized = energy concentrated in few pixels (Gini > 0.6)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "block_role_matrix.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Block Task Visualization")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing spatial_stats/ subdirectories")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for visualization images")
    parser.add_argument("--sample_id", type=str, default=None,
                        help="Specific sample ID (e.g., hdtf_0000). If not set, average all samples.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    stats = load_spatial_stats(args.input_dir, args.sample_id)

    print("\n=== Generating Residual Energy Maps ===")
    plot_residual_energy_map_compact(stats, args.output_dir)
    # 完整版 (40×4, 很大) — 可选
    # plot_residual_energy_map(stats, args.output_dir)

    print("\n=== Generating PCA RGB Maps ===")
    plot_residual_pca(stats, args.output_dir)

    print("\n=== Generating Block Role Summary ===")
    plot_block_role_summary(stats, args.output_dir)

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
