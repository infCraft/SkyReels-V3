#!/usr/bin/env python3
"""
Block 任务可视化脚本

读取 cka_extract.py 产出的空间统计数据，生成以下可视化：
1. 残差 L2 范数空间热力图 (Residual Energy Map)
2. 残差 PCA 主成分 RGB 映射 (Residual PCA)
3. 残差频域分析 (Residual FFT) — 径向频率曲线 + 频段能量比例热力图

用法:
    python scripts/block_task_visualize.py \
        --input_dir /root/autodl-fs/experiments/cka_analysis/spatial_stats \
        --output_dir /root/autodl-fs/experiments/cka_analysis/visualizations \
        --sample_id hdtf_0000
    或者不指定 --sample_id 则对所有样本取平均
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
                        'fft_mag': data['fft_mag'].float(),
                    }
                else:
                    sum_stats[key]['res_norm'] += data['res_norm'].float()
                    sum_stats[key]['pca_proj'] += data['pca_proj'].float()
                    sum_stats[key]['fft_mag'] += data['fft_mag'].float()
        count += 1

    if count > 1:
        for key in sum_stats:
            for field in sum_stats[key]:
                sum_stats[key][field] /= count

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


def compute_radial_profile(fft_mag_2d):
    """
    将 2D FFT 幅度谱转换为径向频率分布。

    Args:
        fft_mag_2d: [H, W//2+1] FFT 幅度

    Returns:
        freqs: 1D 频率轴
        profile: 每个频率 bin 的平均幅度
    """
    H, W_half = fft_mag_2d.shape
    # 构建频率网格
    fy = np.fft.fftfreq(H)  # [-0.5, 0.5)
    fx = np.fft.rfftfreq(W_half * 2 - 1)  # [0, 0.5]

    fy_grid, fx_grid = np.meshgrid(fy, fx, indexing="ij")
    r = np.sqrt(fy_grid ** 2 + fx_grid ** 2)

    # 将径向距离分 bin
    max_r = r.max()
    n_bins = min(H // 2, 30)
    bin_edges = np.linspace(0, max_r, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.sum() > 0:
            profile[i] = fft_mag_2d[mask].mean()

    return bin_centers, profile


def plot_fft_radial_curves(stats, output_dir):
    """
    图表3a: 径向频率能量分布曲线。

    为每个 step 画一张图，40 条曲线对应 40 个 block，颜色渐变区分。
    """
    fig, axes = plt.subplots(1, NUM_STEPS, figsize=(24, 6))
    cmap = plt.cm.viridis

    for step in range(NUM_STEPS):
        ax = axes[step]
        for blk in range(NUM_BLOCKS):
            key = (step, blk)
            if key not in stats:
                continue
            fft_mag = stats[key]['fft_mag'].numpy()
            freqs, profile = compute_radial_profile(fft_mag)
            color = cmap(blk / (NUM_BLOCKS - 1))
            ax.plot(freqs, profile, color=color, alpha=0.7, linewidth=0.8)

        ax.set_xlabel("Spatial Frequency", fontsize=10)
        ax.set_ylabel("Mean Amplitude", fontsize=10)
        ax.set_title(f"Step {step}", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # 添加一个 colorbar 表示 block 索引
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=NUM_BLOCKS - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
    cbar.set_label("Block Index", fontsize=11)

    fig.suptitle("Radial FFT Profile of Residual Energy\n"
                 "Low freq = structure/layout, High freq = texture/detail",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "block_fft_radial_curves.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_fft_band_heatmap(stats, output_dir):
    """
    图表3b: low/mid/high 频段能量比例热力图。

    40(blocks) × 4(steps) 的 3 通道堆叠图。
    """
    # 计算每个 (step, block) 的三个频段能量比例
    band_ratios = np.zeros((NUM_BLOCKS, NUM_STEPS, 3))  # [blocks, steps, (low/mid/high)]

    for step in range(NUM_STEPS):
        for blk in range(NUM_BLOCKS):
            key = (step, blk)
            if key not in stats:
                continue
            fft_mag = stats[key]['fft_mag'].numpy()
            freqs, profile = compute_radial_profile(fft_mag)

            # 将频率分为三个频段
            n = len(freqs)
            low_end = n // 3
            mid_end = 2 * n // 3

            energy_low = profile[:low_end].sum()
            energy_mid = profile[low_end:mid_end].sum()
            energy_high = profile[mid_end:].sum()
            total = energy_low + energy_mid + energy_high + 1e-12

            band_ratios[blk, step, 0] = energy_low / total
            band_ratios[blk, step, 1] = energy_mid / total
            band_ratios[blk, step, 2] = energy_high / total

    band_names = ["Low Freq\n(Structure)", "Mid Freq\n(Contour)", "High Freq\n(Detail)"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    for band_idx in range(3):
        ax = axes[band_idx]
        data = band_ratios[:, :, band_idx]  # [40, 4]
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", interpolation="nearest",
                        vmin=0, vmax=1)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("Block Index", fontsize=11)
        ax.set_title(band_names[band_idx], fontsize=12, fontweight="bold")
        ax.set_xticks(range(NUM_STEPS))
        ax.set_xticklabels([f"Step {s}" for s in range(NUM_STEPS)], fontsize=9)
        ax.set_yticks(range(0, NUM_BLOCKS, 5))
        ax.set_yticklabels(range(0, NUM_BLOCKS, 5), fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Frequency Band Energy Ratios per Block per Step\n"
                 "Each cell shows the proportion of energy in that frequency band",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "block_fft_band_heatmap.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_block_role_summary(stats, output_dir):
    """
    附加图表: Block 角色分类汇总。

    基于残差能量的空间分布和频率特征，为每个 block 生成一个角色标签。
    """
    import json

    sample_key = next(iter(stats))
    T, H, W = stats[sample_key]['res_norm'].shape

    roles = {}  # {(step, block): {"dominant_freq": str, "spatial_focus": str, "energy": float}}

    for step in range(NUM_STEPS):
        for blk in range(NUM_BLOCKS):
            key = (step, blk)
            if key not in stats:
                continue

            res_norm = stats[key]['res_norm']  # [T, H, W]
            fft_mag = stats[key]['fft_mag'].numpy()

            # 频域分析
            freqs, profile = compute_radial_profile(fft_mag)
            n = len(freqs)
            low_end = n // 3
            mid_end = 2 * n // 3
            energy_low = profile[:low_end].sum()
            energy_mid = profile[low_end:mid_end].sum()
            energy_high = profile[mid_end:].sum()
            total_freq = energy_low + energy_mid + energy_high + 1e-12

            if energy_low / total_freq > 0.55:
                dom_freq = "low_freq_structure"
            elif energy_high / total_freq > 0.35:
                dom_freq = "high_freq_detail"
            else:
                dom_freq = "mid_freq_contour"

            # 空间集中度分析（使用基尼系数的简化版本）
            flat = res_norm.flatten().numpy()
            total_energy = float(flat.sum())
            sorted_flat = np.sort(flat)
            cumsum = np.cumsum(sorted_flat)
            n_pixels = len(flat)
            if total_energy > 1e-8:
                # 计算洛伦兹曲线下面积
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
                "dominant_frequency": dom_freq,
                "spatial_focus": spatial_focus,
                "total_energy": float(total_energy),
                "gini_coefficient": float(gini),
                "freq_low_ratio": float(energy_low / total_freq),
                "freq_mid_ratio": float(energy_mid / total_freq),
                "freq_high_ratio": float(energy_high / total_freq),
            }

    save_path = os.path.join(output_dir, "block_role_classification.json")
    with open(save_path, "w") as f:
        json.dump(roles, f, indent=2)
    print(f"Saved: {save_path}")

    # 可视化角色矩阵
    freq_map = {"low_freq_structure": 0, "mid_freq_contour": 1, "high_freq_detail": 2}
    spatial_map = {"globally_distributed": 0, "moderately_localized": 1, "highly_localized": 2}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

    # 频率角色矩阵
    freq_matrix = np.zeros((NUM_BLOCKS, NUM_STEPS))
    for step in range(NUM_STEPS):
        for blk in range(NUM_BLOCKS):
            role = roles.get(f"step{step}_block{blk}", {})
            freq_matrix[blk, step] = freq_map.get(role.get("dominant_frequency", ""), 1)

    freq_cmap = mcolors.ListedColormap(["#2196F3", "#FF9800", "#E91E63"])
    im1 = ax1.imshow(freq_matrix, cmap=freq_cmap, aspect="auto", interpolation="nearest",
                     vmin=-0.5, vmax=2.5)
    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_ylabel("Block Index", fontsize=11)
    ax1.set_title("Dominant Frequency Band", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(NUM_STEPS))
    ax1.set_yticks(range(0, NUM_BLOCKS, 5))
    cbar1 = fig.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
    cbar1.set_ticklabels(["Low\n(Structure)", "Mid\n(Contour)", "High\n(Detail)"])

    # 空间集中度矩阵
    spatial_matrix = np.zeros((NUM_BLOCKS, NUM_STEPS))
    for step in range(NUM_STEPS):
        for blk in range(NUM_BLOCKS):
            role = roles.get(f"step{step}_block{blk}", {})
            spatial_matrix[blk, step] = spatial_map.get(role.get("spatial_focus", ""), 0)

    spatial_cmap = mcolors.ListedColormap(["#4CAF50", "#FFC107", "#F44336"])
    im2 = ax2.imshow(spatial_matrix, cmap=spatial_cmap, aspect="auto", interpolation="nearest",
                     vmin=-0.5, vmax=2.5)
    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("Block Index", fontsize=11)
    ax2.set_title("Spatial Focus Pattern", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(NUM_STEPS))
    ax2.set_yticks(range(0, NUM_BLOCKS, 5))
    cbar2 = fig.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
    cbar2.set_ticklabels(["Global", "Moderate", "Localized"])

    fig.suptitle("Block Role Classification\n"
                 "Left: What frequency the block operates on | Right: Where it focuses spatially",
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

    print("\n=== Generating FFT Radial Curves ===")
    plot_fft_radial_curves(stats, args.output_dir)

    print("\n=== Generating FFT Band Heatmap ===")
    plot_fft_band_heatmap(stats, args.output_dir)

    print("\n=== Generating Block Role Summary ===")
    plot_block_role_summary(stats, args.output_dir)

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
