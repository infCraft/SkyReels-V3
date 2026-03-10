#!/usr/bin/env python3
"""
CKA 热力图可视化脚本

读取 cka_extract.py 产出的 CKA 矩阵和残差余弦相似度矩阵数据，生成以下图表：
1. 全局 160×160 CKA 热力图（跨 step + block）
2. 深度感知 CKA 热力图阵列（4 张 40×40，每个 step 一张）
3. 时序演进 CKA 热力图（40 张 4×4，每个 block 一张，排列为 8×5 网格）
4. 相邻 block 对角 CKA 曲线
5. 全局 160×160 残差余弦相似度热力图
6. 深度感知余弦相似度热力图阵列
7. 相邻 block 余弦相似度对角曲线
8. 相对残差幅度 40×4 热力图

用法:
    python scripts/cka_visualize.py \
        --input_dir /root/autodl-fs/experiments/cka_analysis/cka_matrices \
        --output_dir /root/autodl-fs/experiments/cka_analysis/visualizations
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

NUM_BLOCKS = 40
NUM_STEPS = 4
TOTAL = NUM_BLOCKS * NUM_STEPS  # 160


def find_high_similarity_intervals(adjacent_scores, threshold):
    """将高于阈值的相邻 block pair 聚合成连续 block 区间。"""
    intervals = []
    current_start = None
    current_pair_indices = []

    for pair_idx, score in enumerate(adjacent_scores):
        if score > threshold:
            if current_start is None:
                current_start = pair_idx
            current_pair_indices.append(pair_idx)
            continue

        if current_start is not None:
            pair_scores = [float(adjacent_scores[i]) for i in current_pair_indices]
            intervals.append({
                "start_block": int(current_start),
                "end_block": int(current_pair_indices[-1] + 1),
                "num_blocks": int(current_pair_indices[-1] - current_start + 2),
                "num_pairs": int(len(current_pair_indices)),
                "pair_indices": [int(i) for i in current_pair_indices],
                "mean_cka": float(np.mean(pair_scores)),
                "min_cka": float(np.min(pair_scores)),
                "max_cka": float(np.max(pair_scores)),
            })
            current_start = None
            current_pair_indices = []

    if current_start is not None:
        pair_scores = [float(adjacent_scores[i]) for i in current_pair_indices]
        intervals.append({
            "start_block": int(current_start),
            "end_block": int(current_pair_indices[-1] + 1),
            "num_blocks": int(current_pair_indices[-1] - current_start + 2),
            "num_pairs": int(len(current_pair_indices)),
            "pair_indices": [int(i) for i in current_pair_indices],
            "mean_cka": float(np.mean(pair_scores)),
            "min_cka": float(np.min(pair_scores)),
            "max_cka": float(np.max(pair_scores)),
        })

    return intervals


def load_cka_matrix(input_dir):
    """加载平均 CKA 矩阵。如果不存在，尝试从 per-sample 矩阵计算。"""
    avg_path = os.path.join(input_dir, "cka_average.npy")
    if os.path.exists(avg_path):
        cka = np.load(avg_path)
        print(f"Loaded average CKA matrix from {avg_path}, shape={cka.shape}")
        return cka

    # 回退：从 per-sample 矩阵计算
    sample_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_cka.npy")])
    if not sample_files:
        raise FileNotFoundError(f"No CKA matrices found in {input_dir}")

    matrices = []
    for f in sample_files:
        m = np.load(os.path.join(input_dir, f))
        matrices.append(m)
        print(f"Loaded {f}, shape={m.shape}")

    cka = np.mean(matrices, axis=0)
    print(f"Computed average from {len(matrices)} samples")
    return cka


def idx(step, block):
    """将 (step, block) 映射到全局 160 维索引。"""
    return step * NUM_BLOCKS + block


def plot_global_heatmap(cka, output_dir):
    """图表1: 全局 160×160 CKA 热力图。"""
    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(cka, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")

    # 标注 step 边界
    for s in range(1, NUM_STEPS):
        boundary = s * NUM_BLOCKS - 0.5
        ax.axhline(y=boundary, color="white", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.axvline(x=boundary, color="white", linewidth=1.5, linestyle="--", alpha=0.8)

    # step 标签
    for s in range(NUM_STEPS):
        center = s * NUM_BLOCKS + NUM_BLOCKS / 2
        ax.text(center, -3, f"Step {s}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="black")
        ax.text(-3, center, f"Step {s}", ha="right", va="center", fontsize=11,
                fontweight="bold", color="black", rotation=90)

    # 每10个block标一个tick
    tick_positions = []
    tick_labels = []
    for s in range(NUM_STEPS):
        for b in range(0, NUM_BLOCKS, 10):
            tick_positions.append(s * NUM_BLOCKS + b)
            tick_labels.append(f"{b}")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Block Index (within each Step)", fontsize=12)
    ax.set_ylabel("Block Index (within each Step)", fontsize=12)
    ax.set_title("Global CKA Similarity Matrix (4 Steps × 40 Blocks = 160)", fontsize=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CKA Similarity", fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cka_global_160x160.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_depthwise_heatmaps(cka, output_dir):
    """图表2: 深度感知 CKA 热力图阵列（4 张 40×40）。"""
    fig, axes = plt.subplots(1, NUM_STEPS, figsize=(24, 6))

    for s in range(NUM_STEPS):
        ax = axes[s]
        # 提取该 step 内部的 40×40 子矩阵
        start = s * NUM_BLOCKS
        end = start + NUM_BLOCKS
        sub = cka[start:end, start:end]

        im = ax.imshow(sub, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")

        ax.set_title(f"Step {s}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Block Index", fontsize=10)
        if s == 0:
            ax.set_ylabel("Block Index", fontsize=10)

        ticks = list(range(0, NUM_BLOCKS, 5))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=8)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("CKA Similarity", fontsize=11)

    fig.suptitle("Depth-wise CKA Matrices (Intra-Step Block Similarity)", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "cka_depthwise_40x40.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_temporal_heatmaps(cka, output_dir):
    """图表3: 时序演进 CKA 热力图（40 张 4×4，排列为 8×5 网格）。"""
    nrows, ncols = 8, 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 22))

    for b in range(NUM_BLOCKS):
        r, c = divmod(b, ncols)
        ax = axes[r, c]

        # 提取该 block 跨 4 步的 4×4 子矩阵
        sub = np.zeros((NUM_STEPS, NUM_STEPS))
        for si in range(NUM_STEPS):
            for sj in range(NUM_STEPS):
                sub[si, sj] = cka[idx(si, b), idx(sj, b)]

        im = ax.imshow(sub, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"Block {b}", fontsize=9, fontweight="bold")
        ax.set_xticks(range(NUM_STEPS))
        ax.set_xticklabels([f"S{s}" for s in range(NUM_STEPS)], fontsize=7)
        ax.set_yticks(range(NUM_STEPS))
        ax.set_yticklabels([f"S{s}" for s in range(NUM_STEPS)], fontsize=7)

    fig.suptitle("Temporal CKA: Each Block Across 4 Steps", fontsize=14, y=0.995)

    # 把 colorbar 放到整体右侧
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="CKA Similarity")

    plt.tight_layout(rect=[0, 0, 0.92, 0.98])
    save_path = os.path.join(output_dir, "cka_temporal_4x4_grid.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_diagonal_cka(cka, output_dir, redundancy_threshold=0.95):
    """图表4: 相邻 block 对角 CKA 曲线。"""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, NUM_STEPS))

    for s in range(NUM_STEPS):
        diag_cka = []
        for b in range(NUM_BLOCKS - 1):
            val = cka[idx(s, b), idx(s, b + 1)]
            diag_cka.append(val)
        ax.plot(range(NUM_BLOCKS - 1), diag_cka, marker="o", markersize=3,
                linewidth=1.5, color=colors[s], label=f"Step {s}", alpha=0.85)

    ax.set_xlabel("Block Index i → i+1", fontsize=12)
    ax.set_ylabel("CKA(Block_i, Block_{i+1})", fontsize=12)
    ax.set_title("Adjacent Block CKA (Diagonal) — High values indicate identity-like blocks", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, NUM_BLOCKS - 2)
    ax.set_ylim(0.0, 1.05)

    # 标注高相似区间
    for s in range(NUM_STEPS):
        diag_cka = [cka[idx(s, b), idx(s, b + 1)] for b in range(NUM_BLOCKS - 1)]
        intervals = find_high_similarity_intervals(diag_cka, redundancy_threshold)
        for interval in intervals:
            start = interval["start_block"]
            end = interval["end_block"] - 1
            ax.axvspan(start - 0.3, end + 0.3, alpha=0.08, color="red")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cka_diagonal_adjacent.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cross_step_diagonal(cka, output_dir):
    """附加图表: 跨步对角 CKA（同一 block 在相邻 step 间的 CKA）。"""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.Dark2(np.linspace(0, 1, NUM_STEPS - 1))

    for s in range(NUM_STEPS - 1):
        vals = []
        for b in range(NUM_BLOCKS):
            val = cka[idx(s, b), idx(s + 1, b)]
            vals.append(val)
        ax.plot(range(NUM_BLOCKS), vals, marker="s", markersize=3,
                linewidth=1.5, color=colors[s], label=f"Step {s} → {s+1}", alpha=0.85)

    ax.set_xlabel("Block Index", fontsize=12)
    ax.set_ylabel("CKA(Block_b@Step_s, Block_b@Step_{s+1})", fontsize=12)
    ax.set_title("Cross-Step CKA (Same Block, Adjacent Steps)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, NUM_BLOCKS - 1)
    ax.set_ylim(0.0, 1.05)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cka_cross_step_diagonal.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def generate_summary_stats(cka, output_dir, redundancy_threshold=0.95):
    """生成 CKA 统计摘要 JSON。"""
    import json

    summary = {
        "shape": list(cka.shape),
        "diagonal_mean": float(np.diag(cka).mean()),
        "off_diagonal_mean": float((cka.sum() - np.trace(cka)) / (TOTAL * (TOTAL - 1))),
        "adjacent_redundancy_threshold": float(redundancy_threshold),
    }

    # 每个 step 内的相邻 block CKA 统计
    for s in range(NUM_STEPS):
        diag_vals = [cka[idx(s, b), idx(s, b + 1)] for b in range(NUM_BLOCKS - 1)]
        summary[f"step{s}_adjacent_mean"] = float(np.mean(diag_vals))
        summary[f"step{s}_adjacent_min"] = float(np.min(diag_vals))
        summary[f"step{s}_adjacent_max"] = float(np.max(diag_vals))

        # 高冗余相邻 block 对 (CKA > threshold)
        # 每次存的是 block 对 (b, b+1) 的 CKA 值，方便后续分析
        high_redundancy = [(b, float(diag_vals[b])) for b in range(len(diag_vals)) if diag_vals[b] > redundancy_threshold]
        summary[f"step{s}_high_redundancy_pairs"] = high_redundancy
        summary[f"step{s}_high_redundancy_intervals"] = find_high_similarity_intervals(
            diag_vals,
            redundancy_threshold,
        )

    # 跨步相同 block 的 CKA 统计
    for s in range(NUM_STEPS - 1):
        vals = [cka[idx(s, b), idx(s + 1, b)] for b in range(NUM_BLOCKS)]
        summary[f"cross_step_{s}_{s+1}_mean"] = float(np.mean(vals))
        summary[f"cross_step_{s}_{s+1}_min"] = float(np.min(vals))

    save_path = os.path.join(output_dir, "cka_summary_stats.json")
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {save_path}")


# ============================================================
# 残差余弦相似度可视化
# ============================================================

def load_residual_cosine_matrix(input_dir):
    """加载平均残差余弦相似度矩阵。如果不存在，尝试从 per-sample 矩阵计算。"""
    avg_path = os.path.join(input_dir, "residual_cosine_average.npy")
    if os.path.exists(avg_path):
        cosine = np.load(avg_path)
        print(f"Loaded average residual cosine matrix from {avg_path}, shape={cosine.shape}")
        return cosine

    sample_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_residual_cosine.npy")])
    if not sample_files:
        return None

    matrices = [np.load(os.path.join(input_dir, f)) for f in sample_files]
    cosine = np.mean(matrices, axis=0)
    print(f"Computed average residual cosine from {len(matrices)} samples")
    return cosine


def load_relative_magnitude(input_dir):
    """加载平均相对残差幅度矩阵 [NUM_BLOCKS, NUM_STEPS]。"""
    avg_path = os.path.join(input_dir, "relative_magnitude_average.npy")
    if os.path.exists(avg_path):
        rel_mag = np.load(avg_path)
        print(f"Loaded average relative magnitude from {avg_path}, shape={rel_mag.shape}")
        return rel_mag

    sample_files = sorted([f for f in os.listdir(input_dir) if f.endswith("_relative_magnitude.npy")])
    if not sample_files:
        return None

    matrices = [np.load(os.path.join(input_dir, f)) for f in sample_files]
    rel_mag = np.mean(matrices, axis=0)
    print(f"Computed average relative magnitude from {len(matrices)} samples")
    return rel_mag


def plot_cosine_global_heatmap(cosine, output_dir):
    """
    全局 160×160 残差余弦相似度热力图。

    CosSim(i, j) = (r̄_i · r̄_j) / (‖r̄_i‖ · ‖r̄_j‖)
    其中 r̄_i 是 block i 的残差沿 token 维度的均值。
    值域 [-1, 1]，用发散色图（RdBu_r）中心在 0。
    """
    fig, ax = plt.subplots(figsize=(16, 14))

    im = ax.imshow(cosine, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")

    for s in range(1, NUM_STEPS):
        boundary = s * NUM_BLOCKS - 0.5
        ax.axhline(y=boundary, color="black", linewidth=1.5, linestyle="--", alpha=0.8)
        ax.axvline(x=boundary, color="black", linewidth=1.5, linestyle="--", alpha=0.8)

    for s in range(NUM_STEPS):
        center = s * NUM_BLOCKS + NUM_BLOCKS / 2
        ax.text(center, -3, f"Step {s}", ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="black")
        ax.text(-3, center, f"Step {s}", ha="right", va="center", fontsize=11,
                fontweight="bold", color="black", rotation=90)

    tick_positions = []
    tick_labels = []
    for s in range(NUM_STEPS):
        for b in range(0, NUM_BLOCKS, 10):
            tick_positions.append(s * NUM_BLOCKS + b)
            tick_labels.append(f"{b}")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)

    ax.set_xlabel("Block Index (within each Step)", fontsize=12)
    ax.set_ylabel("Block Index (within each Step)", fontsize=12)
    ax.set_title("Residual Cosine Similarity Matrix (4 Steps × 40 Blocks = 160)\n"
                 "CosSim(i,j) = r̄_i·r̄_j / (‖r̄_i‖·‖r̄_j‖), r̄ = mean-pooled residual",
                 fontsize=13)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine Similarity", fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cosine_global_160x160.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cosine_depthwise_heatmaps(cosine, output_dir):
    """深度感知余弦相似度热力图阵列（4 张 40×40）。"""
    fig, axes = plt.subplots(1, NUM_STEPS, figsize=(24, 6))

    for s in range(NUM_STEPS):
        ax = axes[s]
        start = s * NUM_BLOCKS
        end = start + NUM_BLOCKS
        sub = cosine[start:end, start:end]

        im = ax.imshow(sub, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")

        ax.set_title(f"Step {s}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Block Index", fontsize=10)
        if s == 0:
            ax.set_ylabel("Block Index", fontsize=10)

        ticks = list(range(0, NUM_BLOCKS, 5))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=8)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=11)

    fig.suptitle("Depth-wise Residual Cosine Similarity (Intra-Step)", fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "cosine_depthwise_40x40.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_cosine_diagonal(cosine, output_dir):
    """
    相邻 block 残差余弦相似度对角曲线。

    CosSim(Block_i, Block_{i+1})：高值表示相邻 block 在"同方向"修改表征。
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.Set1(np.linspace(0, 1, NUM_STEPS))

    for s in range(NUM_STEPS):
        diag_cos = []
        for b in range(NUM_BLOCKS - 1):
            val = cosine[idx(s, b), idx(s, b + 1)]
            diag_cos.append(val)
        ax.plot(range(NUM_BLOCKS - 1), diag_cos, marker="o", markersize=3,
                linewidth=1.5, color=colors[s], label=f"Step {s}", alpha=0.85)

    ax.set_xlabel("Block Index i → i+1", fontsize=12)
    ax.set_ylabel("CosSim(Block_i, Block_{i+1})", fontsize=12)
    ax.set_title("Adjacent Block Residual Cosine Similarity\n"
                 "High values = blocks modify representation in the same direction",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, NUM_BLOCKS - 2)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "cosine_diagonal_adjacent.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_relative_magnitude_heatmap(rel_mag, output_dir):
    """
    相对残差幅度 40×4 热力图。

    RelMag_{s,b} = mean_k(‖r_k‖₂) / mean_k(‖x_in_k‖₂)
    其中 r_k = x_out_k − x_in_k，k 遍历所有有效空间 token。

    值越小表示该 block 对表征的改变越小（跟跳候选）。

    Args:
        rel_mag: [NUM_BLOCKS, NUM_STEPS] numpy array
    """
    fig, ax = plt.subplots(figsize=(8, 12))

    im = ax.imshow(rel_mag, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Block Index", fontsize=12)
    ax.set_title("Relative Residual Magnitude per Block per Step\n"
                 r"RelMag = mean($\|r_k\|$) / mean($\|x_{in,k}\|$)",
                 fontsize=13)
    ax.set_xticks(range(NUM_STEPS))
    ax.set_xticklabels([f"Step {s}" for s in range(NUM_STEPS)], fontsize=10)
    ax.set_yticks(range(0, NUM_BLOCKS, 5))
    ax.set_yticklabels(range(0, NUM_BLOCKS, 5), fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Magnitude", fontsize=11)

    # 标注数值（如果 block 数不太多的话）
    for blk in range(NUM_BLOCKS):
        for step in range(NUM_STEPS):
            val = rel_mag[blk, step]
            color = "white" if val > (rel_mag.max() * 0.6) else "black"
            ax.text(step, blk, f"{val:.3f}", ha="center", va="center",
                    fontsize=5, color=color)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "relative_magnitude_heatmap.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="CKA Visualization")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing CKA matrix .npy files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for visualization images")
    parser.add_argument("--redundancy_threshold", type=float, default=0.95,
                        help="Threshold for marking adjacent blocks as highly redundant")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cka = load_cka_matrix(args.input_dir)
    assert cka.shape == (TOTAL, TOTAL), f"Expected ({TOTAL},{TOTAL}), got {cka.shape}"

    # 合理性检查
    diag_vals = np.diag(cka)
    print(f"Diagonal values: min={diag_vals.min():.4f}, max={diag_vals.max():.4f}, mean={diag_vals.mean():.4f}")
    print(f"Off-diagonal: min={cka.min():.4f}, max={cka.max():.4f}")

    plot_global_heatmap(cka, args.output_dir)
    plot_depthwise_heatmaps(cka, args.output_dir)
    plot_temporal_heatmaps(cka, args.output_dir)
    plot_diagonal_cka(cka, args.output_dir, redundancy_threshold=args.redundancy_threshold)
    plot_cross_step_diagonal(cka, args.output_dir)
    generate_summary_stats(cka, args.output_dir, redundancy_threshold=args.redundancy_threshold)

    # 残差余弦相似度可视化
    cosine = load_residual_cosine_matrix(args.input_dir)
    if cosine is not None:
        assert cosine.shape == (TOTAL, TOTAL), f"Cosine matrix shape mismatch: {cosine.shape}"
        print(f"\nCosine diagonal: min={np.diag(cosine).min():.4f}, "
              f"max={np.diag(cosine).max():.4f}, mean={np.diag(cosine).mean():.4f}")
        plot_cosine_global_heatmap(cosine, args.output_dir)
        plot_cosine_depthwise_heatmaps(cosine, args.output_dir)
        plot_cosine_diagonal(cosine, args.output_dir)
    else:
        print("\nNo residual cosine matrix found, skipping cosine plots.")

    # 相对残差幅度可视化
    rel_mag = load_relative_magnitude(args.input_dir)
    if rel_mag is not None:
        print(f"\nRelative magnitude: min={rel_mag.min():.4f}, "
              f"max={rel_mag.max():.4f}, mean={rel_mag.mean():.4f}")
        plot_relative_magnitude_heatmap(rel_mag, args.output_dir)
    else:
        print("\nNo relative magnitude data found, skipping magnitude plot.")

    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
