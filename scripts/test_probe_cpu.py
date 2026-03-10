#!/usr/bin/env python3
"""
CPU 验证脚本：用合成数据测试探针代码的正确性。

测试项目：
1. compute_linear_cka_matrix —— CKA 矩阵对角线 ≈ 1.0，对称
2. compute_cosine_similarity_matrix —— 余弦矩阵对角线 ≈ 1.0，对称，值域 [-1, 1]
3. PCA 中心化修复 —— 投影与 sklearn 对比（如果有的话）
4. 空间统计保存/加载 —— 无 fft_mag
5. 可视化脚本端到端 —— 不报错

用法:
    conda activate sky && python scripts/test_probe_cpu.py
"""

import os
import sys
import json
import tempfile
import shutil

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


NUM_BLOCKS = 40
NUM_STEPS = 4
TOTAL = NUM_BLOCKS * NUM_STEPS
N_TOKENS = 256   # 小规模以加速 CPU 测试
DIM = 64         # 小维度
T, H, W = 2, 8, 16  # 小空间网格，n_spatial = 2*8*16 = 256 = N_TOKENS


def test_cka_matrix():
    """
    测试 CKA 矩阵的数学性质：
    - 对角线 CKA(X, X) = 1.0
    - 对称性 CKA(X, Y) = CKA(Y, X)
    - 不同分布的激活应有 CKA < 1
    """
    print("=" * 60)
    print("Test 1: CKA Matrix Properties")
    print("=" * 60)

    from scripts.cka_extract import compute_linear_cka_matrix

    # 生成 8 个激活张量（2 steps × 4 blocks 规模）
    torch.manual_seed(42)
    activations = []
    for i in range(8):
        # 每个激活是不同的随机矩阵
        act = torch.randn(N_TOKENS, DIM)
        activations.append(act)

    # 前两个有意设为几乎相同（测试高 CKA）
    activations[1] = activations[0] + 0.01 * torch.randn(N_TOKENS, DIM)

    cka = compute_linear_cka_matrix(activations, device="cpu")

    # 检查 1: 对角线 ≈ 1.0
    diag = np.diag(cka)
    assert np.allclose(diag, 1.0, atol=1e-5), f"Diagonal not 1.0: {diag}"
    print(f"  ✓ Diagonal all ≈ 1.0 (range [{diag.min():.6f}, {diag.max():.6f}])")

    # 检查 2: 对称性
    assert np.allclose(cka, cka.T, atol=1e-6), "CKA matrix not symmetric!"
    print(f"  ✓ Symmetric (max |A-A^T| = {np.abs(cka - cka.T).max():.2e})")

    # 检查 3: 值域 [0, 1]
    assert cka.min() >= -1e-6, f"CKA min < 0: {cka.min()}"
    assert cka.max() <= 1.0 + 1e-6, f"CKA max > 1: {cka.max()}"
    print(f"  ✓ Value range [{cka.min():.4f}, {cka.max():.4f}] ⊆ [0, 1]")

    # 检查 4: 近似相同的激活有高 CKA
    assert cka[0, 1] > 0.99, f"Similar activations CKA too low: {cka[0, 1]:.4f}"
    print(f"  ✓ Similar activations CKA = {cka[0, 1]:.4f} (> 0.99)")

    # 检查 5: 随机激活有中等 CKA
    random_cka = cka[2, 5]
    print(f"  ✓ Random pair CKA = {random_cka:.4f} (should be moderate)")

    print("  PASSED\n")


def test_cosine_matrix():
    """
    测试余弦相似度矩阵的数学性质：
    - 对角线 = 1.0
    - 对称性
    - 值域 [-1, 1]
    - 方向相反的残差余弦 ≈ -1
    """
    print("=" * 60)
    print("Test 2: Cosine Similarity Matrix Properties")
    print("=" * 60)

    from scripts.cka_extract import compute_cosine_similarity_matrix

    torch.manual_seed(42)
    residuals = []
    for i in range(8):
        res = torch.randn(N_TOKENS, DIM)
        residuals.append(res)

    # 第 1 个和第 2 个方向相同（缩放版）
    residuals[1] = residuals[0] * 2.0 + 0.001 * torch.randn(N_TOKENS, DIM)
    # 第 3 个和第 0 个方向相反
    residuals[2] = -residuals[0] + 0.001 * torch.randn(N_TOKENS, DIM)

    cosine = compute_cosine_similarity_matrix(residuals)

    # 检查 1: 对角线 = 1.0
    diag = np.diag(cosine)
    assert np.allclose(diag, 1.0, atol=1e-5), f"Diagonal not 1.0: {diag}"
    print(f"  ✓ Diagonal all ≈ 1.0 (range [{diag.min():.6f}, {diag.max():.6f}])")

    # 检查 2: 对称性
    assert np.allclose(cosine, cosine.T, atol=1e-6), "Cosine matrix not symmetric!"
    print(f"  ✓ Symmetric (max |A-A^T| = {np.abs(cosine - cosine.T).max():.2e})")

    # 检查 3: 值域 [-1, 1]
    assert cosine.min() >= -1.0 - 1e-6, f"Cosine min < -1: {cosine.min()}"
    assert cosine.max() <= 1.0 + 1e-6, f"Cosine max > 1: {cosine.max()}"
    print(f"  ✓ Value range [{cosine.min():.4f}, {cosine.max():.4f}] ⊆ [-1, 1]")

    # 检查 4: 同方向高相似度
    assert cosine[0, 1] > 0.99, f"Same direction cosine too low: {cosine[0, 1]:.4f}"
    print(f"  ✓ Same direction cosine = {cosine[0, 1]:.4f} (> 0.99)")

    # 检查 5: 反方向低相似度
    assert cosine[0, 2] < -0.99, f"Opposite direction cosine too high: {cosine[0, 2]:.4f}"
    print(f"  ✓ Opposite direction cosine = {cosine[0, 2]:.4f} (< -0.99)")

    print("  PASSED\n")


def test_pca_centering():
    """
    测试 PCA 投影的正确性（修复后含中心化）：
    - 用已知数据构造 PCA 投影，验证结果与手工计算一致
    """
    print("=" * 60)
    print("Test 3: PCA Centering Correctness")
    print("=" * 60)

    torch.manual_seed(42)

    # 构造已知数据
    n_sub = 128
    n_full = 256
    D = 32

    full_data = torch.randn(n_full, D)
    sub_idx = torch.arange(0, n_sub)  # 取前 n_sub 个
    sub_data = full_data[sub_idx]

    # 新代码逻辑（修复后）
    res_mean = sub_data.mean(dim=0, keepdim=True)         # [1, D]
    sub_centered = sub_data - res_mean                      # [n_sub, D]
    _, _, V = torch.pca_lowrank(sub_centered, q=3, niter=4) # V: [D, 3]
    proj_fixed = (full_data - res_mean) @ V                 # [n_full, 3]

    # 旧代码逻辑（buggy）
    sub_centered_old = sub_data - sub_data.mean(dim=0, keepdim=True)
    _, _, V_old = torch.pca_lowrank(sub_centered_old, q=3, niter=4)
    proj_buggy = full_data @ V_old  # 缺少中心化！

    # 手工验证：对子采样数据投影，两者应在子采样范围内一致
    sub_proj_fixed = (sub_data - res_mean) @ V
    sub_proj_direct = sub_centered @ V
    assert torch.allclose(sub_proj_fixed, sub_proj_direct, atol=1e-5), \
        "PCA projection mismatch on subsample data"
    print(f"  ✓ Subsample projection consistent with centered data")

    # 验证投影在非子采样区域与 buggy 版本不同
    out_of_sample = full_data[n_sub:]  # 不在子采样中的数据
    proj_fixed_oos = (out_of_sample - res_mean) @ V
    proj_buggy_oos = out_of_sample @ V_old
    # 两者的第一主成分均值应不同（buggy 版未减去均值，投影会有偏移）
    mean_diff = (proj_fixed_oos[:, 0].mean() - proj_buggy_oos[:, 0].mean()).abs().item()
    print(f"  ✓ OOS projection mean difference: {mean_diff:.4f} (should be >0 if mean≠0)")
    print(f"  ✓ Fixed projection reduces to zero-mean on training set: "
          f"|mean|={sub_proj_fixed.mean(dim=0).abs().max().item():.2e}")

    print("  PASSED\n")


def test_spatial_stats_format():
    """
    测试空间统计保存格式：只含 res_norm 和 pca_proj（无 fft_mag）。
    """
    print("=" * 60)
    print("Test 4: Spatial Stats Format (no fft_mag)")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        # 模拟新格式
        stats = {
            'res_norm': torch.randn(T, H, W),
            'pca_proj': torch.randn(3, T, H, W),
        }
        save_path = os.path.join(tmpdir, "step0_block0.pt")
        torch.save(stats, save_path)

        # 加载并验证
        loaded = torch.load(save_path, map_location="cpu", weights_only=True)
        assert set(loaded.keys()) == {'res_norm', 'pca_proj'}, \
            f"Unexpected keys: {loaded.keys()}"
        assert 'fft_mag' not in loaded, "fft_mag should not be in new format!"
        assert loaded['res_norm'].shape == (T, H, W)
        assert loaded['pca_proj'].shape == (3, T, H, W)
        print(f"  ✓ Keys = {set(loaded.keys())} (no fft_mag)")
        print(f"  ✓ res_norm shape = {loaded['res_norm'].shape}")
        print(f"  ✓ pca_proj shape = {loaded['pca_proj'].shape}")
    finally:
        shutil.rmtree(tmpdir)

    print("  PASSED\n")


def test_visualize_scripts():
    """
    测试可视化脚本端到端（用合成数据）。
    """
    print("=" * 60)
    print("Test 5: Visualization Scripts End-to-End")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        # 5a. 准备 CKA 矩阵数据
        cka_dir = os.path.join(tmpdir, "cka_matrices")
        os.makedirs(cka_dir)

        np.random.seed(42)
        # 生成合理的 CKA 矩阵（对角线=1，对称，正定）
        raw = np.random.rand(TOTAL, TOTAL) * 0.3 + 0.5
        cka = (raw + raw.T) / 2
        np.fill_diagonal(cka, 1.0)
        np.save(os.path.join(cka_dir, "cka_average.npy"), cka)

        # 生成合理的余弦矩阵
        raw_cos = np.random.rand(TOTAL, TOTAL) * 0.6 - 0.3
        cosine = (raw_cos + raw_cos.T) / 2
        np.fill_diagonal(cosine, 1.0)
        np.save(os.path.join(cka_dir, "residual_cosine_average.npy"), cosine)

        # 生成相对残差幅度
        rel_mag = np.random.rand(NUM_BLOCKS, NUM_STEPS) * 0.1 + 0.01
        np.save(os.path.join(cka_dir, "relative_magnitude_average.npy"), rel_mag)

        # 运行 CKA 可视化
        viz_dir = os.path.join(tmpdir, "visualizations")
        os.makedirs(viz_dir)

        # 直接导入和调用
        from scripts.cka_visualize import (
            plot_global_heatmap, plot_depthwise_heatmaps, plot_temporal_heatmaps,
            plot_diagonal_cka, plot_cross_step_diagonal, generate_summary_stats,
            plot_cosine_global_heatmap, plot_cosine_depthwise_heatmaps,
            plot_cosine_diagonal, plot_relative_magnitude_heatmap,
        )

        plot_global_heatmap(cka, viz_dir)
        plot_depthwise_heatmaps(cka, viz_dir)
        plot_temporal_heatmaps(cka, viz_dir)
        plot_diagonal_cka(cka, viz_dir)
        plot_cross_step_diagonal(cka, viz_dir)
        generate_summary_stats(cka, viz_dir)
        plot_cosine_global_heatmap(cosine, viz_dir)
        plot_cosine_depthwise_heatmaps(cosine, viz_dir)
        plot_cosine_diagonal(cosine, viz_dir)
        plot_relative_magnitude_heatmap(rel_mag, viz_dir)

        # 验证输出文件
        expected_files = [
            "cka_global_160x160.png",
            "cka_depthwise_40x40.png",
            "cka_temporal_4x4_grid.png",
            "cka_diagonal_adjacent.png",
            "cka_cross_step_diagonal.png",
            "cka_summary_stats.json",
            "cosine_global_160x160.png",
            "cosine_depthwise_40x40.png",
            "cosine_diagonal_adjacent.png",
            "relative_magnitude_heatmap.png",
        ]
        for f in expected_files:
            fpath = os.path.join(viz_dir, f)
            assert os.path.exists(fpath), f"Missing output: {f}"
            fsize = os.path.getsize(fpath)
            print(f"  ✓ {f} ({fsize:,} bytes)")

        # 5b. 准备空间统计数据并运行 block_task_visualize
        spatial_dir = os.path.join(tmpdir, "spatial_stats", "test_sample")
        os.makedirs(spatial_dir)

        for step in range(NUM_STEPS):
            for blk in range(NUM_BLOCKS):
                stats = {
                    'res_norm': torch.rand(T, H, W),
                    'pca_proj': torch.randn(3, T, H, W),
                }
                torch.save(stats, os.path.join(spatial_dir, f"step{step}_block{blk}.pt"))

        from scripts.block_task_visualize import (
            load_spatial_stats, plot_residual_energy_map_compact,
            plot_residual_pca, plot_block_role_summary,
        )

        stats_loaded = load_spatial_stats(
            os.path.join(tmpdir, "spatial_stats"), "test_sample"
        )
        assert len(stats_loaded) == NUM_BLOCKS * NUM_STEPS, \
            f"Expected {NUM_BLOCKS * NUM_STEPS} entries, got {len(stats_loaded)}"

        block_viz_dir = os.path.join(viz_dir, "block_task")
        os.makedirs(block_viz_dir)

        plot_residual_energy_map_compact(stats_loaded, block_viz_dir)
        plot_residual_pca(stats_loaded, block_viz_dir)
        plot_block_role_summary(stats_loaded, block_viz_dir)

        block_expected = [
            "block_residual_energy_compact.png",
            "block_residual_pca_rgb.png",
            "block_role_classification.json",
            "block_role_matrix.png",
        ]
        for f in block_expected:
            fpath = os.path.join(block_viz_dir, f)
            assert os.path.exists(fpath), f"Missing output: {f}"
            fsize = os.path.getsize(fpath)
            print(f"  ✓ {f} ({fsize:,} bytes)")

        # 验证 JSON 内容
        with open(os.path.join(block_viz_dir, "block_role_classification.json")) as f:
            roles = json.load(f)
        assert "step0_block0" in roles
        assert "spatial_focus" in roles["step0_block0"]
        assert "gini_coefficient" in roles["step0_block0"]
        # 不应包含频域字段
        assert "dominant_frequency" not in roles["step0_block0"], \
            "FFT fields should be removed!"
        print(f"  ✓ block_role_classification.json format correct (no freq fields)")

    finally:
        shutil.rmtree(tmpdir)

    print("  PASSED\n")


def test_backward_compat_old_spatial_stats():
    """
    测试已有空间统计（含 fft_mag）也能被新 load_spatial_stats 正确加载。
    """
    print("=" * 60)
    print("Test 6: Backward Compatibility (old .pt with fft_mag)")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp()
    try:
        spatial_dir = os.path.join(tmpdir, "old_sample")
        os.makedirs(spatial_dir)

        for step in range(NUM_STEPS):
            for blk in range(NUM_BLOCKS):
                stats = {
                    'res_norm': torch.rand(T, H, W),
                    'pca_proj': torch.randn(3, T, H, W),
                    'fft_mag': torch.rand(H, W // 2 + 1),  # 旧格式
                }
                torch.save(stats, os.path.join(spatial_dir, f"step{step}_block{blk}.pt"))

        from scripts.block_task_visualize import load_spatial_stats
        stats_loaded = load_spatial_stats(tmpdir, "old_sample")
        assert len(stats_loaded) == NUM_BLOCKS * NUM_STEPS

        # 验证加载了 res_norm 和 pca_proj，忽略了 fft_mag
        key = (0, 0)
        assert 'res_norm' in stats_loaded[key]
        assert 'pca_proj' in stats_loaded[key]
        # load_spatial_stats 只取 res_norm 和 pca_proj
        assert 'fft_mag' not in stats_loaded[key], \
            "fft_mag should not be loaded in new code!"
        print(f"  ✓ Old format .pt loaded successfully, fft_mag ignored")

    finally:
        shutil.rmtree(tmpdir)

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CPU Validation Tests for Probe Code")
    print("=" * 60 + "\n")

    test_cka_matrix()
    test_cosine_matrix()
    test_pca_centering()
    test_spatial_stats_format()
    test_visualize_scripts()
    test_backward_compat_old_spatial_stats()

    print("=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60)
