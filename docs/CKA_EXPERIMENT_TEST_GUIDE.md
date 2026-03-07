# CKA 冗余分析实验 — 测试指南

## 实验概览

本实验对 SkyReels-V3 Talking Avatar (19B) 模型进行 CKA 冗余分析：
- **模型**: 40 个 WanAttentionBlock，dim=5120
- **去噪**: 4 步 flow matching
- **总激活数**: 4 steps × 40 blocks = 160 个中间激活
- **校准集**: 10 个 HDTF 样本（720P, 5秒音频）
- **子采样**: 每次提取 8192 个 token（从 ~56700 个有效 token 中随机选取）
- **CKA**: 线性 CKA (HSIC-based)，逐样本计算后取平均

## 目录结构

```
scripts/
├── cka_extract.py              # Phase 1: 激活提取 + CKA 计算（需要 GPU）
├── run_cka_extraction.sh       # Phase 1: 启动脚本
├── cka_visualize.py            # Phase 2: CKA 热力图可视化（纯 CPU）
└── block_task_visualize.py     # Phase 3: Block 任务可视化（纯 CPU）
```

产出目录结构：
```
/root/autodl-fs/experiments/cka_analysis/
├── cka_matrices/
│   ├── hdtf_0000_cka.npy      # 单样本 CKA 矩阵 [160, 160]
│   ├── hdtf_0001_cka.npy
│   ├── ...
│   ├── cka_average.npy         # 跨样本平均 CKA 矩阵
│   └── cka_summary.pt          # 包含元数据的 PyTorch 保存文件
├── spatial_stats/
│   ├── hdtf_0000/
│   │   ├── step0_block0.pt     # 包含 res_norm, pca_proj, fft_mag
│   │   ├── step0_block1.pt
│   │   ├── ...                 # 每样本 160 个文件
│   │   └── step3_block39.pt
│   ├── hdtf_0001/
│   └── ...
└── visualizations/
    ├── cka_global_160x160.png
    ├── cka_depthwise_40x40.png
    ├── cka_temporal_4x4_grid.png
    ├── cka_diagonal_adjacent.png
    ├── cka_cross_step_diagonal.png
    ├── cka_summary_stats.json
    ├── block_residual_energy_compact.png
    ├── block_residual_pca_rgb.png
    ├── block_fft_radial_curves.png
    ├── block_fft_band_heatmap.png
    ├── block_role_classification.json
    └── block_role_matrix.png
```

---

## Phase 1: 激活提取 + CKA 计算

### 环境要求
- **GPU**: A800-80GB（单卡）
- **显存**: 模型加载 ~40GB + CKA 计算 ~10GB（模型先卸载到 CPU 再算 CKA）
- **磁盘**: 每样本 ~1GB spatial_stats（160 个 .pt 文件），CKA 矩阵 ~200KB
- **Python**: conda activate sky

### 1d. 完整运行验证

```bash
# CKA 矩阵文件
ls /root/autodl-fs/experiments/cka_analysis/cka_matrices/
# 预期: hdtf_0000_cka.npy ... hdtf_0009_cka.npy + cka_average.npy + cka_summary.pt

# 每个样本的 spatial stats 应有 160 个文件
for d in /root/autodl-fs/experiments/cka_analysis/spatial_stats/hdtf_*/; do
    n=$(ls "$d" | wc -l)
    echo "$(basename $d): $n files"
done

# 验证平均矩阵
python -c "
import numpy as np
cka = np.load('/root/autodl-fs/experiments/cka_analysis/cka_matrices/cka_average.npy')
print(f'Average CKA matrix shape: {cka.shape}')
print(f'Diagonal mean: {np.diag(cka).mean():.6f}')  # 应非常接近 1.0
print(f'Off-diagonal mean: {(cka.sum() - np.trace(cka)) / (160*159):.6f}')
print(f'Symmetric: {np.allclose(cka, cka.T, atol=1e-5)}')
"
```

---

## Phase 2: CKA 可视化

**不需要 GPU**，可在无卡模式下运行。

### 2a. 运行 CKA 可视化

```bash
cd /root/SkyReels-V3
conda activate sky

python scripts/cka_visualize.py \
    --input_dir /root/autodl-fs/experiments/cka_analysis/cka_matrices \
    --output_dir /root/autodl-fs/experiments/cka_analysis/visualizations \
    --redundancy_threshold 0.95
```

### 2b. 验证产出

```bash
ls -la /root/autodl-fs/experiments/cka_analysis/visualizations/cka_*.png
ls -la /root/autodl-fs/experiments/cka_analysis/visualizations/cka_summary_stats.json

# 预期文件:
# cka_global_160x160.png         — 全局热力图（最重要）
# cka_depthwise_40x40.png        — 4张 step 内部热力图
# cka_temporal_4x4_grid.png      — 40张 block 跨步热力图
# cka_diagonal_adjacent.png      — 相邻 block CKA 曲线
# cka_cross_step_diagonal.png    — 跨步同 block CKA 曲线
# cka_summary_stats.json         — 统计摘要

# 查看关键统计
cat /root/autodl-fs/experiments/cka_analysis/visualizations/cka_summary_stats.json | python -m json.tool
```

### 2c. 如何解读 CKA 结果

- **对角线高亮块 (CKA ≈ 1.0)**: 表示相邻 block 输出几乎相同，这些 block 是**冗余的**，可以尝试剪枝或共享权重
- **cka_diagonal_adjacent.png 中的高值区间**: CKA(block_i, block_{i+1}) > 0.95 意味着 block_i 和 block_{i+1} 做了几乎相同的事
- **cka_summary_stats.json 中的 `high_redundancy_pairs`**: 列出了每个 step 中 CKA > 0.95 的 block 对，这些是剪枝候选
- **cka_summary_stats.json 中的 `high_redundancy_intervals`**: 把连续高相似 pair 聚合成 block 区间，例如 pair 16,17,18,19 会被汇总成 block 16-20 这一段，可直接作为候选裁减区间
- **跨步 CKA**: 如果 block_b@step_0 和 block_b@step_1 的 CKA 很高，说明该 block 在不同去噪阶段的行为一致，可以考虑跨步共享

---

## Phase 3: Block 任务可视化

**不需要 GPU**，可在无卡模式下运行。

### 3a. 对单个样本可视化

```bash
cd /root/SkyReels-V3
conda activate sky

python scripts/block_task_visualize.py \
    --input_dir /root/autodl-fs/experiments/cka_analysis/spatial_stats \
    --output_dir /root/autodl-fs/experiments/cka_analysis/visualizations/block/hdtf_0000 \
    --sample_id hdtf_0000
```

### 3b. 对所有样本取平均后可视化（推荐）

```bash
python scripts/block_task_visualize.py \
    --input_dir /root/autodl-fs/experiments/cka_analysis/spatial_stats \
    --output_dir /root/autodl-fs/experiments/cka_analysis/visualizations/block
```

### 3c. 验证产出

```bash
ls -la /root/autodl-fs/experiments/cka_analysis/visualizations/block_*.png
ls -la /root/autodl-fs/experiments/cka_analysis/visualizations/block_role_classification.json

# 预期文件:
# block_residual_energy_compact.png    — 残差能量空间热力图 (选取 block)
# block_residual_pca_rgb.png           — PCA 主成分彩色映射
# block_fft_radial_curves.png          — FFT 径向频率曲线
# block_fft_band_heatmap.png           — 低/中/高频能量比例热力图
# block_role_classification.json       — 每个 block 的角色分类
# block_role_matrix.png                — 角色分类离散矩阵

# 查看角色分类结果（前5个）
python -c "
import json
with open('/root/autodl-fs/experiments/cka_analysis/visualizations/block_role_classification.json') as f:
    roles = json.load(f)
for k in sorted(roles.keys())[:5]:
    print(f'{k}: freq={roles[k][\"dominant_frequency\"]}, spatial={roles[k][\"spatial_focus\"]}, energy={roles[k][\"total_energy\"]:.2f}')
"
```

### 3d. 如何解读 Block 任务结果

- **残差能量图 (residual_energy_compact.png)**: 亮区 = block 在该空间位置改变特征最多。观察模式：
  - 早期 block (0-10) 通常全局均匀 → 做全局结构调整
  - 中间 block (10-30) 可能在人脸/嘴唇区域集中 → 语音驱动的面部特征
  - 晚期 block (30-39) 可能是细节增强或全局收尾

- **PCA RGB 图 (residual_pca_rgb.png)**: 不同颜色 = 不同语义方向的残差。颜色一致区域表示该 block 对这些区域做了相似的修改。

- **FFT 频段热力图 (fft_band_heatmap.png)**:
  - Low freq 高 → block 负责大尺度结构/布局
  - Mid freq 高 → block 负责轮廓/边界
  - High freq 高 → block 负责纹理/细节

- **角色分类 (block_role_classification.json)**: 综合频率和空间分析，给出每个 block 的角色标签：
  - `low_freq_structure` + `globally_distributed` → 全局结构 block
  - `high_freq_detail` + `highly_localized` → 局部细节 block
  - `mid_freq_contour` + `moderately_localized` → 轮廓/边界 block

---

## 重要技术备注

### Hook 机制
- 使用 `register_forward_hook(fn, with_kwargs=True)` (PyTorch ≥ 2.0)
- Hook 签名: `hook_fn(module, args, kwargs, output)`
- `args = (x,)` 其中 `x` 是 block 输入 `[B=1, S_padded, 5120]`
- `kwargs` 包含 `seq_lens`, `grid_sizes`, `freqs`, `context` 等
- `output` 是 block 输出 `[B=1, S_padded, 5120]`

### Step 计数
- Pipeline 可能有多 segment（当 audio > 81 frames window 时）
- 通过 monkey-patch `model.forward` 的计数器，只收集前 4 次调用（第一个 segment）
- 后续 segment 的 forward 调用 step > 3 → `set_step(-1)` → hook 跳过

### 内存管理
- CKA 计算前：将模型和 VAE 卸载到 CPU → 释放 ~40GB VRAM
- CKA 矩阵 160×160 float64 只需 ~200KB
- 每个 spatial stat 文件中 `res_norm` (T×H×W float32) + `pca_proj` (3×T×H×W) + `fft_mag` (H×W//2+1) 约 ~5MB

### 720P 典型张量尺寸
- 输入图像: 720×1280 → VAE 编码后 90×120 → patch_embedding (stride 1,2,2) 后 ~90×60 → T=21 帧时 grid_sizes=(21, 45, 60)
- 有效 token 数: `seq_len = T × H × W ≈ 56700`
- 子采样: 8192 个 token → CKA 计算的每个激活为 [8192, 5120]

---

## 如果启用了 block_offload 模式

当前 cka_extract.py 中 `offload=False`，模型完全加载到 GPU。如果 80GB 显存仍不够（不太可能），可以修改为：

```python
pipe = TalkingAvatarPipeline(
    config=config,
    model_path=args.model_id,
    device_id=0,
    rank=0,
    use_usp=False,
    offload=True,      # 改为 True
    low_vram=False,
)
```

注意: `offload=True` 时 block 逐个从 CPU→GPU→CPU，hook 仍然可以正常工作（hook 在 block.forward 时触发，此时 block 已在 GPU 上）。但推理速度会变慢。



