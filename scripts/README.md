# 运行指南

**Step 1: 提取（需要有卡模式 A800）**
```bash
sh scripts/run_cka_extraction_multigpu.sh
```

产出（在 `cka_analysis_v2/cka_matrices/` 下）：
- `{sample_id}_cka.npy` — per-sample CKA 矩阵 [160×160]
- `{sample_id}_residual_cosine.npy` — per-sample 余弦矩阵 [160×160]
- `{sample_id}_relative_magnitude.npy` — per-sample 相对幅度 [40×4]
- `cka_average.npy`, `residual_cosine_average.npy`, `relative_magnitude_average.npy` — 跨样本平均
- `cka_summary.pt` — 汇总 .pt

**Step 2: 可视化（无卡模式即可）**
```bash
# CKA + 余弦 + 幅度矩阵可视化
python scripts/cka_visualize.py \
    --input_dir /root/autodl-fs/experiments/cka_analysis_v2/cka_matrices \
    --output_dir /root/autodl-fs/experiments/cka_analysis_v2/visualizations

# 空间可视化（指定单样本做 PCA）
python scripts/block_task_visualize.py \
    --input_dir /root/autodl-fs/experiments/cka_analysis_v2/spatial_stats \
    --output_dir /root/autodl-fs/experiments/cka_analysis_v2/visualizations \
    --sample_id hdtf_0000
```

**Step 3: 生成 skip_list（与之前相同）**
```bash
python scripts/generate_skip_list.py \
    --cka_summary /root/autodl-fs/experiments/cka_analysis_v2/visualizations/cka_summary_stats.json \
    --output /root/autodl-fs/experiments/cka_analysis_v2/skip_list.json \
    --sampling_steps 4 \
    --strides 3,2,1,30
```
