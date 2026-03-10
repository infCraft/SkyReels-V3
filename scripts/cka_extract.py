#!/usr/bin/env python3
"""
CKA (Centered Kernel Alignment) 激活提取与在线CKA计算脚本

对 SkyReels-V3 Talking Avatar (19B) 模型的 4步去噪 × 40 blocks = 160 个中间激活
进行提取，计算 160×160 的 CKA 矩阵，并保存空间统计数据（残差L2范数、PCA投影、FFT频谱）。

用法:
    python scripts/cka_extract.py \
        --model_id /root/autodl-tmp/SkyReels-V3-A2V-19B \
        --manifest_json /root/autodl-fs/experiments/calibration_manifest.json \
        --output_dir /root/autodl-fs/experiments/cka_analysis \
        --num_subsample_tokens 8192 \
        --seed 42
"""

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - cka_extract - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler()],
)

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from skyreels_v3.configs import WAN_CONFIGS
from skyreels_v3.pipelines import TalkingAvatarPipeline
from skyreels_v3.utils.avatar_preprocess import preprocess_audio
from skyreels_v3.utils.profiler import profiler

NUM_BLOCKS = 40
NUM_STEPS = 4
TOTAL_ACTIVATIONS = NUM_BLOCKS * NUM_STEPS  # 160


# ============================================================
# CKA 计算工具函数
# ============================================================

def compute_linear_cka_matrix(activations_list, device="cuda"):
    """
    计算 160 个激活张量之间的线性 CKA 矩阵。

    基于无偏 Linear HSIC 的简化公式:
        HSIC(X, Y) = ||X^T Y||_F^2 / (n-1)^2
    其中 X, Y 已列中心化。由于 (n-1)^2 在归一化时约去，实际只需:
        CKA(X, Y) = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Args:
        activations_list: list of [n_tokens, dim] tensors (CPU, float32)
        device: 计算设备

    Returns:
        cka_matrix: [160, 160] numpy array
    """
    n = len(activations_list)
    logging.info(f"Computing CKA matrix for {n} activations on {device}...")

    # 列中心化并传输到GPU
    centered = []
    for act in activations_list:
        act_gpu = act.to(device)
        act_gpu = act_gpu - act_gpu.mean(dim=0, keepdim=True)
        centered.append(act_gpu)

    # 计算每个激活的自相关范数 ||X_i^T X_i||_F（用于归一化）
    self_hsic = torch.zeros(n, device=device)
    for i in range(n):
        # ||X^T X||_F^2 = sum((X^T X)^2) = ||X X^T||_F^2 (由迹的恒等关系)
        # 但我们直接算 ||X^T X||_F = sqrt(sum((X^T X)_{jk}^2))
        # 注意: HSIC正比于 ||X^T Y||_F^2，所以自相关是 ||X^T X||_F^2
        gram_self = centered[i].T @ centered[i]  # [dim, dim]
        self_hsic[i] = (gram_self ** 2).sum()
        del gram_self

    # 逐行计算交叉 HSIC
    cka_matrix = torch.zeros(n, n, device=device)
    for i in range(n):
        if i % 20 == 0:
            logging.info(f"  CKA row {i}/{n}")
        Xi = centered[i]  # [tokens, dim]
        for j in range(i, n):
            Xj = centered[j]  # [tokens, dim]
            cross = Xi.T @ Xj  # [dim, dim]
            hsic_ij = (cross ** 2).sum()
            cka_val = hsic_ij / torch.sqrt(self_hsic[i] * self_hsic[j])
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
            del cross

    # 清理GPU内存
    del centered, self_hsic
    torch.cuda.empty_cache()

    return cka_matrix.cpu().numpy()


def compute_cosine_similarity_matrix(residuals_list):
    """
    计算 160 个残差张量之间的余弦相似度矩阵（方案 A：mean-pool → cosine）。

    对每个残差 R_i ∈ R^{n×D}，先沿 token 维度取均值得到 r̄_i ∈ R^D
    （表示该 block 的"平均修改方向"），然后计算两两余弦相似度:

        CosSim(i, j) = (r̄_i · r̄_j) / (‖r̄_i‖ · ‖r̄_j‖)

    值域 [-1, 1]。高值表示两个 block 在同方向修改表征。

    Args:
        residuals_list: list of [n_tokens, dim] tensors (CPU, float32)

    Returns:
        cosine_matrix: [160, 160] numpy array, float64
    """
    n = len(residuals_list)
    logging.info(f"Computing cosine similarity matrix for {n} residuals...")

    # Step 1: 沿 token 维度取均值  r̄_i = (1/n) Σ_k R_i[k, :]
    mean_residuals = torch.stack([r.mean(dim=0) for r in residuals_list])  # [160, D]

    # Step 2: 行归一化  r̂_i = r̄_i / ‖r̄_i‖
    norms = mean_residuals.norm(dim=1, keepdim=True).clamp(min=1e-12)  # [160, 1]
    normalized = mean_residuals / norms  # [160, D]

    # Step 3: 余弦矩阵 C = r̂ @ r̂^T，即 C[i,j] = r̂_i · r̂_j
    cosine_matrix = (normalized @ normalized.T).numpy().astype(np.float64)  # [160, 160]

    return cosine_matrix


# ============================================================
# 激活 Hook 管理器
# ============================================================

class ActivationCollector:
    """
    管理 forward hooks，在每个 WanAttentionBlock 的输出处收集激活。

    收集两类数据：
    1. 子采样激活 [num_subsample_tokens, dim] 用于CKA计算
    2. 空间统计（残差L2范数、PCA投影、FFT频谱）用于block任务可视化
    """

    def __init__(self, model, num_subsample_tokens=8192, seed=42):
        self.model = model
        self.num_subsample_tokens = num_subsample_tokens
        self.seed = seed

        self.hooks = []
        self.current_step = -1

        # 存储：key = (step, block_idx)
        self.subsampled_activations = {}  # {(step, blk): [n_tokens, dim] CPU float32}
        self.subsampled_residuals = {}   # {(step, blk): [n_tokens, dim] CPU float32}
        self.relative_magnitude = {}     # {(step, blk): float, mean(‖r‖)/mean(‖x_in‖)}
        self.spatial_stats = {}          # {(step, blk): dict of stats}

        # 子采样索引（在第一次 forward 时根据实际 seq_len 计算）
        self._subsample_indices = None
        self._seq_len = None  # 有效 token 数
        self._grid_sizes = None  # (T, H, W) 空间网格

    def register_hooks(self):
        """注册 forward hooks 到模型的所有 blocks（使用 with_kwargs=True 以获取 seq_lens 等）。"""
        for blk_idx, block in enumerate(self.model.blocks):
            hook = block.register_forward_hook(
                self._make_hook_fn(blk_idx),
                with_kwargs=True,
            )
            self.hooks.append(hook)
        logging.info(f"Registered {len(self.hooks)} forward hooks")

    def remove_hooks(self):
        """移除所有 hooks。"""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        logging.info("Removed all forward hooks")

    def set_step(self, step_idx):
        """设置当前去噪步编号（由外部循环调用）。"""
        self.current_step = step_idx

    def clear(self):
        """清除所有已收集的数据。"""
        self.subsampled_activations.clear()
        self.subsampled_residuals.clear()
        self.relative_magnitude.clear()
        self.spatial_stats.clear()
        self._subsample_indices = None
        self._seq_len = None
        self._grid_sizes = None

    def _make_hook_fn(self, blk_idx):
        """为指定 block 创建 hook 函数（with_kwargs=True 签名）。"""
        collector = self

        def hook_fn(module, args, kwargs, output):
            step = collector.current_step
            if step < 0:
                return  # hook 未激活

            # args = (x,)  其中 x 是 block 的输入张量 [B, S_pad, D]
            # kwargs 包含 seq_lens, grid_sizes 等
            # output 是 block forward 的返回值 [B, S_pad, D]
            x_in = args[0].detach()   # [B, S_pad, D]
            x_out = output.detach()    # [B, S_pad, D]

            # 从 kwargs 中提取空间信息
            seq_lens = kwargs.get('seq_lens')
            grid_sizes = kwargs.get('grid_sizes')

            if seq_lens is None:
                logging.warning(f"seq_lens not found in kwargs for step={step}, block={blk_idx}")
                return

            seq_len = seq_lens[0].item()
            grid = grid_sizes[0]
            T, H, W = grid[0].item(), grid[1].item(), grid[2].item()

            # 更新 collector 状态
            if collector._seq_len is None:
                collector._seq_len = seq_len
                collector._grid_sizes = (T, H, W)
                logging.info(f"Captured seq_len={seq_len}, grid_sizes=({T}, {H}, {W})")

            x_in_valid = x_in[0, :seq_len, :]    # [seq_len, D]
            x_out_valid = x_out[0, :seq_len, :]   # [seq_len, D]

            # 初始化子采样索引
            if collector._subsample_indices is None:
                n_tokens = seq_len
                n_sub = min(collector.num_subsample_tokens, n_tokens)
                rng = torch.Generator()
                rng.manual_seed(collector.seed)
                perm = torch.randperm(n_tokens, generator=rng)
                collector._subsample_indices = perm[:n_sub].sort().values
                logging.info(f"Subsample indices: {n_sub} out of {n_tokens} tokens")

            sub_idx = collector._subsample_indices.to(x_out_valid.device)

            # 1. 子采样激活 (CKA用)
            sub_act = x_out_valid[sub_idx].cpu().float()  # [n_sub, D]
            collector.subsampled_activations[(step, blk_idx)] = sub_act

            # 2. 残差计算
            residual = (x_out_valid - x_in_valid).float()  # [seq_len, D]

            # 2a. 子采样残差 (余弦相似度矩阵用)
            sub_res = residual[sub_idx].cpu().float()  # [n_sub, D]
            collector.subsampled_residuals[(step, blk_idx)] = sub_res

            # 2b. 相对残差幅度 (per-block 标量)
            # RelMag_{s,b} = mean_k(‖r_k‖₂) / mean_k(‖x_in_k‖₂)
            # 其中 r_k = x_out_k − x_in_k，k 遍历所有有效空间 token
            n_spatial = T * H * W
            with torch.no_grad():
                res_norms = residual[:n_spatial].norm(dim=-1)  # [n_spatial]
                xin_norms = x_in_valid[:n_spatial].float().norm(dim=-1)  # [n_spatial]
                rel_mag = res_norms.mean().item() / (xin_norms.mean().item() + 1e-12)
            collector.relative_magnitude[(step, blk_idx)] = rel_mag

            # 3. 空间统计
            stats = {}

            # 3a. 残差 L2 范数 → reshape to (T, H, W)
            res_norm_spatial = res_norms.reshape(T, H, W)
            stats['res_norm'] = res_norm_spatial.cpu()

            # 3b. PCA 投影 (top-3 主成分)
            # PCA: 对子采样残差列中心化 X_c = X − μ，SVD(X_c) = UΣV^T
            # 投影: proj = (X_full − μ) @ V，其中 V ∈ R^{D×3}
            sub_residual = residual[sub_idx]  # [n_sub, D]
            res_mean = sub_residual.mean(dim=0, keepdim=True)  # [1, D]
            sub_res_centered = sub_residual - res_mean
            try:
                _, _, V = torch.pca_lowrank(sub_res_centered, q=3, niter=4)  # V: [D, 3]
                pca_proj = (residual[:n_spatial] - res_mean) @ V  # [n_spatial, 3]
                pca_proj_spatial = pca_proj.reshape(T, H, W, 3).permute(3, 0, 1, 2)  # [3, T, H, W]
                stats['pca_proj'] = pca_proj_spatial.cpu()
            except Exception as e:
                logging.warning(f"PCA failed for step={step}, block={blk_idx}: {e}")
                stats['pca_proj'] = torch.zeros(3, T, H, W)

            collector.spatial_stats[(step, blk_idx)] = stats

            # 释放 GPU 上的中间张量
            del residual, res_norms, xin_norms, sub_residual, sub_res_centered, res_mean
            del x_in_valid, x_out_valid

        return hook_fn


# ============================================================
# 主流程 — 多 GPU 数据并行
# ============================================================

def worker_fn(rank, num_gpus, args):
    """
    单 GPU worker：加载模型到 GPU `rank`，处理分配到的样本，保存 per-sample 结果到磁盘。

    样本分配采用交错策略: manifest[rank::num_gpus]，保证各 GPU 负载均衡。
    """
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # 加载 manifest 并分配样本
    with open(args.manifest_json, "r") as f:
        all_items = json.load(f)
    my_indices = list(range(rank, len(all_items), num_gpus))
    my_items = [all_items[i] for i in my_indices]

    if not my_items:
        logging.info(f"[GPU {rank}] No samples assigned, exiting")
        return

    logging.info(f"[GPU {rank}] Assigned {len(my_items)}/{len(all_items)} samples (indices: {my_indices})")

    cka_dir = os.path.join(args.output_dir, "cka_matrices")
    spatial_dir = os.path.join(args.output_dir, "spatial_stats")

    # 初始化 Pipeline（每个 worker 独立加载模型到自己的 GPU）
    config = WAN_CONFIGS["talking-avatar-19B"]
    logging.info(f"[GPU {rank}] Initializing TalkingAvatarPipeline on {device}...")
    pipe = TalkingAvatarPipeline(
        config=config,
        model_path=args.model_id,
        device_id=rank,
        rank=0,
        use_usp=False,
        offload=False,
        low_vram=False,
    )
    logging.info(f"[GPU {rank}] Pipeline initialized")

    # 创建 ActivationCollector
    collector = ActivationCollector(
        model=pipe.model,
        num_subsample_tokens=args.num_subsample_tokens,
        seed=args.seed,
    )
    collector.register_hooks()

    # 每个 worker 使用独立的音频预处理目录，避免文件冲突
    audio_dir = os.path.join("processed_audio", f"gpu_{rank}")

    # 逐样本处理
    for local_idx, (orig_idx, item) in enumerate(zip(my_indices, my_items)):
        sample_id = item.get("id", f"sample_{orig_idx:04d}")
        logging.info(f"[GPU {rank}] {'='*50}")
        logging.info(f"[GPU {rank}] [{local_idx + 1}/{len(my_items)}] Processing: {sample_id}")
        logging.info(f"[GPU {rank}] {'='*50}")

        # 准备输入数据
        input_data = {
            "prompt": item.get("prompt", "A person is talking to the camera."),
            "cond_image": item["ref_image_path"],
            "cond_audio": {"person1": item["audio_path"]},
        }

        # 音频预处理
        logging.info(f"[GPU {rank}] Preprocessing audio for {sample_id}...")
        input_data, _ = preprocess_audio(args.model_id, input_data, audio_dir)

        # 清理 collector 状态
        collector.clear()

        # Monkey-patch model.forward 以跟踪步骤编号
        original_model_forward = pipe.model.forward
        step_counter = [0]

        def counting_forward(*args_f, **kwargs_f):
            current_step = step_counter[0]
            if current_step < NUM_STEPS:
                collector.set_step(current_step)
            else:
                collector.set_step(-1)
            result = original_model_forward(*args_f, **kwargs_f)
            step_counter[0] += 1
            return result

        pipe.model.forward = counting_forward

        # 执行推理
        try:
            logging.info(f"[GPU {rank}] Running inference for {sample_id}...")
            step_counter[0] = 0

            kwargs = {
                "input_data": input_data,
                "size_buckget": "720P",
                "motion_frame": 5,
                "frame_num": 81,
                "drop_frame": 12,
                "shift": 11,
                "text_guide_scale": 1.0,
                "audio_guide_scale": 1.0,
                "seed": args.seed,
                "sampling_steps": NUM_STEPS,
                "max_frames_num": 5000,
            }
            _ = pipe.generate(**kwargs)

            logging.info(f"[GPU {rank}] Inference complete. Total model.forward calls: {step_counter[0]}. "
                         f"Collected {len(collector.subsampled_activations)} activations")

        finally:
            pipe.model.forward = original_model_forward

        # 验证收集的激活数量
        expected = TOTAL_ACTIVATIONS
        actual = len(collector.subsampled_activations)
        if actual != expected:
            logging.warning(f"[GPU {rank}] Expected {expected} activations, got {actual}. Skipping this sample.")
            continue

        # 保存空间统计数据
        sample_spatial_dir = os.path.join(spatial_dir, sample_id)
        os.makedirs(sample_spatial_dir, exist_ok=True)
        for (step, blk_idx), stats in collector.spatial_stats.items():
            save_path = os.path.join(sample_spatial_dir, f"step{step}_block{blk_idx}.pt")
            torch.save(stats, save_path)
        logging.info(f"[GPU {rank}] Saved spatial stats to {sample_spatial_dir}")

        # 构建激活列表（按 step 和 block 排序）
        activation_list = []
        for step in range(NUM_STEPS):
            for blk in range(NUM_BLOCKS):
                activation_list.append(collector.subsampled_activations[(step, blk)])

        # 将模型移到 CPU 以释放显存给 CKA 计算
        logging.info(f"[GPU {rank}] Offloading model to CPU for CKA computation...")
        pipe.model.to("cpu")
        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        gc.collect()

        # 在 GPU 上计算 CKA 矩阵
        logging.info(f"[GPU {rank}] Computing CKA matrix ({TOTAL_ACTIVATIONS}x{TOTAL_ACTIVATIONS})...")
        t_start = time.time()
        cka_matrix = compute_linear_cka_matrix(activation_list, device=device)
        t_elapsed = time.time() - t_start
        logging.info(f"[GPU {rank}] CKA computation took {t_elapsed:.1f}s")

        # 保存 per-sample CKA 矩阵
        cka_path = os.path.join(cka_dir, f"{sample_id}_cka.npy")
        np.save(cka_path, cka_matrix)
        logging.info(f"[GPU {rank}] Saved CKA matrix to {cka_path}")

        # 构建残差列表并计算余弦相似度矩阵 (CPU 计算)
        residual_list = []
        for step in range(NUM_STEPS):
            for blk in range(NUM_BLOCKS):
                residual_list.append(collector.subsampled_residuals[(step, blk)])
        cosine_matrix = compute_cosine_similarity_matrix(residual_list)
        cosine_path = os.path.join(cka_dir, f"{sample_id}_residual_cosine.npy")
        np.save(cosine_path, cosine_matrix)
        logging.info(f"[GPU {rank}] Saved residual cosine matrix to {cosine_path}")

        # 保存 per-sample 相对残差幅度 [NUM_BLOCKS, NUM_STEPS]
        rel_mag_array = np.zeros((NUM_BLOCKS, NUM_STEPS))
        for (step, blk), val in collector.relative_magnitude.items():
            rel_mag_array[blk, step] = val
        rel_mag_path = os.path.join(cka_dir, f"{sample_id}_relative_magnitude.npy")
        np.save(rel_mag_path, rel_mag_array)
        logging.info(f"[GPU {rank}] Saved relative magnitude to {rel_mag_path}")

        # 释放内存
        del activation_list, cka_matrix, residual_list, cosine_matrix, rel_mag_array
        collector.clear()
        gc.collect()

        # 将模型移回 GPU
        logging.info(f"[GPU {rank}] Moving model back to {device}...")
        pipe.model.to(device)
        pipe.vae.to(device)
        torch.cuda.empty_cache()

    # 清理
    collector.remove_hooks()
    logging.info(f"[GPU {rank}] Worker finished. Processed {len(my_items)} samples.")


def aggregate_results(args):
    """
    从磁盘加载所有 per-sample 结果，计算跨样本平均值并保存。

    在所有 worker 完成后由主进程调用。
    """
    cka_dir = os.path.join(args.output_dir, "cka_matrices")

    # 发现所有 per-sample CKA 文件
    all_files = os.listdir(cka_dir)
    cka_files = sorted(f for f in all_files
                       if f.endswith("_cka.npy") and f != "cka_average.npy")

    if not cka_files:
        logging.warning("No per-sample CKA files found for aggregation!")
        return

    n_samples = len(cka_files)
    logging.info(f"Aggregating results from {n_samples} samples...")

    cka_sum = None
    cosine_sum = None
    cosine_count = 0
    rel_mag_sum = None
    rel_mag_count = 0

    for cka_file in cka_files:
        sample_id = cka_file.replace("_cka.npy", "")

        cka_matrix = np.load(os.path.join(cka_dir, cka_file))
        if cka_sum is None:
            cka_sum = cka_matrix.copy()
        else:
            cka_sum += cka_matrix

        cosine_path = os.path.join(cka_dir, f"{sample_id}_residual_cosine.npy")
        if os.path.exists(cosine_path):
            cosine_matrix = np.load(cosine_path)
            if cosine_sum is None:
                cosine_sum = cosine_matrix.copy()
            else:
                cosine_sum += cosine_matrix
            cosine_count += 1

        rel_mag_path = os.path.join(cka_dir, f"{sample_id}_relative_magnitude.npy")
        if os.path.exists(rel_mag_path):
            rel_mag_array = np.load(rel_mag_path)
            if rel_mag_sum is None:
                rel_mag_sum = rel_mag_array.copy()
            else:
                rel_mag_sum += rel_mag_array
            rel_mag_count += 1

    # 保存平均值
    cka_avg = cka_sum / n_samples
    np.save(os.path.join(cka_dir, "cka_average.npy"), cka_avg)
    logging.info(f"Saved average CKA matrix ({n_samples} samples)")

    summary = {
        "cka_average": torch.from_numpy(cka_avg),
        "n_samples": n_samples,
        "num_blocks": NUM_BLOCKS,
        "num_steps": NUM_STEPS,
        "num_subsample_tokens": args.num_subsample_tokens,
    }

    if cosine_sum is not None:
        cosine_avg = cosine_sum / cosine_count
        np.save(os.path.join(cka_dir, "residual_cosine_average.npy"), cosine_avg)
        summary["cosine_average"] = torch.from_numpy(cosine_avg)
        logging.info(f"Saved average residual cosine matrix ({cosine_count} samples)")

    if rel_mag_sum is not None:
        rel_mag_avg = rel_mag_sum / rel_mag_count
        np.save(os.path.join(cka_dir, "relative_magnitude_average.npy"), rel_mag_avg)
        summary["relative_magnitude_average"] = torch.from_numpy(rel_mag_avg)
        logging.info(f"Saved average relative magnitude ({rel_mag_count} samples)")

    torch.save(summary, os.path.join(cka_dir, "cka_summary.pt"))
    logging.info(f"Aggregation complete. Results in {cka_dir}")


def run_extraction(args):
    """主入口：启动 worker 并聚合结果。"""

    # 预创建输出目录
    cka_dir = os.path.join(args.output_dir, "cka_matrices")
    spatial_dir = os.path.join(args.output_dir, "spatial_stats")
    os.makedirs(cka_dir, exist_ok=True)
    os.makedirs(spatial_dir, exist_ok=True)

    num_gpus = args.num_gpus

    # 检查 GPU 可用性
    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        logging.warning(f"Requested {num_gpus} GPUs but only {available_gpus} available. "
                        f"Using {available_gpus} GPUs.")
        num_gpus = available_gpus

    if num_gpus > 1:
        logging.info(f"Launching {num_gpus} workers for data-parallel extraction...")
        mp.spawn(worker_fn, nprocs=num_gpus, args=(num_gpus, args))
    else:
        worker_fn(0, 1, args)

    # 所有 worker 完成后，聚合 per-sample 结果
    aggregate_results(args)

    logging.info("CKA extraction complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CKA Activation Extraction for SkyReels-V3")
    parser.add_argument("--model_id", type=str, required=True,
                        help="Model path (e.g., /root/autodl-tmp/SkyReels-V3-A2V-19B)")
    parser.add_argument("--manifest_json", type=str, required=True,
                        help="Path to calibration manifest JSON")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for CKA results")
    parser.add_argument("--num_subsample_tokens", type=int, default=8192,
                        help="Number of tokens to subsample for CKA computation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs for data-parallel extraction (default: 1)")
    args = parser.parse_args()

    run_extraction(args)
