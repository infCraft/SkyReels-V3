#!/usr/bin/env python3
"""
子模块级能量探测核心模块。

通过 monkey-patch WanAttentionBlock.forward() 方法，在 SA / CCA / ACA / FFN
四个残差连接点前后捕获隐状态 x，在 GPU 上在线计算三种标量度量指标：
  - cos_sim   : 余弦相似度（方向偏移）
  - res_mag   : 相对 L2 残差幅值（能量注入）
  - energy    : 综合能量指数 = (1 - cos_sim) × res_mag

额外记录 SA / FFN 的 AdaLN 调制因子 e[2] / e[5] 的平均绝对值，用于
区分 "子模块本身低能量" 和 "网络主动抑制其输出" 两种冗余机制。

设计原则：
  - 不修改原始仓库任何代码，全部通过 monkey-patch 实现
  - 在线降维：高维张量 → 标量，立即释放，显存开销可忽略
  - 提供 install(model) / remove(model) 接口

数据结构：
  probe_results[step, block_idx, submodule] = {
      cos_sim, res_mag, energy,
      cos_sim_var, res_mag_var, energy_var,   # token 级方差
      mod_scale             # e[2] 或 e[5] 的平均绝对值 (仅 SA/FFN)
  }
"""

import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 子模块名称常量
SA = "SA"
CCA = "CCA"
ACA = "ACA"
FFN = "FFN"
SUBMODULE_NAMES = (SA, CCA, ACA, FFN)


# ────────────────────────────────────────
# 度量计算（纯函数，可独立测试）
# ────────────────────────────────────────

def compute_metrics(x_in: torch.Tensor, x_out: torch.Tensor) -> dict:
    """
    在线计算三种标量度量指标及其 token 级方差。

    Parameters
    ----------
    x_in, x_out : Tensor, shape [B, S, C]  (或可 view 为此形状)

    Returns
    -------
    dict with keys: cos_sim, res_mag, energy,
                    cos_sim_var, res_mag_var, energy_var
    """
    # 展平为 [N, C]  (N = B*S)
    flat_in = x_in.reshape(-1, x_in.shape[-1])
    flat_out = x_out.reshape(-1, x_out.shape[-1])

    # token 级余弦相似度  [N]
    cos_per_token = F.cosine_similarity(flat_in, flat_out, dim=-1)

    # token 级相对 L2 残差幅值  [N]
    residual = flat_out - flat_in
    in_norm = flat_in.norm(p=2, dim=-1).clamp(min=1e-8)
    res_per_token = residual.norm(p=2, dim=-1) / in_norm

    # token 级综合能量  [N]
    energy_per_token = (1.0 - cos_per_token) * res_per_token

    return {
        "cos_sim": cos_per_token.mean().item(),
        "res_mag": res_per_token.mean().item(),
        "energy": energy_per_token.mean().item(),
        "cos_sim_var": cos_per_token.var().item(),
        "res_mag_var": res_per_token.var().item(),
        "energy_var": energy_per_token.var().item(),
    }


# ────────────────────────────────────────
# Monkey-patched forward
# ────────────────────────────────────────

def _make_probed_forward(original_forward, block_idx: int, probe_results: dict,
                         step_getter):
    """
    构造替代 WanAttentionBlock.forward 的 wrapper。

    Parameters
    ----------
    original_forward : bound method
        原始 block.forward。
    block_idx : int
        该 block 在 model.blocks 中的索引。
    probe_results : dict
        共享的数据存储字典 (mutable reference)。
    step_getter : callable() -> int
        返回当前去噪步索引。
    """

    def probed_forward(
        self_block,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        enable_sp=False,
        audio_embedding=None,
        ref_target_masks=None,
        human_num=None,
        audio_mask=None,
    ):
        step = step_getter()
        dtype = x.dtype
        e_chunks = (self_block.modulation + e).chunk(6, dim=1)

        # ──── SA ────
        x_before_sa = x.detach()
        y, x_ref_attn_map = self_block.self_attn(
            (self_block.norm1(x) * (1 + e_chunks[1]) + e_chunks[0]),
            seq_lens,
            grid_sizes,
            freqs,
            ref_target_masks=ref_target_masks,
            human_num=human_num,
        )
        x = x + y * e_chunks[2]
        x = x.to(dtype)
        # 计算 SA 度量
        metrics_sa = compute_metrics(x_before_sa, x)
        metrics_sa["mod_scale"] = e_chunks[2].abs().mean().item()
        probe_results[(step, block_idx, SA)] = metrics_sa
        del x_before_sa

        # ──── CCA ────
        x_before_cca = x.detach()
        x = x + self_block.cross_attn(self_block.norm3(x), context, context_lens)
        metrics_cca = compute_metrics(x_before_cca, x)
        probe_results[(step, block_idx, CCA)] = metrics_cca
        del x_before_cca

        # ──── ACA ────
        x_before_aca = x.detach()
        x_a = self_block.audio_cross_attn(
            self_block.norm_x(x),
            encoder_hidden_states=audio_embedding,
            shape=grid_sizes[0],
            x_ref_attn_map=x_ref_attn_map,
            human_num=human_num,
            enable_sp=enable_sp,
        )
        if audio_mask is not None:
            x_a = x_a * audio_mask
        x = x + x_a
        metrics_aca = compute_metrics(x_before_aca, x)
        probe_results[(step, block_idx, ACA)] = metrics_aca
        del x_before_aca

        # ──── FFN ────
        x_before_ffn = x.detach()
        y = self_block.ffn((self_block.norm2(x) * (1 + e_chunks[4]) + e_chunks[3]))
        x = x + y * e_chunks[5]
        x = x.to(dtype)
        metrics_ffn = compute_metrics(x_before_ffn, x)
        metrics_ffn["mod_scale"] = e_chunks[5].abs().mean().item()
        probe_results[(step, block_idx, FFN)] = metrics_ffn
        del x_before_ffn

        return x

    return probed_forward


# ────────────────────────────────────────
# 对外接口
# ────────────────────────────────────────

class SubmoduleProbeWrapper:
    """
    子模块级能量探测控制器。

    Usage
    -----
    >>> wrapper = SubmoduleProbeWrapper()
    >>> wrapper.install(model)            # monkey-patch all blocks
    >>> # ... run inference, set wrapper.current_step before each model call ...
    >>> data = wrapper.collect()          # {(step, blk, sub): {...}}
    >>> wrapper.remove(model)             # 恢复原始 forward
    """

    def __init__(self):
        self.current_step: int = -1
        self.probe_results: Dict[Tuple[int, int, str], dict] = {}
        self._original_forwards = {}   # block_idx -> original bound method

    # ────────── install ──────────

    def install(self, model) -> None:
        """Monkey-patch 所有 WanAttentionBlock 的 forward。"""
        blocks = model.blocks
        self.probe_results.clear()
        self._original_forwards.clear()

        for idx, block in enumerate(blocks):
            self._original_forwards[idx] = block.forward
            patched = _make_probed_forward(
                original_forward=block.forward,
                block_idx=idx,
                probe_results=self.probe_results,
                step_getter=lambda: self.current_step,
            )
            # 将 patched 函数绑定到 block 实例
            import types
            block.forward = types.MethodType(patched, block)

        logger.info(f"SubmoduleProbeWrapper installed on {len(blocks)} blocks.")

    # ────────── remove ──────────

    def remove(self, model) -> None:
        """恢复所有 block 的原始 forward。"""
        blocks = model.blocks
        for idx, block in enumerate(blocks):
            if idx in self._original_forwards:
                block.forward = self._original_forwards[idx]
        self._original_forwards.clear()
        logger.info("SubmoduleProbeWrapper removed; original forwards restored.")

    # ────────── collect ──────────

    def collect(self) -> Dict[Tuple[int, int, str], dict]:
        """返回已采集的探测数据（引用）。"""
        return self.probe_results

    def clear(self) -> None:
        """清空探测数据（用于下一个样本）。"""
        self.probe_results.clear()

    # ────────── 序列化 ──────────

    @staticmethod
    def serialize(probe_results: dict) -> dict:
        """
        将 (step, block_idx, submodule) tuple 键转为
        "{step}_{block}_{submodule}" 字符串键，便于 JSON 存储。
        """
        out = {}
        for (step, blk, sub), metrics in probe_results.items():
            key = f"{step}_{blk}_{sub}"
            out[key] = metrics
        return out

    @staticmethod
    def deserialize(data: dict) -> Dict[Tuple[int, int, str], dict]:
        """反序列化：字符串键 → tuple 键。"""
        out = {}
        for key, metrics in data.items():
            parts = key.split("_", 2)
            step, blk, sub = int(parts[0]), int(parts[1]), parts[2]
            out[(step, blk, sub)] = metrics
        return out
