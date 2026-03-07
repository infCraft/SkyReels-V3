#!/usr/bin/env python3
"""
基于 CKA 冗余区间 + 手动指定 stride，生成 skip_list.json。

原理:
    1. 为每个去噪步骤手动指定一个 stride，例如 [3, 2, 1, 30]。
    2. 对每个步骤的 CKA 高冗余区间，按 stride 分段，每段只保留头部 block。
    3. 每段剩余的 block 加入 skip_list。

例如:
    stride = 3, 区间 [15, 20]
    -> [15, 16, 17] 保留 15，跳过 16,17
    -> [18, 19, 20] 保留 18，跳过 19,20

用法:
    python scripts/generate_skip_list.py \
        --cka_summary /root/autodl-fs/experiments/cka_analysis/visualizations/cka_summary_stats.json \
        --output /root/autodl-fs/experiments/cka_analysis/skip_list.json \
        --sampling_steps 4 \
        --strides 3,2,1,30

产出:
    skip_list.json — [[step, block_idx], ...] 格式，
    与 SkyReels-V3-stepblock_pruning 项目的推理代码兼容。
"""

import argparse
import json
import os
import sys


def parse_strides(strides_arg, sampling_steps):
    """解析逗号分隔的 stride 参数，并校验长度与取值。"""
    try:
        strides = [int(part.strip()) for part in strides_arg.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"--strides 必须是逗号分隔的整数列表，例如 3,2,1,30: {exc}") from exc

    if len(strides) != sampling_steps:
        raise ValueError(
            f"--strides 长度必须等于 sampling_steps={sampling_steps}，当前得到 {len(strides)} 个值"
        )

    if any(stride < 1 for stride in strides):
        raise ValueError("--strides 中的每个 stride 都必须 >= 1")

    return strides


# ============================================================
# Skip list 生成
# ============================================================

def get_blocks_to_skip(interval, stride):
    """
    在一个冗余区间内，按 stride 分段，保留每段头部 block，
    其余 block 作为 skip 返回。

    例如 interval = [15, 20]（6 blocks），stride = 3:
        chunk1 = [15, 16, 17] → keep 15, skip 16, 17
        chunk2 = [18, 19, 20] → keep 18, skip 19, 20
        最后剩余不足 stride 个也只保留头部。
    """
    if stride <= 1:
        return []

    start = interval["start_block"]
    num = interval["num_blocks"]
    blocks = list(range(start, start + num))

    skip = []
    for i, b in enumerate(blocks):
        # 每 stride 个 block 中，只有 i % stride == 0 的是"头部"，保留
        if i % stride != 0:
            skip.append(b)
    return skip


def generate_skip_list(cka_summary, sampling_steps, strides):
    """
    主逻辑：结合手动 stride 和 CKA 冗余区间，生成 skip_list。
    """
    print("=" * 60)
    print("手动 stride 配置")
    print("=" * 60)
    for s in range(sampling_steps):
        print(f"  Step {s}: stride={strides[s]}")

    print()

    # 2. 读取每个 step 的 CKA 冗余区间
    skip_list = []
    total_skipped = 0

    print("=" * 60)
    print("Skip list 生成")
    print("=" * 60)

    num_blocks = cka_summary.get("shape", [160, 160])[0] // sampling_steps

    for s in range(sampling_steps):
        stride = strides[s]
        intervals_key = f"step{s}_high_redundancy_intervals"
        intervals = cka_summary.get(intervals_key, [])

        print(f"\n  Step {s} (stride={stride}):")

        if stride <= 1:
            print(f"    → stride ≤ 1，不裁减，保留全部 block")
            continue

        if not intervals:
            print(f"    → 无高冗余区间")
            continue

        step_skipped = 0
        for interval in intervals:
            blocks_to_skip = get_blocks_to_skip(interval, stride)
            for b in blocks_to_skip:
                skip_list.append([s, b])
            step_skipped += len(blocks_to_skip)

            start = interval["start_block"]
            end = start + interval["num_blocks"] - 1
            n_kept = interval["num_blocks"] - len(blocks_to_skip)
            print(f"    区间 [{start}, {end}] ({interval['num_blocks']} blocks, "
                  f"mean_cka={interval['mean_cka']:.4f}): "
                  f"保留 {n_kept}, 跳过 {len(blocks_to_skip)}")

        total_skipped += step_skipped
        print(f"    小计: 跳过 {step_skipped}/{num_blocks} blocks")

    total_blocks = sampling_steps * num_blocks
    print(f"\n{'=' * 60}")
    print(f"汇总: 共跳过 {total_skipped}/{total_blocks} 次 block 计算 "
          f"({total_skipped/total_blocks*100:.1f}% 裁减率)")
    print(f"{'=' * 60}")

    # 排序
    skip_list.sort(key=lambda x: (x[0], x[1]))

    return skip_list


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="基于 CKA 冗余区间 + 手动 stride 生成 skip_list.json"
    )
    parser.add_argument("--cka_summary", type=str, required=True,
                        help="cka_summary_stats.json 路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 skip_list.json 路径")
    parser.add_argument("--sampling_steps", type=int, default=4,
                        help="去噪步数（默认 4）")
    parser.add_argument("--strides", type=str, required=True,
                        help="逗号分隔的 stride 列表，长度必须等于 sampling_steps，例如 3,2,1,30")
    args = parser.parse_args()

    try:
        strides = parse_strides(args.strides, args.sampling_steps)
    except ValueError as exc:
        print(f"错误: {exc}")
        sys.exit(1)

    # 加载 CKA 摘要
    with open(args.cka_summary, "r") as f:
        cka_summary = json.load(f)

    # 检查是否包含区间数据
    has_intervals = any(
        f"step{s}_high_redundancy_intervals" in cka_summary
        for s in range(args.sampling_steps)
    )
    if not has_intervals:
        print("错误: cka_summary_stats.json 中没有 high_redundancy_intervals 字段。")
        print("请先运行 cka_visualize.py 生成含区间数据的摘要。")
        sys.exit(1)

    threshold = cka_summary.get("adjacent_redundancy_threshold", "N/A")
    print(f"CKA 冗余阈值: {threshold}")
    print()

    # 生成 skip_list
    skip_list = generate_skip_list(
        cka_summary,
        sampling_steps=args.sampling_steps,
        strides=strides,
    )

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(skip_list, f, indent=2)

    print(f"\n已保存: {args.output}")
    print(f"共 {len(skip_list)} 个 (step, block) 对")


if __name__ == "__main__":
    main()
