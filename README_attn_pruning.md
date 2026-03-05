# Cross Attention Pruning实验指南
## Submodule级别计算指标探测
本实验将 SkyReels-V3 Talking Avatar 模型（19B, 40 Blocks）的探测粒度从
**block 级**细化到 **sub-module 级**（SA / CCA / ACA / FFN），在 4 步去噪 × 40 层 × 4 子模块 = **640 维**的能量图谱空间中，寻找可安全跳过的冗余计算单元。

### 新增文件清单

| 文件 | 说明 |
|------|------|
| `tools/submodule_probe.py` | 核心探测逻辑（monkey-patch + 在线度量） |
| `generate_video_submodule_probe.py` | 推理入口脚本（批量探测） |
| `scripts/run_submodule_probe.sh` | 一键批量执行脚本 |
| `tools/analyze_submodule_probe.py` | 聚合分析与可视化 |

### 完整运行流程

先进行完整的校准数据集推理：

```bash
cd /root/SkyReels-V3
conda activate sky
bash scripts/run_submodule_probe.sh
```

接下来运行分析脚本。

```bash
conda run -n sky --no-capture-output python tools/analyze_submodule_probe.py \
    --input_dir /root/autodl-fs/experiments/submodule_probe_raw \
    --output_dir /root/autodl-fs/experiments/submodule_probe_processed \
    --num_steps 4 \
    --num_blocks 40 \
    --energy_threshold 0.0001
```

运行完毕后将会在output_dir得到所有的评测结果。

## 跳过部分cross-attention以得到少量加速（等待继续）

### 方案1：前P个和后P个block完整计算，然后中间的cross attention每隔T个才算1次

首先，让AI实现在generate_video处能够识别一个`skip.json`文件，这个文件里面存放了(step, block)块，代表着这些块在去噪的时候，不计算两个cross attention。

先设P=2，剩下36个块，设T=4，只在后2个step进行去噪