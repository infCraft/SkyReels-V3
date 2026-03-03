# 按照[step:block]冗余度排序，去掉r%
## 数据集准备
实验使用的是HDTF数据集。

1. 首先运行`python tools/download_hdtf_youtube.py --output_dir /root/autodl-fs/experiments/hdtf_raw --max_videos 0 2>&1`，下载hdtf数据集到`/root/autodl-fs`当中，可以按需修改目录和指令。
2. 然后运行预处理脚本：

```sh
python tools/prepare_hdtf.py \
    --raw_dir /root/autodl-fs/experiments/hdtf_raw/clips \
    --output_dir /root/autodl-fs/experiments/hdtf_processed \
    --experiment_dir /root/autodl-fs/experiments \
    --min_duration 5.0 \
    --max_duration 300.0 \
    --num_samples 50 \
    --seed 42 2>&1
```

生成好的视频，音频和参考图片会存在`/root/autodl-fs/experiments/hdtf_processed/`里面。在`/root/autodl-fs/experiments/`文件夹下还会存在两个manifest文件，代表标定数据集和测试数据集。

## 跑通baseline推理
开启虚拟环境：

```sh
conda activate sky
```

在虚拟环境当中运行40个测试数据集的baseline：

```sh
sh scripts/run_baseline.sh
```

## 跑通探针
首先运行探针推理：

```sh
cd /root/SkyReels-V3
conda activate sky
bash scripts/run_probe.sh
```

然后分析探针推理得到的结果：

```sh
python tools/analyze_probe.py \
      --input_dir /root/autodl-fs/experiments/probe_results_raw \
      --output_dir /root/autodl-fs/experiments/probe_results_processed \
      --num_steps 4 --num_blocks 40 --probe_alpha 0.75
```

## 简单评测2个视频
```sh
bash scripts/run_probe.sh
```

然后发现生成出来的依托。



## 数据指标评测
首先先把GT video的帧率都转成了25fps。

```sh
python tools/convert_gt_fps.py --manifest /root/autodl-fs/experiments/calibration_manifest.json --manifest /root/autodl-fs/experiments/evaluation_manifest.json --target_fps 25 --skip_if_close
```

期望评测的指标：

| 指标 | 方向 | 范围 | 说明 |
|------|------|------|------|
| **CSIM** | ↑ 越高越好 | [0, 1] | ArcFace 身份余弦相似度（ref image vs 生成帧） | 
| **PSNR** | ↑ 越高越好 | dB | 峰值信噪比（逐帧，gen vs GT） |
| **SSIM** | ↑ 越高越好 | [0, 1] | 结构相似度（逐帧，gen vs GT） |
| **FID**  | ↓ 越低越好 | ≥ 0 | Fréchet Inception Distance（数据集级，帧分布） | 
| **FVD**  | ↓ 越低越好 | ≥ 0 | Fréchet Video Distance（标准 I3D Kinetics-400） | 
| **LSE-D** | ↓ 越低越好 | ≥ 0 | SyncNet 唇音同步欧氏距离 | python_speech_features, 
| **LSE-C** | ↑ 越高越好 | [-1, 1] | SyncNet 唇音同步余弦置信度 | 


