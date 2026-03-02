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

生成好的视频，音频和参考图片会存在`experiments/hdtf_processed/`里面。在`experiments/`文件夹下还会存在两个manifest文件，代表标定数据集和测试数据集。

