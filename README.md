# 视频人-物交互关系检测使用说明

## 1. 算法描述
该算法用于视频人-物交互关系检测。该算法基于PyTorch框架开发，输入一段视频，算法会检测视频中人和物的交互关系，输出人、物、关系三元组。

## 2. 依赖及安装

CUDA版本: 11.7
其他依赖库的安装命令如下：

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

可使用如下命令下载安装算法包：
```bash
pip install -U mmkg-video-hoi
```

## 3. 使用示例及运行参数说明

```python
from mmkg_video_hoi import MMVideoHOI

MMVideoHOI().inference(frames)
```

## 4. 论文
```
@inproceedings{chiou2021st,
title = {ST-HOI: A Spatial-Temporal Baseline for Human-Object Interaction Detection in Videos},
author = {Chiou, Meng-Jiun and Liao, Chun-Yu and Wang, Li-Wei and Zimmermann, Roger and Feng, Jiashi},
booktitle = {Proceedings of the 2021 Workshop on Intelligent Cross-Data Analysis and Retrieval},
pages = {9–17},
year = {2021},
}
```
