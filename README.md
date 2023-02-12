# 视频HOI使用说明

## 1. 算法描述

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

MMVideoHOI().inference()
```
