# Towards Fast, Accurate and Stable 3D Dense Face Alignment

[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
![GitHub repo size](https://img.shields.io/github/repo-size/cleardusk/3DDFA_V2.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/cleardusk/3ddfa_v2/badge)](https://www.codefactor.io/repository/github/cleardusk/3ddfa_v2)

By [Jianzhu Guo](https://guojianzhu.com), [Xiangyu Zhu](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/), [Yang Yang](http://www.cbsr.ia.ac.cn/users/yyang/main.htm), Fan Yang, [Zhen Lei](http://www.cbsr.ia.ac.cn/users/zlei/) and [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ).

<p align="center">
  <img src="docs/images/out.gif" alt="demo" width="512px">
</p>

## Introduction

This work extends [3DDFA](https://github.com/cleardusk/3DDFA), named **3DDFA_V2**, titled [Towards Fast, Accurate and Stable 3D Dense Face Alignment](https://guojianzhu.com/assets/pdfs/3162.pdf), accepted by [ECCV 2020](https://eccv2020.eu/). The supplementary material is [here](https://guojianzhu.com/assets/pdfs/3162-supp.pdf). The [gif](./docs/images/out.gif) above shows a demo of the tracking result.
This repo is the official implementation of 3DDFA_V2.

Compared to [3DDFA](https://github.com/cleardusk/3DDFA), 3DDFA_V2 achieves better performance and stability. Besides, 3DDFA_V2 incorporates the fast face detector [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch) instead of Dlib. A simple 3D render written by c++ and cython is also included. If you are interested in this repo, just try it! Welcome for valuable issues and PRs 😄

<!-- Currently, the pre-trained model, inference code and some utilities are released.  -->

## Getting started

### Requirements
See [requirements.txt](./requirements.txt), tested on macOS and Linux platforms. Note that this repo uses Python3. The major dependencies are PyTorch, numpy and opencv-python, etc.

### Usage

1. Clone this repo
   
```shell script
git clone https://github.com/cleardusk/3DDFA_V2.git
cd 3DDFA_V2
```

2. Build the cython version of NMS, and Sim3DR
<!-- ```shell script
cd FaceBoxes
sh ./build_cpu_nms.sh
cd ..

cd Sim3DR
sh ./build_sim3dr.sh
cd ..
```

or simply build them by -->
```shell script
sh ./build.sh
```

3. Run demos

```shell script
# running on still image
python3 demo.py -f examples/inputs/emma.jpg

# running on videos
python3 demo_video.py -f examples/inputs/videos/214.avi

# running on videos smoothly by looking ahead by `n_next` frames
python3 demo_video_smooth.py -f examples/inputs/videos/214.avi
```

The implementation of tracking is simply by alignment. If the head pose > 90° or the motion is too fast, the alignment may fail. A threshold is used to trickly check the tracking state, but it is unstable.

You can refer to [demo.ipynb](./demo.ipynb) for the step-by-step tutorial of running on the still image.

For example, running `python3 demo.py -f examples/inputs/emma.jpg -o 3d` will give the result below:

<p align="center">
  <img src="docs/images/emma_3d.jpg" alt="demo" width="640px">
</p>

## Acknowledgement

* The FaceBoxes module is modified from [FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch)

## Citation

If your work or research benefits from this repo, please cite two bibs below : )

    @inproceedings{guo2020towards,
        title =        {Towards Fast, Accurate and Stable 3D Dense Face Alignment},
        author =       {Guo, Jianzhu and Zhu, Xiangyu and Yang, Yang and Yang, Fan and Lei, Zhen and Li, Stan Z},
        booktitle =    {Proceedings of the European Conference on Computer Vision (ECCV)},
        year =         {2020}
    }

    @misc{3ddfa_cleardusk,
        author =       {Guo, Jianzhu and Zhu, Xiangyu and Lei, Zhen},
        title =        {3DDFA},
        howpublished = {\url{https://github.com/cleardusk/3DDFA}},
        year =         {2018}
    }

## Contact
**Jianzhu Guo (郭建珠)** [[Homepage](http://guojianzhu.com), [Google Scholar](https://scholar.google.com/citations?user=W8_JzNcAAAAJ&hl=en&oi=ao)]:  **jianzhu.guo@nlpr.ia.ac.cn** or **guojianzhu1994@foxmail.com**.