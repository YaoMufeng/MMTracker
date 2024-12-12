# MMTracker

## Abstract
Codes for MMTracker:Motion Mamba with Margin Loss for UAV-platform Multiple Object Tracking (Accepted by AAAI2025)


## Installation
pip3 install -r requirements.txt
python3 setup.py develop


If your cuda version is not 11.3, maybe you should install pytorch from official site (https://pytorch.org/get-started/previous-versions) first.
Pytorch version and cuda version is flexible.
This code runs successfully in both Pytorch 1.XX and 2.XX
However, Mamba's installation is not easy. Installing mamba_ssm != 1.2.2 may cause many problems in this code, which is the only version I can successfully runs in my machine.



datasets
   |——————Visdrone
   |        └——————VisDrone2019-MOT-train
                    └——————sequences - *.jpgs
                    └——————annotations
                    └——————flow (extracted by pretrained optical-flow net) *.flo
                    └——————ignore_mask (optional)
   |        └——————VisDrone2019-MOT-val
   |        └——————VisDrone2019-MOT-test-dev
   |        └——————annotations (put converted coco label here)
```

Get COCO format  training data:

## conventional training
python ./tools/convert_visdrone_to_coco.py


## motion margin loss requires extral field in json files:
python ./tools/convert_visdrone_to_coco_flow.py


## extract flow
 I use (https://github.com/gddcx/EMD-Flow), other flow estimator is ok. Make sure flow prediction is put on the same folder as "sequences"

## merge flow with gt:
python ./tools/box_offset_over_flow.py

## Step1:Training with motion margin loss:
source scripts/train_visdrone_mot.sh

## Step2:Based on step1, training the motion model:
source scripts/motion_visdrone_train.sh

## Eval:

source scripts/motion_visdrone_test.sh
make sure the weights path exists.

## pretrained Weights:
Baidu Net Disk:
https://pan.baidu.com/s/1p6Ep-75NoD9E9S_NZ0B-FA?pwd=wviz 
提取码: wviz



## Maybe training this code is a little bit troublesome, if you just want see how MMLoss and Motion Mamba works,
you can refer to ./yolox/models/losses.py and ./motion/motion_model.py
## Citation

```
@misc{yao2024mmtrackermotionmambamargin,
      title={MM-Tracker: Motion Mamba with Margin Loss for UAV-platform Multiple Object Tracking}, 
      author={Mufeng Yao and Jinlong Peng and Qingdong He and Bo Peng and Hao Chen and Mingmin Chi and Chao Liu and Jon Atli Benediktsson},
      year={2024},
      eprint={2407.10485},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.10485}, 
}
```

## Acknowledgement

A large part of the code is borrowed from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX), 
[ByteTrack](https://github.com/ifzhang/ByteTrack),

Many thanks for their wonderful works.
