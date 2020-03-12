#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import mmcv
import numpy as np
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result

data_path = '../../data/phone/train2014/images/'
result_path = '../../data/VOCdevkit/SAR-Ship-Dataset/results/retinanet_r50_fpn_1x/'
config_file = 'configs/retinanet_r50_fpn_1x.py'
checkpoint_file = 'work_dirs/retinanet_r50_fpn_1x/checkpoints/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# cfg = mmcv.Config.fromfile(config_file)
# model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
# _ = load_checkpoint(model, checkpoint_file)

# test a single image and show the results
data_list = os.listdir(data_path)

# filename = 'Gao_ship_hh_0201608254401010021.jpg'
filelist = data_list[:10]

# img_list = []
# for filename in filelist:
#     img_path = data_path + filename
#     img = mmcv.imread(img_path)
#     img_list.append(img)
#
# img_list = np.array(img_list)
#
# for i, result in enumerate(inference_detector(model, img_list)):
#     print(result)
#     out_path = result_path + filelist[i]
#     show_result(img_list[i], result, model.CLASSES, show=False, out_file=out_path)

for filename in filelist:

    img_path = data_path + filename
    out_path = result_path + filename

    img = mmcv.imread(img_path)

    result = inference_detector(model, img)
    print(result)

    # or save the visualization results to image files
    show_result(img, result, model.CLASSES, show=False, out_file=out_path)

