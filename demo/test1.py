#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

#!/usr/bin/python3
#!--*-- coding:utf-8 --*--
import os
from mmdet.apis import init_detector, inference_detector, show_result

# 首先下载模型文件https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth
config_file = 'configs/faster_rcnn_r50_fpn_1x.py'
checkpoint_file = '/home/speciallan/.cache/torch/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file)

# 测试一张图片
img = 'test.jpg'
img = 'demo/000000015254.jpg'
result = inference_detector(model, img)
# print(result)
show_result(img, result, model.CLASSES, score_thr=0.3, show=False, out_file='./demo/save.jpg')

# 测试一系列图片
# imgs = ['test1.jpg', 'test2.jpg']
# for i, result in enumerate(inference_detector(model, imgs, device='cuda:0')):
#     show_result(imgs[i], result, model.CLASSES, out_file='result_{}.jpg'.format(i))