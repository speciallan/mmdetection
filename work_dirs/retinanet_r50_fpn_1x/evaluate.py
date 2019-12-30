#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

# cmd1 = 'python tools/voc_eval.py work_dirs/ssd300_sar/results.pkl configs/pascal_voc/ssd300_sar.py'
# os.system(cmd1)

cmd2 = 'python tools/coco_eval.py work_dirs/retinanet_r50_fpn_1x/results.bbox.json --ann ../../data/VOCdevkit/SAR-Ship-Dataset/ImageSets/Main/instances_val2017.json --types bbox --classwise'
os.system(cmd2)
