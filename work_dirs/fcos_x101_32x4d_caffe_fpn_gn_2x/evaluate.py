#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd2 = 'python tools/coco_eval.py work_dirs/fcos_x101_32x4d_caffe_fpn_gn_2x/results.bbox.json --ann ../../data/VOCdevkit/SAR-Ship-Dataset/ImageSets/Main/instances_val2017.json --types bbox --classwise'
os.system(cmd2)
