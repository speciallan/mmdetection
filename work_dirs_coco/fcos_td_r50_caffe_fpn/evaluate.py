#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd2 = 'python tools/coco_eval.py work_dirs_coco/fcos_td_r50_caffe_fpn/results.bbox.json --ann ../../data/coco/annotations/instances_val2017.json --types bbox --classwise'
os.system(cmd2)
