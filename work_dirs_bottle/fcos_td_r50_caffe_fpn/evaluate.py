#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd2 = 'python tools/coco_eval.py work_dirs_bottle/fcos_td_r50_caffe_fpn/results.bbox.json --ann ../../data/bottle/chongqing1_round1_train1_20191223/annotations_washed.json --types bbox --classwise'
os.system(cmd2)
