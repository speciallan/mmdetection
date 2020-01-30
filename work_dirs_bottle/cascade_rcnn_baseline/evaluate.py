#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd2 = 'python tools/coco_eval.py work_dirs_bottle/cascade_rcnn_baseline/results.bbox.json --ann ../../data/bottle/chongqing1_round1_train1_20191223/annotations_val.json --types bbox --classwise'
os.system(cmd2)
