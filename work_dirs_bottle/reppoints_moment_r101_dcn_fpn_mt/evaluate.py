#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd2 = 'python tools/coco_eval.py work_dirs_bottle/reppoints_moment_r101_dcn_fpn_mt/results.bbox.json --ann ../../data/bottle/chongqing1_round1_train1_20191223/annotations_val.json --types bbox --classwise'
os.system(cmd2)
