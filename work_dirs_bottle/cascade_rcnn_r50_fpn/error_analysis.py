#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

anno = '../../data/bottle/chongqing1_round1_train1_20191223/annotations_val.json'
result = 'work_dirs_bottle/cascade_rcnn_r50_fpn/results.bbox.json'
out_dir = 'work_dirs_bottle/cascade_rcnn_r50_fpn/analysis/'
cmd = 'python tools/coco_error_analysis.py {} {} --ann {}'.format(result, out_dir, anno)
os.system(cmd)
print('done')
