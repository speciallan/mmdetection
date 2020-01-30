#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

config = 'configs/bottle/cascade_rcnn_x101_fpn_dcn.py'
checkpoint = './work_dirs_bottle/cascade_rcnn_x101_fpn_dcn/checkpoints/latest.pth'
img_path = '../../data/bottle/chongqing1_round1_testA_20191223/images'
out = './work_dirs_bottle/cascade_rcnn_x101_fpn_dcn/results_out.json'
cmd = 'python ./work_dirs_bottle/gen_results.py -c {} -m {} -im {} -o {}'.format(config, checkpoint, img_path, out)

result = os.system(cmd)
print('生成结果完毕{}'.format(out))
