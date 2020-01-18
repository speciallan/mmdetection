#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

config = 'configs/bottle/fcos_r50_caffe_fpn.py'
checkpoint = './work_dirs_bottle/fcos_r50_caffe_fpn/checkpoints/latest.pth'
img_path = '../../data/bottle/chongqing1_round1_testA_20191223/images'
out = './work_dirs_bottle/fcos_r50_caffe_fpn/results_out.json'
cmd = 'python ./work_dirs_bottle/gen_results.py -c {} -m {} -im {} -o {}'.format(config, checkpoint, img_path, out)

result = os.system(cmd)
print('生成结果完毕{}'.format(out))
