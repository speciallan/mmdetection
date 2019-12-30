#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

checkpoint = './work_dirs/fcos_x101_32x4d_caffe_fpn_gn_2x/checkpoints/latest.pth'
json_path = './work_dirs/fcos_x101_32x4d_caffe_fpn_gn_2x/results.json'
cmd = 'python tools/test.py configs/fcos/fcos_x101_32x4d_caffe_fpn_gn_2x.py {} --json_out {} --eval bbox'.format(checkpoint, json_path)

result = os.system(cmd)
print('模型评估完毕{}'.format(json_path))
