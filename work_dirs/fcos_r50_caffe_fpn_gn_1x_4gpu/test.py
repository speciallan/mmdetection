#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

checkpoint = './work_dirs/fcos_r50_caffe_fpn_gn_1x_4gpu/checkpoints/latest.pth'
json_path = './work_dirs/fcos_r50_caffe_fpn_gn_1x_4gpu/results.json'
cmd = 'python tools/test.py configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py {} --json_out {} --eval bbox'.format(checkpoint, json_path)

result = os.system(cmd)
print('模型评估完毕{}'.format(json_path))
