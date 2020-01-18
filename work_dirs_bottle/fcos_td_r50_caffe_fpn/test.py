#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

checkpoint = './work_dirs_bottle/fcos_td_r50_caffe_fpn/checkpoints/latest.pth'
json_path = './work_dirs_bottle/fcos_td_r50_caffe_fpn/results.json'
cmd = 'python tools/test.py configs/bottle/fcos_td_r50_caffe_fpn.py {} --json_out {} --eval bbox'.format(checkpoint, json_path)

result = os.system(cmd)
print('模型评估完毕{}'.format(json_path))
