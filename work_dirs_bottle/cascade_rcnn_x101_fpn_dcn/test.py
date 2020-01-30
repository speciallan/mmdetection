#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

checkpoint = './work_dirs_bottle/cascade_rcnn_x101_fpn_dcn/checkpoints/latest.pth'
json_path = './work_dirs_bottle/cascade_rcnn_x101_fpn_dcn/results.json'
cmd = 'python tools/test.py configs/bottle/cascade_rcnn_x101_fpn_dcn.py {} --json_out {} --eval bbox'.format(checkpoint, json_path)

result = os.system(cmd)
print('模型评估完毕{}'.format(json_path))
