#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

checkpoint = './work_dirs/fovea_r50_fpn_4gpu_1x/checkpoints/latest.pth'
json_path = './work_dirs/fovea_r50_fpn_4gpu_1x/results.json'
cmd = 'python tools/test.py configs/foveabox/fovea_r50_fpn_4gpu_1x.py {} --json_out {} --eval bbox'.format(checkpoint, json_path)

result = os.system(cmd)
print('模型评估完毕{}'.format(json_path))
