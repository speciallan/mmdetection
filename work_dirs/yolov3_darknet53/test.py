#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

checkpoint = './work_dirs/yolov3_darknet53/checkpoints/latest.pth'
json_path = './work_dirs/yolov3_darknet53/results.json'
cmd = 'python tools/test.py configs/yolov3_darknet53.py {} --json_out {} --eval bbox'.format(checkpoint, json_path)

result = os.system(cmd)
print('模型评估完毕{}'.format(json_path))
