#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

checkpoint = './work_dirs/ssd512_coco/checkpoints/latest.pth'
json_path = './work_dirs/ssd512_coco/results.json'
cmd = 'python tools/test.py configs/ssd512_coco.py {} --json_out {} --eval bbox'.format(checkpoint, json_path)

result = os.system(cmd)
print('模型评估完毕{}'.format(json_path))
