#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

checkpoint = './work_dirs/ssd300_sar/checkpoints/latest.pth'
# pkl_path = './work_dirs/ssd300_sar/results.pkl'
json_path = './work_dirs/ssd300_sar/results.json'
# cmd = 'python tools/test.py configs/pascal_voc/ssd300_sar.py {} --out {}'.format(checkpoint, pkl_path)
cmd = 'python tools/test.py configs/pascal_voc/ssd300_sar.py {} --json_out {} --eval bbox'.format(checkpoint, json_path)

result = os.system(cmd)
print('模型评估完毕{}'.format(json_path))
