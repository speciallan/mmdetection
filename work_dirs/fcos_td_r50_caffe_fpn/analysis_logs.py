#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

json_path = 'work_dirs/fcos_td_r50_caffe_fpn/checkpoints/'
out_file1 = 'work_dirs/fcos_td_r50_caffe_fpn/plot_losses.pdf'
out_file2 = 'work_dirs/fcos_td_r50_caffe_fpn/plot_ap.pdf'
keys1 = 'loss_rpn_cls loss_rpn_bbox loss_cls loss_bbox loss '
legends1 = 'rpn_cls rpn_bbox loss_cls loss_bbox loss'
keys2 = 'bbox_mAP bbox_mAP_50 bbox_mAP_s bbox_mAP_m bbox_mAP_l'
legends2 = 'mAP mAP_50 mAP_s mAP_m mAP_l'

files = os.listdir(json_path)
files = sorted(files, reverse=True)
filename = ''
for k,v in enumerate(files):
    if '.json' in v:
        filename = v
        break
json_file_path = json_path + filename

cmd = 'python tools/analyze_logs.py plot_curve {} --keys {} --out {} --legend {}'.format(json_file_path, keys1, out_file1, legends1)
cmd2 = 'python tools/analyze_logs.py plot_curve {} --keys {} --out {} --legend {}'.format(json_file_path, keys2, out_file2, legends2)

os.system(cmd)
os.system(cmd2)
print('done')
