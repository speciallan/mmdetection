#!/usr/bin/env bash

python tools/test.py configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc0712.py /home/speciallan/.cache/torch/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth --out work_dirs/faster_rcnn_r50_fpn_1x_voc0712/results.pkl