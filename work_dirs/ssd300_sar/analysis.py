#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd2 = 'python tools/coco_error_analysis.py work_dirs/ssd300_sar/results.bbox.json ../../data/VOCdevkit/SAR-Ship-Dataset/analysis --ann ../../data/VOCdevkit/SAR-Ship-Dataset/ImageSets/Main/instances_val2017.json --types bbox'

os.system(cmd2)
