#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd2 = 'python tools/coco_eval.py work_dirs/ssd512_coco/results.bbox.json --ann ../../data/VOCdevkit/SAR-Ship-Dataset/ImageSets/Main/instances_val2017.json --types bbox --classwise'
os.system(cmd2)
