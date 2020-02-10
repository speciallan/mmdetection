#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = './tools/dist_train.sh configs/bottle/cascade_rcnn_r50_fpn.py 1 --validate'
result = os.system(cmd)
