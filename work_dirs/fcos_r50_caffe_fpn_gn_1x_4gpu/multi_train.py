#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = './tools/dist_train.sh configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py 1 --validate'
result = os.system(cmd)
