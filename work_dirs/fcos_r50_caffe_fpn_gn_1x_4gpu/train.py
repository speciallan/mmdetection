#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = 'python tools/train.py configs/fcos/fcos_r50_caffe_fpn_gn_1x_4gpu.py'
result = os.system(cmd)
