#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = 'python tools/train.py configs/fcos/fcos_x101_32x4d_caffe_fpn_gn_2x.py'
result = os.system(cmd)
