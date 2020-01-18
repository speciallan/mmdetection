#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = 'python tools/train.py configs/bottle/fcos_td_r50_caffe_fpn.py'
result = os.system(cmd)
