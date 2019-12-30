#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = 'python tools/train.py configs/retinanet_x101_32x4d_fpn_1x.py'
result = os.system(cmd)
