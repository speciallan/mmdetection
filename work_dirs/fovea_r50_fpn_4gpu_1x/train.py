#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = 'python tools/train.py configs/foveabox/fovea_r50_fpn_4gpu_1x.py'
result = os.system(cmd)
