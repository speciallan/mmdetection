#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = './tools/dist_train.sh configs/my/fcos_td_r50_caffe_fpn.py 1 --validate'
result = os.system(cmd)
