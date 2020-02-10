#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = './tools/dist_train.sh configs/bottle/reppoints_moment_r101_dcn_fpn_mt.py 1 --validate'
result = os.system(cmd)
