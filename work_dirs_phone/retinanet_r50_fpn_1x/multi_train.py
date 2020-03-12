#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

cmd = './tools/dist_train.sh configs/phone/retinanet_r50_fpn_1x.py 1 --validate'
result = os.system(cmd)
