#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
from skimage import io
from PIL import Image

data_path = '../../data/phone/train2014/images/'

files = os.listdir(data_path)

def check_pic(path):
    try:
        Image.open(path).load()
    except:
        print('ERROR: %s' % path)
        return False
    else:
        return True

for k,img_filename in enumerate(files):

    # if img_filename.split('.')[-1].lower() == 'jpg':
    #     with open(data_path + img_filename, 'rb') as f:
    #         f.seek(-2, 2)
    #         print(f.read() == '\xff\xd9')
    # else:
    #     print(True)


    # try:
    #     io.imread(data_path + img_filename)
    # except Exception as e:
    #     print(e)

    check_pic(data_path + img_filename)

'''
ERROR: ../../data/phone/train2014/images/aug_20191027172347068788-iPhoneSE-L1-220-1.0-132-6500-31-A2.jpg
ERROR: ../../data/phone/train2014/images/aug_20191020182607227603-iPhoneX-L1-220-1.0-132-6500-31-A2.jpg
'''
