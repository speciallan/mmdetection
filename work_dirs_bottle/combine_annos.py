#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import json

DATASET_PATH_TRAIN = '/Users/speciallan/Downloads/chongqing1_round1_train1_20191223/'
DATASET_PATH_TEST = '/Users/speciallan/Downloads/chongqing1_round1_testA_20191223/'

with open(os.path.join(DATASET_PATH_TRAIN, 'annotations_washed.json')) as f:
    json_file = json.load(f)

with open(os.path.join(DATASET_PATH_TEST, 'annotations_checked.json')) as f:
    json_file_cked = json.load(f)

# 所有图片的数量： 3348
# 所有标注的数量： 5775
# print(json_file['info'], json_file['categories'])
print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))
print('所有图片的数量：', len(json_file_cked['images']))
print('所有标注的数量：', len(json_file_cked['annotations']))

imgs_train = json_file['images']
annos_train = json_file['annotations']
imgs_test = json_file_cked['images']
annos_test = json_file_cked['annotations']
