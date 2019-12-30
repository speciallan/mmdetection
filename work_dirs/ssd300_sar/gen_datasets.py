#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os

anno_path = '../../data/VOCdevkit/SAR-Ship-Dataset/dataset_voc'
main_path = '../../data/VOCdevkit/SAR-Ship-Dataset/ImageSets/Main'

train_list = os.listdir(anno_path + '/train/Annotations')
test_list = os.listdir(anno_path + '/test/Annotations')
print(len(train_list), len(test_list))

with open(main_path + '/trainval.txt', 'w') as f:
    for name in train_list:
        if '.xml' in name:
            f.write(name.replace('.xml', '') + '\n')

with open(main_path + '/test.txt', 'w') as f:
    for name in test_list:
        if '.xml' in name:
            f.write(name.replace('.xml', '') + '\n')
