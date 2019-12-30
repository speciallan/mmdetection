#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import random

origin_path = '../../../data/VOCdevkit/SAR_ship/'
h5_path = '../../../data/VOCdevkit/SAR_high_quality_datasets/h5/'
h6_path = '../../../data/VOCdevkit/SAR_high_quality_datasets/h6/'

origin_img_path = origin_path + 'JPEGImages'
h5_train = h5_path + 'train/JPEGImages'
h5_test = h5_path + 'test/JPEGImages'

origin_list = os.listdir(origin_img_path)
h5_train_list = os.listdir(h5_train)
h5_test_list = os.listdir(h5_test)

# print(origin_list)
print(len(origin_list), len(h5_train_list), len(h5_test_list))

new = []
for i in origin_list:
    if i not in h5_train_list and i not in h5_test_list:
        new.append(i)

random.shuffle(new)
print(len(new))

split = 0.8
new_train = new[:int(len(new)*split)]
new_test = new[int(len(new)*split):]
print(len(new_train), len(new_test))

h5_train_list.extend(new_train)
h5_test_list.extend(new_test)


print(len(h5_train_list), len(h5_test_list))

split = 7.0/8

train_list = h5_train_list[:int(len(h5_train_list)*split)]
val_list = h5_train_list[int(len(h5_train_list)*split):]
test_list = h5_test_list

print(len(train_list), len(val_list), len(test_list))
print('----------')

# 使用全数据集
new = origin_list
split = 0.9

new_train = new[:int(len(new)*split)]
new_test = new[int(len(new)*split):]
print(len(new_train), len(new_test))

split = 7.0/9

train_list = new_train[:int(len(new_train)*split)]
val_list = new_train[int(len(new_train)*split):]
test_list = new_test
print(len(train_list), len(val_list), len(test_list))

f = open(h6_path+'train.txt', 'w')
for i in train_list:
    i = i.replace('.jpg', '')
    f.write(i+'\n')
f.close()

f = open(h6_path+'./valid.txt', 'w')
for i in val_list:
    i = i.replace('.jpg', '')
    f.write(i+'\n')
f.close()

f = open(h6_path+'./test.txt', 'w')
for i in test_list:
    i = i.replace('.jpg', '')
    f.write(i+'\n')
f.close()
