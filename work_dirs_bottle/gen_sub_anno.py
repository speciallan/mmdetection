#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import json
from numpy.random import randint

'''
类别序号	类别名称	类别权重 标签数量 总得分
1	瓶盖破损	0.15 1619 242.85 *
2	瓶盖变形	0.09 705 63.45 *
3	瓶盖坏边	0.09 656 59.04
4	瓶盖打旋	0.05 480 24
5	瓶盖断点	0.13 614 79.82 *
6	标贴歪斜	0.05 186 9.3
7	标贴起皱	0.12 384 46.08
8	标贴气泡	0.13 443 57.59 *
9	正常喷码	0.07 489 34.23
10	异常喷码	0.12 199 23.88
CLASSES = ('瓶盖破损','瓶盖变形','瓶盖坏边','瓶盖打旋','瓶盖断点','标贴歪斜','标贴起皱','标贴气泡','喷码正常','喷码异常')
index = {1: 1, 2: 9, 3: 5, 4: 3, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
'''

DATASET_PATH = '../../data/bottle/chongqing1_round1_train1_20191223'

with open(os.path.join(DATASET_PATH, 'annotations_val.json')) as f:
    json_file = json.load(f)

print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))

imgs = json_file['images']
annos = json_file['annotations']

cate_ids = [8]
img_ids = []
imgs_new, annos_new = [], []
json_file_new = json_file
gt_bbox = []
total_w, total_h = 0, 0

imgs_dict = {}
for k,img in enumerate(imgs):
    imgs_dict[img['id']] = img

# 只把该类目标加入到anno
for k,anno in enumerate(annos):
    if anno['category_id'] in cate_ids:
        img_ids.append(anno['image_id'])
        annos_new.append(anno)
        w, h = anno['bbox'][2], anno['bbox'][3]
        gt_bbox.append([w, h])
        total_h += h
        total_w += w
        print(imgs_dict[anno['image_id']]['file_name'], int(w), int(h), round(w/h, 2))

for k, img in enumerate(imgs):
    if img['id'] in img_ids:
        imgs_new.append(img)

print('avg w/h', total_w/len(annos_new), total_h/len(annos_new))
print('new imgs/annos', len(imgs_new), len(annos_new))

# origin annotations washed
json_file_new['annotations'] = annos_new
json_file_new['images'] = imgs_new
with open(os.path.join(DATASET_PATH, 'annotations_val_new.json'), 'w') as f:
    json.dump(json_file_new, f)
