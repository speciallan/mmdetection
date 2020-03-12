#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import json
from numpy.random import randint

DATASET_PATH = '../../data/phone/train2014'

origin_annos_file = 'cut_annotations.json'
gen_annos_file_train = 'cut_annotations_train.json'
gen_annos_file_val = 'cut_annotations_val.json'

with open(os.path.join(DATASET_PATH, origin_annos_file)) as f:
    json_file = json.load(f)

# 所有图片的数量： 4516
# 所有标注的数量： 6945
print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))

# print(dir(json_file), json_file.keys(), json_file['annotations'])


# ----------------------------- 分析 --------------------------

DATASET_PATH = '../../data/phone/train2014'

with open(os.path.join(DATASET_PATH, origin_annos_file)) as f:
    json_file = json.load(f)

annos = json_file['annotations']

s, m, l = 0, 0, 0
for k,v in enumerate(annos):
    x1, y1, w, h = v['bbox']
    if w * h <= 32*32:
        s += 1
    elif w * h <= 96*96:
        m += 1
    else:
        l += 1
print('s/m/l {} {} {}'.format(s, m, l))

# ----------------------------- 划分train/val --------------------------

DATASET_PATH = '../../data/phone/train2014'

with open(os.path.join(DATASET_PATH, origin_annos_file)) as f:
    json_file = json.load(f)

# 所有图片的数量： 3348
# 所有标注的数量： 5775
# print(json_file['info'], json_file['categories'])
print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))

imgs, annos = json_file['images'], json_file['annotations']

train_img_ids = []
val_img_ids = []
imgs_train, imgs_val, annos_train, annos_val = [], [], [], []

# 图片字典
imgs_dict, categories_dict = {}, {}
categories = json_file['categories']

for k,img in enumerate(imgs):
    imgs_dict[img['id']] = img

# 用于排除不存在的label
for k,cate in enumerate(categories):
    categories_dict[cate['id']] = cate

CLASSES = ['SL', 'HH', 'BHH', 'DQ', 'QJ', 'XK', 'KP', 'QS', 'LF', 'PFL', 'QP', 'M', 'BX', 'HJ',
           'FC', 'TT', 'QZ', 'GG', 'PP', 'AJ', 'HM', 'LH', 'QM', 'CCC', 'HZ', 'HZW', 'SG', 'NFC', 'LGO', 'XZF', 'ZF', 'SSS', 'JD', 'CJ', 'JJ', 'DJ', 'KT', 'KZ', 'KQ', 'YJ', 'WC', 'SH', 'RK', 'SX', 'MK', 'JG', 'HD', 'NGC', 'BQ', 'LS']

for k,v in enumerate(imgs):

    # 排除掉不完整的图片
    error_imgs = ['aug_20191027172347068788-iPhoneSE-L1-220-1.0-132-6500-31-A2.jpg',
                  'aug_20191020182607227603-iPhoneX-L1-220-1.0-132-6500-31-A2.jpg']

    if v['file_name'] in error_imgs:
        continue

    # 输出长宽
    # print(v['width'], v['height'])

    rand = randint(0,10)

    # 9:1
    if rand <= 8:
        train_img_ids.append(v['id'])
        imgs_train.append(v)
    else:
        val_img_ids.append(v['id'])
        imgs_val.append(v)

print('train/val imgs', len(train_img_ids), len(val_img_ids))

total = 0
for k,anno in enumerate(annos):

    cate_name = categories_dict[anno['category_id']]['name']
    if cate_name not in CLASSES:
        print('error in {}'.format(cate_name))
        continue

    _, _, w, h = anno['bbox']

    # if w == 0 or h == 0:
    #     continue

    # if cate_name == 'HH' and w< 20 and h < 20:
    if w< 16 and h < 16:
        total += 1
        print(anno)

    if anno['image_id'] in train_img_ids:
        annos_train.append(anno)
    elif anno['image_id'] in val_img_ids:
        annos_val.append(anno)
    else:
        print('error')
        # print('error', imgs_dict[anno['image_id']], anno)

print('train/val annos', len(annos_train), len(annos_val))
print(total)

json_file_train = json_file.copy()
json_file_val = json_file.copy()

# origin annotations washed
json_file_train['annotations'] = annos_train
json_file_train['images'] = imgs_train
with open(os.path.join(DATASET_PATH, gen_annos_file_train), 'w') as f:
    json.dump(json_file_train, f)

json_file_val['annotations'] = annos_val
json_file_val['images'] = imgs_val
with open(os.path.join(DATASET_PATH, gen_annos_file_val), 'w') as f:
    json.dump(json_file_val, f)


'''
所有图片的数量： 315
所有标注的数量： 31518
s/m/l 5504 15015 10999
train/val imgs 284 31
train/val annos 28415 3103
'''
