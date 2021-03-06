#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import json
from numpy.random import randint

DATASET_PATH = '../../data/bottle/chongqing1_round1_train1_20191223'

with open(os.path.join(DATASET_PATH, 'annotations.json')) as f:
    json_file = json.load(f)

# 所有图片的数量： 4516
# 所有标注的数量： 6945
print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))

# print(dir(json_file), json_file.keys(), json_file['annotations'])


bg_imgs = set() # 所有标注中包含背景的图片 id
for c in json_file['annotations']:
    if c['category_id'] == 0:
        bg_imgs.add(c['image_id'])
    print('所有标注中包含背景的图片数量：', len(bg_imgs))

bg_only_imgs = set()  # 只有背景的图片的 id
for img_id in bg_imgs:
    co = 0
    for c in json_file['annotations']:
        if c['image_id'] == img_id:
            co += 1
        if co == 1:
            bg_only_imgs.add(img_id)

print('只包含背景的图片数量：', len(bg_only_imgs))

images_to_be_deleted = []
for img in json_file['images']:
    if img['id'] in bg_only_imgs:
        images_to_be_deleted.append(img)
    # 删除的是只有一个标注，且为 background 的的图片

print('待删除图片的数量：', len(images_to_be_deleted))

for img in images_to_be_deleted:
    json_file['images'].remove(img)

print('处理之后图片的数量：', len(json_file['images']))

# 删除所有关于背景的标注
ann_to_be_deleted = []
for c in json_file['annotations']:
    if c['category_id'] == 0:
        ann_to_be_deleted.append(c)

print('待删除标注的数量：', len(ann_to_be_deleted))

for img in ann_to_be_deleted:
    json_file['annotations'].remove(img)
print('处理之后标注的数量：', len(json_file['annotations']))

# 删除 categories 中关于背景的部分
bg_cate = {'supercategory': '背景', 'id': 0, 'name': '背景'}
json_file['categories'].remove(bg_cate)
print(json_file['categories'])

# 标注的 id 有重复的，这里重新标号
for idx in range(len(json_file['annotations'])):
    json_file['annotations'][idx]['id'] = idx
    json_file['annotations'][idx]['segmentation'] = []

# 所有图片的数量： 3348
# 所有标注的数量： 5775
print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))

# print(json_file['annotations'][0])
# exit()

# 分析图片数量
json_file_cap = json_file_glass = json_file_train = json_file_val = json_file_train_cked = json_file_val_cked =  json_file
imgs = json_file['images']
annos = json_file['annotations']
imgs_cap = []
imgs_glass = []
imgs_train, imgs_val = [], []
annos_cap = []
annos_glass = []
annos_train, annos_val = [], []

cap_num, glass_num = 0, 0
imgs_dict = {}
for k,img in enumerate(imgs):
    imgs_dict[img['id']] = img
    # if img['id'] > 200 and img['id'] < 300:
    #     print(img['id'])

for k,anno in enumerate(annos):
    # print(imgs_dict[anno['image_id']], anno)
    if anno['image_id'] not in imgs_dict.keys():
        continue
    img = imgs_dict[anno['image_id']]

    if img['height'] == 492:
        cap_num += 1
        imgs_cap.append(img)
        annos_cap.append(anno)
    elif img['height'] == 3000:
        glass_num += 1
        imgs_glass.append(img)
        annos_glass.append(anno)
    else:
        print('there are other sizes')

    rand = randint(0,10)
    if rand <= 8:
        imgs_train.append(img)
        annos_train.append(anno)
    else:
        imgs_val.append(img)
        annos_val.append(anno)

# print(cap_num, glass_num)
# exit()

# 导出数据
# with open(os.path.join(DATASET_PATH, 'annotations_washed.json'), 'w') as f:
#     json.dump(json_file, f)
#
# json_file_cap['annotations'] = annos_cap
# json_file_cap['images'] = imgs_cap
#
# with open(os.path.join(DATASET_PATH, 'annotations_cap.json'), 'w') as f:
#     json.dump(json_file_cap, f)
#
# json_file_glass['annotations'] = annos_glass
# json_file_glass['images'] = imgs_glass
#
# with open(os.path.join(DATASET_PATH, 'annotations_glass.json'), 'w') as f:
#     json.dump(json_file_glass, f)

# ----------------------------- 分析 --------------------------

DATASET_PATH = '../../data/bottle/chongqing1_round1_train1_20191223'

with open(os.path.join(DATASET_PATH, 'annotations_washed.json')) as f:
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

exit()

# ----------------------------- 划分train/val --------------------------

DATASET_PATH = '../../data/bottle/chongqing1_round1_train1_20191223'

with open(os.path.join(DATASET_PATH, 'annotations_washed.json')) as f:
    json_file = json.load(f)

with open(os.path.join(DATASET_PATH, 'annotations_checked.json')) as f:
    json_file_cked = json.load(f)

# 所有图片的数量： 3348
# 所有标注的数量： 5775
# print(json_file['info'], json_file['categories'])
print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))

imgs, annos = json_file['images'], json_file['annotations']
imgs_cked, annos_cked = json_file_cked['images'], json_file_cked['annotations']

train_img_ids = []
val_img_ids = []
imgs_train, imgs_val, annos_train, annos_val = [], [], [], []
imgs_train_cked, imgs_val_cked, annos_train_cked, annos_val_cked = [], [], [], []

# 图片字典
imgs_dict = {}
for k,img in enumerate(imgs):
    imgs_dict[img['id']] = img

for k,(v,v_cked) in enumerate(zip(imgs, imgs_cked)):
    rand = randint(0,10)
    # 9:1
    if rand <= 8:
        train_img_ids.append(v['id'])
        imgs_train.append(v)
        imgs_train_cked.append(v_cked)
    else:
        val_img_ids.append(v['id'])
        imgs_val.append(v)
        imgs_val_cked.append(v_cked)

print('train/val imgs', len(train_img_ids), len(val_img_ids))

for k,(anno, anno_cked) in enumerate(zip(annos, annos_cked)):
    if anno['image_id'] in train_img_ids:
        annos_train.append(anno)
        annos_train_cked.append(anno_cked)
    elif anno['image_id'] in val_img_ids:
        annos_val.append(anno)
        annos_val_cked.append(anno_cked)
    else:
        print('error')
        # print('error', imgs_dict[anno['image_id']], anno)

print('train/val annos', len(annos_train), len(annos_val))

# origin annotations washed
json_file_train['annotations'] = annos_train
json_file_train['images'] = imgs_train
with open(os.path.join(DATASET_PATH, 'annotations_train.json'), 'w') as f:
    json.dump(json_file_train, f)

json_file_val['annotations'] = annos_val
json_file_val['images'] = imgs_val
with open(os.path.join(DATASET_PATH, 'annotations_val.json'), 'w') as f:
    json.dump(json_file_val, f)

# checked annotations washed
json_file_train_cked['annotations'] = annos_train_cked
json_file_train_cked['images'] = imgs_train_cked
with open(os.path.join(DATASET_PATH, 'annotations_checked_train.json'), 'w') as f:
    json.dump(json_file_train_cked, f)

json_file_val_cked['annotations'] = annos_val_cked
json_file_val_cked['images'] = imgs_val_cked
with open(os.path.join(DATASET_PATH, 'annotations_checked_val.json'), 'w') as f:
    json.dump(json_file_val_cked, f)

'''
{'瓶盖破损': 1478, '喷码正常': 442, '瓶盖断点': 567, '瓶盖坏边': 583, '瓶盖打旋': 428, '瓶盖变形': 632, '标贴气泡': 399, '标贴歪斜': 169, '喷码异常': 183, '标贴起皱': 357}
'''

