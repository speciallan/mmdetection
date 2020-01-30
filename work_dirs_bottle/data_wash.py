#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import json

DATASET_PATH = '../../data/bottle/chongqing1_round1_train1_20191223'

with open(os.path.join(DATASET_PATH, 'annotations.json')) as f:
    json_file = json.load(f)

# 所有图片的数量： 4516
# 所有标注的数量： 6945
print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))

# print(dir(json_file), json_file.keys(), json_file['annotations'])

annos = json_file['annotations']
# for k,v in enumerate(annos):
#     print(v)
#     exit()

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

# 删除只有背景标注的图片
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

# 导出数据
with open(os.path.join(DATASET_PATH, 'annotations_washed.json'), 'w') as f:
    json.dump(json_file, f)


# 所有图片的数量： 3348
# 所有标注的数量： 5775
print(json_file['info'], json_file['categories'])
print('所有图片的数量：', len(json_file['images']))
print('所有标注的数量：', len(json_file['annotations']))


