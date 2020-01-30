#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import json
from numpy.random import randint
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import colorsys
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)

DATASET_TRAIN_PATH = '../../data/bottle/chongqing1_round1_train1_20191223'
DATASET_TRAIN_VIS_PATH = '../../data/bottle/train_vis'
DATASET_TRAIN_VIS_PATH = '../../data/bottle/train_vis_checked'
DATASET_TEST_PATH = '../../data/bottle/chongqing1_round1_testA_20191223'

with open(os.path.join(DATASET_TRAIN_PATH, 'annotations_train.json')) as f:
    json_file_train = json.load(f)

with open(os.path.join(DATASET_TRAIN_PATH, 'annotations_val.json')) as f:
    json_file_val = json.load(f)

# 创建类别标签字典
category_dic=dict([(i['id'],i['name']) for i in json_file_train['categories']])
print(category_dic)


# 长宽比

# 数据分布


# show

imgs = json_file_train['images']
annos = json_file_train['annotations']

counts_label=dict([(i['name'],0) for i in json_file_train['categories']])
for i in json_file_train['annotations']:
    counts_label[category_dic[i['category_id']]]+=1

print(counts_label)


def plot_imgs(img_data, gap=10, path=''):
    files_name = img_data['file_name']
    img_annotations = img_data['annotations']
    n = len(img_annotations)
    boxs = np.zeros((n, 4))
    tag = []
    img = Image.open(DATASET_TRAIN_PATH + '/images/' + files_name)  # 图片路径
    img_w = img.size[0]
    img_h = img.size[1]
    for i in range(n):
        bbox = img_annotations[i]['bbox']
        tag.append(category_dic[img_annotations[i]['category_id']])
        y1 = max(0, np.floor(bbox[1] + 0.5).astype('int32'))
        x1 = max(0, np.floor(bbox[0] + 0.5).astype('int32'))
        y2 = min(img_h, np.floor(bbox[1] + bbox[3] + 0.5).astype('int32'))
        x2 = min(img_w, np.floor(bbox[0] + bbox[2] + 0.5).astype('int32'))
        boxs[i] = [x1, y1, x2, y2]

    ttf = '/usr/share/fonts/SimHei.ttf'
    font = ImageFont.truetype(font=ttf, size=np.floor(3.5e-2 * img_w).astype('int32'), encoding="unic")
    hsv_tuples = [(x / n, 1., 1.) for x in range(n)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                      colors))
    for index in range(len(boxs)):
        draw = ImageDraw.Draw(img)
        label_size = draw.textsize(tag[index], font)
        text_origin = np.array([20, 25 + index * label_size[1]])
        for i in range(gap):
            draw.rectangle(
                [boxs[index][0] + i, boxs[index][1] + i, boxs[index][2] - i, boxs[index][3] - i], outline=colors[index], width=1)
        #     draw.rectangle(list(),outline=colors[index])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                       fill=colors[index])
        draw.text(text_origin, tag[index], fill=(0, 0, 0), font=font)
    plt.imshow(img)
    plt.savefig(path)

data_dict = {}
for k,v in enumerate(imgs):
    data_dict[v['id']] = {'file_name':v['file_name'], 'annotations':[]}

for k,v in enumerate(annos):
    data_dict[v['image_id']]['annotations'].append(v)

# print(data_dict.keys())
for k,v in tqdm(data_dict.items()):
    if len(v['annotations']) < 3:
        continue
    plot_imgs(v, gap=1, path=DATASET_TRAIN_VIS_PATH + '/' + str(len(v['annotations'])) + '_' + v['file_name'])
    # print('{} done'.format(v['file_name']))
