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
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)
font = cv2.FONT_HERSHEY_SIMPLEX

DATASET_TRAIN_PATH = '../../data/phone/train2014'
DATASET_TRAIN_VIS_PATH = '../../data/phone/train2014/train_vis'
DATASET_TEST_PATH = '../../data/phone/train2014'

with open(os.path.join(DATASET_TRAIN_PATH, 'annotations_train.json')) as f:
    json_file_train = json.load(f)

# with open(os.path.join(DATASET_TRAIN_PATH, 'annotations_val.json')) as f:
#     json_file_val = json.load(f)

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

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):

    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)

    # 字体格式
    ttf = '/usr/share/fonts/SimHei.ttf'
    font = ImageFont.truetype(font=ttf, size=textSize, encoding="unic")

    # 绘制文本
    draw.text((left, top), text, textColor, font=font)

    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def plot_imgs(img_data, gap=10, path=''):
    files_name = img_data['file_name']
    img_annotations = img_data['annotations']
    n = len(img_annotations)
    img_path = DATASET_TRAIN_PATH + '/images/' + files_name  # 图片路径
    img = cv2.imread(img_path)
    img_h, img_w, _ = img.shape

    for k, gt_box in enumerate(img_annotations):
        # print(gt_box['bbox'])
        x1, y1, ww, hh = gt_box['bbox']
        x1, y1, ww, hh = int(x1), int(y1), int(ww), int(hh)

        # img = cv2ImgAddText(img, '{}'.format(category_dic[gt_box['category_id']]), x1+2, y1-12, (0, 255, 0), textSize=12)
        img = cv2.putText(img, '{}'.format(category_dic[gt_box['category_id']]), (x1+2, y1-4), font, 0.5, (0, 255, 0), 1)
        img = cv2.rectangle(img, (x1, y1), (x1+ww, y1+hh), (0, 255, 0), 1)

    cv2.imwrite(path, img)

    del img

data_dict = {}
for k,v in enumerate(imgs):
    data_dict[v['id']] = {'file_name':v['file_name'], 'annotations':[]}

for k,v in enumerate(annos):
    data_dict[v['image_id']]['annotations'].append(v)

exist_out = os.listdir(DATASET_TRAIN_VIS_PATH)

# print(data_dict.keys())
for k,v in tqdm(data_dict.items()):

    # if v['file_name'] in exist_out:
    #     continue

    # if len(v['annotations']) < 3:
    #     continue

    plot_imgs(v, gap=1, path=DATASET_TRAIN_VIS_PATH + '/' + str(len(v['annotations'])) + '_' + v['file_name'])
    # print('{} done'.format(v['file_name']))
