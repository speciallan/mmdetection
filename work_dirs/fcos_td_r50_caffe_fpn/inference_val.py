#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

import os
import mmcv
import json
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import init_detector, inference_detector, show_result
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw

from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)
font = cv2.FONT_HERSHEY_SIMPLEX
# height, width = 492, 658

data_path = '../../data/VOCdevkit/SAR-Ship-Dataset/JPEGImages/'
anno_path = '../../data/VOCdevkit/SAR-Ship-Dataset/ImageSets/Main/instances_val2017.json'
result_path = '../../data/VOCdevkit/SAR-Ship-Dataset/results/fcos_td_r50_caffe_fpn/'
config_file = 'configs/my/fcos_td_r50_caffe_fpn.py'
checkpoint_file = 'work_dirs/fcos_td_r50_caffe_fpn/checkpoints/latest.pth'


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

# 获取anno
with open(anno_path) as f:
    json_file_val = json.load(f)

imgs = json_file_val['images']
annos = json_file_val['annotations']

data_dict, data_dict_filename = {}, {}
for k,v in enumerate(imgs):
    data_dict[v['id']] = {'file_name':v['file_name'], 'annotations':[]}

for k,v in enumerate(annos):
    data_dict[v['image_id']]['annotations'].append(v)

# for k,v in data_dict.items():
#     data_dict_filename[v['file_name']] = v

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
classes = model.CLASSES
category_dic = dict([(i['id'],i['name']) for i in json_file_val['categories']])
index = {1: 1, 2: 9, 3: 5, 4: 3, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}

# cfg = mmcv.Config.fromfile(config_file)
# model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
# _ = load_checkpoint(model, checkpoint_file)

# test a single image and show the results
# data_list = os.listdir(data_path)

# filename = 'Gao_ship_hh_0201608254401010021.jpg'
# filelist = data_list[:10]

total_gt, total_predict = 0, 0

for idx, gt in tqdm(data_dict.items()):

    filename = gt['file_name']

    # arr = ['newship0503504.jpg', 'newship090301.jpg']
    # if filename.replace('.jpg', '').replace('img_', '') not in arr:
    #     continue

    img_path = data_path + filename
    out_path = result_path + filename

    img = cv2.imread(img_path)
    h, w, _ = img.shape
    gt_img = show_img = img.copy()

    result = inference_detector(model, img)
    # print(result)
    # x1,y1,x2,y2,score

    # test
    # if len(result) >= len(gt['annotations']):
    #     continue

    # 画gt
    if len(gt['annotations']) != 0:
        gt_boxes = gt['annotations']
        for k, gt_box in enumerate(gt_boxes):
                # print(gt_box['bbox'])
                x1, y1, ww, hh = gt_box['bbox']
                x1, y1, ww, hh = int(x1), int(y1), int(ww), int(hh)

                gt_img = cv2ImgAddText(gt_img, '{}'.format(category_dic[gt_box['category_id']]), x1+2, y1-12, (0, 255, 0), textSize=12)
                gt_img = cv2.rectangle(gt_img, (x1, y1), (x1+ww, y1+hh), (0, 255, 0), 1)

        total_gt += len(gt_boxes)

    # 画result
    for classid, boxes in enumerate(result, 1):

        if len(boxes) == 0:
            continue

        # 单个类别预测分析用
        # arr = [8]
        # if 'new' in anno_path and index[classid] not in arr:
        #     continue

        # print(classid, boxes)

        for k,v in enumerate(boxes):

            if len(v) == 0:
                continue

            x1, y1, x2, y2, score = v
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # show_img = cv2.putText(show_img, '{} {:.2f}'.format(classes[classid], score), (x1 - 2, y1 - 2), font, 0.5, (0, 255, 0), 1)
            show_img = cv2ImgAddText(show_img, '{} {:.4f}'.format(category_dic[index[classid]], score), x1+2, y1-12, (0, 255 , 0), textSize=12)
            show_img = cv2.rectangle(show_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        total_predict += len(boxes)

    combine = np.array(np.zeros((h, w*2, 3)))
    combine[:, 0:w, :] = gt_img
    combine[:, w:w*2, :] = show_img
    # combine[:, 256*2:256*3, :] = results_img
    # combine = cv2.vconcat(origin_img, show_img)
    cv2.imwrite(out_path, combine)

print(total_gt, total_predict)

    # or save the visualization results to image files
    # show_result(img, result, model.CLASSES, show=False, out_file=out_path)

    # print('{} 预测完毕'.format(filename))



