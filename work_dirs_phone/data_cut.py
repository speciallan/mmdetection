#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from __future__ import division, absolute_import
import os
import cv2
import numpy as np
import xml.etree.cElementTree as ET
from xml.dom import minidom
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family']='sans-serif'
plt.rcParams['figure.figsize'] = (10.0, 10.0)
font = cv2.FONT_HERSHEY_SIMPLEX

def load_annotations(anno_path, img_path):
    et = ET.parse(anno_path)
    element = et.getroot()

    # 解析基础图片数据
    element_objs = element.findall('object')
    element_filename = element.find('filename').text
    element_filename = element_filename + '.jpg' if '.jpg' not in element_filename else element_filename
    element_width = int(element.find('size').find('width').text)
    element_height = int(element.find('size').find('height').text)

    annotation_data = {}

    # 如果有检测目标，解析目标数据
    if len(element_objs) > 0:
        annotation_data = {'filename': element_filename,
                           'filepath': os.path.join(img_path, element_filename),
                           'width': element_width,
                           'height': element_height,
                           'bboxes': []}

    # 加入类别映射
    for element_obj in element_objs:

        class_name = element_obj.find('name').text
        obj_bbox = element_obj.find('bndbox')

        # voc的坐标格式
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))
        # difficulty = int(element_obj.find('difficult').text) == 1

        annotation_data['bboxes'].append(
            {'name': class_name,
             'x1': x1, 'x2': x2,
             'y1': y1, 'y2': y2,
             # 'difficult': difficulty
             })

    return annotation_data

#保存xml格式的
def save_annotations(anno_path, filename, image, boxes):

    #保存画框图像
    # for i in range(len(boxes)):
    #     cv2.rectangle(image,
    #                   (boxes[i]['xmin'],#xmin
    #                    boxes[i]['ymin']), #ymin
    #                   (boxes[i]['xmax'],#xmax
    #                    boxes[i]['ymax']), #ymax
    #                   (0,255,0),
    #                   1)
    # cv2.imwrite(os.path.join(os.path.join(filepath, "DrawBoxes"), filename + ".jpg"), image)

    #保存labels
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = filename + ".jpg"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image.shape[0])
    ET.SubElement(size, "height").text = str(image.shape[1])
    ET.SubElement(size, "depth").text = str(image.shape[2])
    for box in boxes:
        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = box["name"]
        ET.SubElement(object, "pose").text = "Unspecified"
        ET.SubElement(object, "truncated").text = "0"
        ET.SubElement(object, "difficult").text = "0"
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(box["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(box["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(box["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(box["ymax"])

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    myfile = open(os.path.join(anno_path, filename + ".xml"), "w")
    myfile.write(xmlstr)


if __name__ == '__main__':

    img_path = "../../data/phone/train2014/images/"
    cut_img_path = "../../data/phone/train2014/cut_images/"
    anno_path = "../../data/phone/train2014/xml/"
    cut_anno_path = "../../data/phone/train2014/cut_xml/"

    cut_width = 512
    stride = 256

    if not os.path.exists(cut_img_path):
        os.mkdir(cut_img_path)

    if not os.path.exists(cut_anno_path):
        os.mkdir(cut_anno_path)

    files = os.listdir(img_path)

    for filename in files:

        filename = filename[:-4]
        ext = '.jpg'

        # if filename not in ['aug_20191016205306031849-iPhone6S-L1-220-1.0-12-6500-31-A2']:
        #     continue

        img = cv2.imread(img_path + filename + ext)
        h, w, _ = img.shape
        print(w,h)

        finish_x, finish_y = False, False
        total, total_labels, used_labels = 0, 0, 0

        anno = load_annotations(anno_path+filename+'.xml', img_path)
        bboxes = anno['bboxes']
        total_labels += len(bboxes)

        for y in range(0, h, stride):

            if h-y < cut_width:
                y = h-cut_width
                finish_y = True

            for x in range(0, w, stride):

                # x最后一个
                if w-x < cut_width:
                    x = w-cut_width
                    finish_x = True

                # 切图并保存
                # img = cv2.rectangle(img, (x, y), (x+cut_width, y+cut_width), (0, 0, 255), 1)
                x1, x2, y1, y2 = x, x+cut_width, y, y+cut_width
                cut_img = img[y:y+cut_width, x:x+cut_width, :]

                # 切标注
                cut_bboxes = []

                for k,bbox in enumerate(bboxes):
                    gt_x1, gt_x2, gt_y1, gt_y2 = bbox['x1'], bbox['x2'], bbox['y1'], bbox['y2']

                    # 在切图范围内的目标
                    if x1 <= gt_x1 and gt_x2 <= x2 and y1 <= gt_y1 and gt_y2 <= y2:
                        cut_x1 = gt_x1 - x1
                        cut_x2 = gt_x2 - x1
                        cut_y1 = gt_y1 - y1
                        cut_y2 = gt_y2 - y1
                        cut_bbox = {
                            'name': bbox['name'],
                            'xmin': cut_x1,
                            'ymin': cut_y1,
                            'xmax': cut_x2,
                            'ymax': cut_y2
                        }
                        cut_bboxes.append(cut_bbox)

                        used_labels += 1

                        # 可视化label
                        # cut_img = cv2.putText(cut_img, '{}'.format(cut_bbox['name']), (cut_x1+2, cut_y1-4), font, 0.5, (0, 255, 0), 1)
                        # cut_img = cv2.rectangle(cut_img, (cut_x1, cut_y1), (cut_x2, cut_y2), (0, 255, 0), 1)

                cut_filename = '{}_{}_{}'.format(filename, x, y)

                # 保存切图
                cv2.imwrite('{}{}{}'.format(cut_img_path, cut_filename, ext), cut_img)

                if len(cut_bboxes) > 0:
                    # 保存备注
                    save_annotations(cut_anno_path, cut_filename, cut_img, cut_bboxes)
                    total += 1

                # 终止
                if finish_x:
                    break

            if finish_y:
                break


        # cv2.imwrite(cut_img_path+'test.jpg', img)
        print('img_filename:{}, total imgs:{}, total labels:{}, used labels:{}'.format(filename, total, total_labels, used_labels))

        # exit()
