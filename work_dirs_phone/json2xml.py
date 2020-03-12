#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

from pycocotools.coco import COCO
import json
from lxml.etree import Element, ElementTree, SubElement
import os
import shutil

'''
    with open("COCO_train.json","r+") as f:
        data = json.load(f)
        print("read ready")

        for i in data:
            print(i)
            # info
            # licenses
            # categories
            # __raw_Chinese_name_df
            # images
            # annotations
'''

if __name__ == "__main__":

    root_path = '/Users/speciallan/Downloads/chongqing1_round1_train1_20191223/'
    json_name = root_path+'annotations_washed.json'
    xml_path = root_path+'imgs2check'
    if os.path.exists(xml_path):
        shutil.rmtree(xml_path)
    os.mkdir(xml_path)

    # 构建coco实例
    coco = COCO(json_name)

    # 得到所有图片id
    images_id = coco.getImgIds()

    # 通过id遍历得到图像的具体信息
    for id in images_id:
        img = coco.loadImgs(id)

        jpg_name = img[0].get('file_name')
        xml_name = os.path.join(xml_path, jpg_name.strip('.jpg') + '.xml')

        # 复制图片
        os.system('cp {}images/{} {}/{}'.format(root_path, jpg_name, xml_path, jpg_name))

        file = open(xml_name, 'wb+')
        node_root = Element('annotation')

        node_filename = SubElement(node_root, 'filename')
        node_filename.text = str(jpg_name)

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_height = SubElement(node_size, 'height')
        node_depth = SubElement(node_size, 'depth')
        node_width.text = str(img[0].get('width'))
        node_height.text = str(img[0].get('height'))
        node_depth.text = '3'

        # 根据图像id得到所有的标签id
        annIds = coco.getAnnIds(imgIds=id)
        # 根据标签的所有id得到具体的annotation信息
        cats = coco.loadAnns(annIds)

        '''
         [{'supercategory': '瓶盖破损', 'id': 1, 'name': '瓶盖破损'}, {'supercategory': '喷码正常', 'id': 9, 'name': '喷码正常'}, {'supercategory': '瓶盖断点', 'id': 5, 'name': '瓶盖断点'}, {'supercategory': '瓶盖坏边', 'id': 3, 'name': '瓶盖坏边'}, {'supercategory': '瓶盖打旋', 'id': 4, 'name': '瓶盖打旋'}, {'supercategory': '瓶盖变形', 'id': 2, 'name': '瓶盖变形'}, {'supercategory': '标贴气泡', 'id': 8, 'name': '标贴气泡'}, {'supercategory': '标贴歪斜', 'id': 6, 'name': '标贴歪斜'}, {'supercategory': '喷码异常', 'id': 10, 'name': '喷码异常'}, {'supercategory': '标贴起皱', 'id': 7, 'name': '标贴起皱'}]
        '''

        for anno in cats:
            node_object = SubElement(node_root, 'object')

            node_name = SubElement(node_object, 'name')
            '''
            # 按照id排序得到的类别list
            cats = coco.loadCats(coco.getCatIds())
            nms=[cat['name'] for cat in cats]
            '''
            # 根据类别id直接得到真实类别
            category_id = anno.get('category_id')
            cat_name = coco.loadCats(category_id)[0].get('name')

            # if id == 3603:
            #     print(anno.get('id'))
            #     print(anno.get('image_id'))
            #     print(anno.get('category_id'))
            #     print(cat_name)
            #     exit()

            node_bnbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bnbox, 'xmin')
            node_ymin = SubElement(node_bnbox, 'ymin')
            node_xmax = SubElement(node_bnbox, 'xmax')
            node_ymax = SubElement(node_bnbox, 'ymax')
            bbox_info = anno.get('bbox')

            # coco格式是左上角(x, y)与宽长(w, h)
            node_xmin.text = str(int(bbox_info[0]))
            node_ymin.text = str(int(bbox_info[1]))
            node_xmax.text = str(int(bbox_info[0] + bbox_info[2]))
            node_ymax.text = str(int(bbox_info[1] + bbox_info[3]))

            # 测试
            # if id == 3603:
            #     332 3603 737 1
                # print(jpg_name, id, annIds, category_id, cat_name, bbox_info)
            # else:
            #     continue

            node_name.text = str(cat_name)

        doc = ElementTree(node_root)
        doc.write(file, pretty_print=True)
    print('xml make done!')