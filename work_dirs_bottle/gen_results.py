#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

# encoding:utf/8
import sys
from mmdet.apis import inference_detector, init_detector
import json
import os
import numpy as np
import argparse
from tqdm import tqdm


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


'''
类别序号	类别名称	类别权重 标签数量 总得分
1	瓶盖破损	0.15 1619 242.85 * 0.1<
2	瓶盖变形	0.09 705 63.45 *
3	瓶盖坏边	0.09 656 59.04
4	瓶盖打旋	0.05 480 24
5	瓶盖断点	0.13 614 79.82 *
6	标贴歪斜	0.05 186 9.3
7	标贴起皱	0.12 384 46.08
8	标贴气泡	0.13 443 57.59 *  0.6<
9	正常喷码	0.07 489 34.23
10	异常喷码	0.12 199 23.88
'''
# generate result
def result_from_dir():
    index = {1: 1, 2: 9, 3: 5, 4: 3, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
    thr = {1: 0.1, 2: 0.02, 3: 0.05, 4: 0.1, 5: 0.1, 6: 0.05, 7: 0.05, 8: 0.02, 9: 0.05, 10: 0.05}
    # build the model from a config file and a checkpoint file
    model = init_detector(config2make_json, model2make_json, device='cuda:0')
    pics = os.listdir(pic_path)
    meta = {}
    images = []
    annotations = []
    num = 0
    for im in tqdm(pics):
        num += 1
        img = os.path.join(pic_path, im)
        result_ = inference_detector(model, img)
        images_anno = {}
        images_anno['file_name'] = im
        images_anno['id'] = num
        images.append(images_anno)
        for i, boxes in enumerate(result_, 1):
            if len(boxes):
                defect_label = index[i]
                for box in boxes:
                    anno = {}
                    anno['image_id'] = num
                    anno['category_id'] = defect_label
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                    anno['bbox'][2] = round(anno['bbox'][2] - anno['bbox'][0], 2)
                    anno['bbox'][3] = round(anno['bbox'][3] - anno['bbox'][1], 2)
                    anno['score'] = round(float(box[4]), 2)

                    score_thr = thr[defect_label]
                    if anno['score'] < score_thr:
                        continue

                    annotations.append(anno)

    meta['images'] = images
    meta['annotations'] = annotations
    with open(json_out_path, 'w') as fp:
        # json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))
        json.dump(meta, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate result")
    parser.add_argument("-m", "--model", help="Model path", type=str, )
    parser.add_argument("-c", "--config", help="Config path", type=str, )
    parser.add_argument("-im", "--im_dir", help="Image path", type=str, )
    parser.add_argument('-o', "--out", help="Save path", type=str, )
    args = parser.parse_args()
    model2make_json = args.model
    config2make_json = args.config
    json_out_path = args.out
    pic_path = args.im_dir
    result_from_dir()