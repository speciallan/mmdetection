#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

def main():
    #gen coco pretrained weight
    path = './work_dirs_bottle/cascade_rcnn_r50_fpn/checkpoints/'
    import torch
    num_classes = 11
    model_coco = torch.load(path+"cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166.pth") # weight
    model_coco = torch.load(path+"cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth") # weight
    model_coco = torch.load(path+"cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth") # weight

    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] =    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] =    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] =    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] =    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] =    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][ :num_classes]
    # save new model
    torch.save(model_coco, path+"cascade_rcnn_x101_coco_pretrained_weights_classes_%d.pth" % num_classes)
if __name__ == "__main__":
    main()