#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

def main():
    #gen coco pretrained weight
    path = './work_dirs_bottle/cascade_rcnn_x101_fpn_dcn/checkpoints/'
    import torch
    num_classes = 11
    model_coco = torch.load(path+"cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth") # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] =    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] =    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] =    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] =    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][ :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] =    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][ :num_classes]
    # save new model
    torch.save(model_coco, path+"cascade_rcnn_r50_coco_pretrained_weights_classes_%d.pth" % num_classes)
if __name__ == "__main__":
    main()