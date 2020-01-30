#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan

def main():
    #gen coco pretrained weight
    path = './work_dirs_bottle/fcos_r50_caffe_fpn/checkpoints/'
    import torch
    num_classes = 10
    model_coco = torch.load(path+"fcos_r50_caffe_fpn_gn_1x_4gpu_20190516-9f253a93.pth") # weight
    model_coco["state_dict"]["bbox_head.fcos_cls.weight"] =    model_coco["state_dict"]["bbox_head.fcos_cls.weight"][ :num_classes, :]
    model_coco["state_dict"]["bbox_head.fcos_cls.bias"] = model_coco["state_dict"]["bbox_head.fcos_cls.bias"][ :num_classes]
    # save new model
    torch.save(model_coco, path+"fcos_r50_coco_pretrained_weights_classes_%d.pth" % num_classes)
if __name__ == "__main__":
    main()