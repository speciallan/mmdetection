import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, Scale, bias_init_with_prob

INF = 1e8


@HEADS.register_module
class FCOSTDHead(nn.Module):
    """
    Fully Convolutional One-Stage Object Detection head from [1]_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.

    References:
        .. [1] https://arxiv.org/abs/1904.01355

    Example:
        >>> self = FCOSTDHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 loss_rpn_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_rpn_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 # dcn
                 conv_cfg=None,
                 # gn
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(FCOSTDHead, self).__init__()

        self.num_classes = num_classes
        self.rpn_cls_out_channels = 2 - 1
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        # 可以调整
        self.loss_rpn_cls = build_loss(loss_rpn_cls)
        self.loss_rpn_bbox = build_loss(loss_rpn_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self._init_layers()

    def _init_layers(self):
        # rpn branch
        self.rpn_cls_convs = nn.ModuleList()
        self.rpn_reg_convs = nn.ModuleList()

        # cls branch
        self.cls_convs = nn.ModuleList()
        # reg branch
        self.reg_convs = nn.ModuleList()

        # rpn 一层卷积
        for i in range(1):
            self.rpn_cls_convs.append(
                ConvModule(
                    self.in_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None
                )
            )
            self.rpn_reg_convs.append(
                ConvModule(
                    self.in_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None
                )
            )

        # 叠加卷积
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))

        # rpn 一个通道 target是不同res的mask
        self.rpn_cls = nn.Conv2d(self.feat_channels, self.rpn_cls_out_channels, 3, padding=1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)

        # 类别数
        self.fcos_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        # 坐标数 4
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        # 原图大小二值图
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        for m in self.rpn_cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.rpn_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.rpn_cls, std=0.01, bias=bias_cls)
        normal_init(self.rpn_reg, std=0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):

        rpn_cls_feat = x
        rpn_reg_feat = x

        # rpn cls
        for cls_layer in self.rpn_cls_convs:
            rpn_cls_feat = cls_layer(rpn_cls_feat)

        rpn_cls_scores = self.rpn_cls(rpn_cls_feat)
        rpn_cls_logits = torch.sigmoid(rpn_cls_scores)

        # rpn reg
        for reg_layer in self.rpn_reg_convs:
            rpn_reg_feat = reg_layer(rpn_reg_feat)

        rpn_bbox_preds = scale(self.rpn_reg(rpn_reg_feat)).float().exp()
        # print(rpn_cls_scores.shape, rpn_bbox_preds.shape)


        # 经过rpn处理后的特征 像素级加权 + 跳接，rpn回归特征复用
        cls_feat = x * rpn_cls_logits
        cls_feat = x + cls_feat
        reg_feat = rpn_reg_feat

        # cls特征提取
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        # cls branch
        cls_score = self.fcos_cls(cls_feat)
        centerness = self.fcos_centerness(cls_feat)

        # reg branch
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(self.fcos_reg(reg_feat)).float().exp()

        return rpn_cls_scores, rpn_bbox_preds, cls_score, bbox_pred, centerness

    @force_fp32(apply_to=('rpn_cls_scores','rpn_bbox_preds','cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             rpn_cls_scores,
             rpn_bbox_preds,
             cls_scores,
             bbox_preds,
             centernesses,
             # 5输出+2信号输入
             gt_bboxes,
             gt_labels,
             # others
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        # 前向传播的所有值+gt_bboxes+gt_labels
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        # print(len(rpn_cls_scores), rpn_cls_scores[0].shape, rpn_bbox_preds[0].shape)

        # c4 c5 c6 c7 256/2^7
        # cls_scores [64, 1, 32, 32] ... [64, 1, 2, 2]   2,4,8,16,32 * 8
        # featmap.size()[-2:] = [2,2] 特征图尺度
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # print(cls_scores[4].shape, featmap_sizes)
        # exit()

        # bbox_preds [64, 4, 32, 32] ... 同上
        # 根据特征图大小和步长 每个像素点生成 strides=[8,16,32,64,128] 的x，y 因为图片为256*256，所以共(256/8)^2+256+64+8+2个
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)

        # ------------------------------ rpn -------------------------------

        # 根据gt生成rpn监督信号
        rpn_labels, rpn_bbox_targets = self.rpn_target(all_level_points, gt_bboxes, gt_labels)
        num_imgs = rpn_cls_scores[0].size(0)

        # 通过rpn修正后的类激活图和回归边框
        flatten_rpn_cls_scores = [
            rpn_cls_score.permute(0, 2, 3, 1).reshape(-1, self.rpn_cls_out_channels) # class 1/0
            for rpn_cls_score in rpn_cls_scores
        ]
        flatten_rpn_bbox_preds = [
            rpn_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for rpn_bbox_pred in rpn_bbox_preds
        ]

        flatten_rpn_cls_scores = torch.cat(flatten_rpn_cls_scores)
        flatten_rpn_bbox_preds = torch.cat(flatten_rpn_bbox_preds)

        flatten_rpn_labels = torch.cat(rpn_labels)
        flatten_rpn_bbox_targets = torch.cat(rpn_bbox_targets)
        flatten_rpn_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_rpn_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)

        # 这句话 && index < sizes[i] && "index out of bounds"` failed.
        # for k,v in enumerate(flatten_rpn_cls_scores):
        #     print(flatten_rpn_cls_scores[k], flatten_rpn_labels[k])
        # exit()
        loss_rpn_cls = self.loss_rpn_cls(
            flatten_rpn_cls_scores, flatten_rpn_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_rpn_bbox_preds = flatten_rpn_bbox_preds[pos_inds]

        if num_pos > 0:
            pos_rpn_bbox_targets = flatten_rpn_bbox_targets[pos_inds]
            pos_rpn_points = flatten_rpn_points[pos_inds]

            # 正采样点 正预测框
            pos_decoded_rpn_bbox_preds = distance2bbox(pos_rpn_points, pos_rpn_bbox_preds)
            # 正采样点 正框target
            pos_decoded_rpn_target_preds = distance2bbox(pos_rpn_points, pos_rpn_bbox_targets)

            loss_rpn_bbox = self.loss_bbox(
                pos_decoded_rpn_bbox_preds,
                pos_decoded_rpn_target_preds,
            )
        else:
            loss_rpn_bbox = pos_rpn_bbox_preds.sum()

        # ------------------------------ fcos -------------------------------

        # 根据gt生成fcos监督信号
        labels, bbox_targets = self.fcos_target(all_level_points, gt_bboxes, gt_labels)
        num_imgs = cls_scores[0].size(0)

        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        # torch.Size([87296, 1]) torch.Size([87296, 4]) torch.Size([87296])
        # print(flatten_cls_scores.shape, flatten_bbox_preds.shape, flatten_centerness.shape)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # torch.Size([1024, 2]) torch.Size([65536, 2])
        # print(all_level_points[0].shape, all_level_points[0].repeat(num_imgs, 1).shape)

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]

            # 正采样点 正预测框
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            # 正采样点 正框target
            pos_decoded_target_preds = distance2bbox(pos_points, pos_bbox_targets)

            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_rpn_cls=loss_rpn_cls,
            loss_rpn_bbox=loss_rpn_bbox,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        # [  0.,   8.,  16.,  24 ... 248] 共32个
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        # torch.Size([32, 32]) torch.Size([32, 32]) torch.Size([1024, 2])
        # print(x.shape, y.shape, points.shape)
        # exit()
        return points

    def rpn_target(self, points, gt_bboxes_list, gt_labels_list):
        """将gt映射到特征图上"""

        # 先验设置特征图的回归范围，没有进行在线特征选择
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]

        # 共1364个 = 32^2=1024个(0,64) 256个(64,128) 64(128,256) 16个(256,512) 4个(512,+)
        # print(len(expanded_regress_ranges),expanded_regress_ranges[0].shape, expanded_regress_ranges)
        # exit()

        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # labels (32,1364) bboxes (32,1364,4)
        # print(len(labels_list), labels_list[0].shape, len(bbox_targets_list), bbox_targets_list[0].shape)
        # exit()

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))

        # fpn每一层所有生成框的拼接
        # torch.Size([32768]) torch.Size([32768, 4])
        # print(concat_lvl_labels[0].shape, concat_lvl_bbox_targets[0].shape)
        # exit()

        return concat_lvl_labels, concat_lvl_bbox_targets

    def rpn_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        # x1 y1 x2 y2   x2-x1
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)

        # 每个gt的面积
        areas = areas[None].repeat(num_points, 1)
        # print(areas.shape, areas)
        # exit()

        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        # condition1: inside a gt bbox
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (max_regress_distance >= regress_ranges[..., 0]) & (
                                       max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        # labels 1/0
        labels = gt_labels[min_area_inds]
        labels[min_area != INF] = 1
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges)

        # split to per img, per level
        num_points = [center.size(0) for center in points]
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges):
        # torch.Size([1, 4]) torch.Size([1])
        # print(gt_bboxes, gt_labels)
        # exit()
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        # x1 y1 x2 y2   x2-x1
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)

        # 每个gt的面积
        areas = areas[None].repeat(num_points, 1)
        # print(areas.shape, areas)
        # exit()

        # torch.Size([1364, 2])
        # print(regress_ranges.shape)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        # torch.Size([1364, 1, 2])
        # 每个像素点 对应一个框 和 回归范围

        # torch.Size([2, 4]) 2个gtbbox
        # torch.Size([1, 2, 4])
        # torch.Size([1364, 2, 4]) 每个像素点对应1个gtbbox
        # print(gt_bboxes.shape)
        # print(gt_bboxes[None].shape)
        # print(gt_bboxes[None].expand(num_points, num_gts, 4).shape)
        # exit()

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        # 每个像素点坐标，即候选框中间点坐标
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        # torch.Size([1364, 1, 4]) x1,y1,x2,y2
        # torch.Size([1364, 1]) x1
        # print(gt_bboxes.shape)
        # print(gt_bboxes[...,0].shape)
        # exit()

        # 计算中心点到四条边的偏移量
        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        # print(bbox_targets)
        # print(bbox_targets.min(-1))
        # exit()

        # condition1: inside a gt bbox 如果偏移量都大于0，则点在框内
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
                max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area 将点在框外和大于回归范围的区域都赋值为inf
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        # print(min_area, min_area_inds)
        # exit()

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0 # 将点在框外和大于回归范围的区域 label置为0，bg
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
