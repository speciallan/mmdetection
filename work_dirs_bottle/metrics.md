# Plan

使用FCOS baseline
1、使用FCOSTD [n]
2、使用Cascade [y]
3、增加epoch [y]
4、使用大size做test [y]
5、使用数据增强提高泛化性能，翻转，多尺度训练，增加模型尺度不变形。 [y]
6、使用可变卷积，因为目标占框比例很小，用可变卷积减少背景影响。[y]
7、分出val进行验证调参、查看train和test的分布情况，得出差异，使用9:1 [y]
8、分析数据源、背景不算分，不需要背景，清洗掉。[y]
9、修数据，很多标注不准确，大尺寸标注不好修改小，因为test可能也标注的很大且评价iou为0.8。[x]
10、模型集成，数据分为658x492的rgb图和4096x3000的灰度图，且为两种不同环境，使用多模型集成。[n]
11、通过人工对test进行部分标注，扩大训练数据集。[?]
12、使用resnet的前几层做fpn [n]
13、使用densenet或者resnext101等做骨干网 [n] resnet50模型容量完全够了，但是特征表达能力不一定够
14、修改FPN出来的层数，使用第一层，尝试增加一层，并增加anchor_strides，以拟合小目标 [n] 第3层特征不适合检测小目标
且召回足够，不需要更多anchor的覆盖。修改为多ratio和stride也没有提高。[n]
15、使用coco预训练模型增加收敛速度和模型泛化能力 [y]
16、因为数据太少，需要降低正样本iou，提高正例数量，减少过拟合风险。因为瑕疵占框比很小，rpn正负样本划分iou改为0.5:0.1[n]
17、对原始数据进行数据增强、包括裁剪、旋转、平移，复制到各个地方，增强平移不变性和旋转不变性，从而提高模型泛化能力。
18、并不是学的越多就越好，变成书呆子，根据情况控制学习率和训练程度，最好epoch到8-12[y]
19、对gt目标进行聚类，判断40、120、420区间的数量，根据数量分布设计anchor。
20、增加全局上下文，提高分类能力。
21、断点、破损尺度比较小，数量多且权重占比高，在保持AP波动不大的情况下，尽可能提高高权值类别AP。
22、使用SWA做单模融合[y]
23、增加OHEM进行rcnn阶段困难样本采样。[n] 大目标损失高，小困难目标损失小，被忽略。
24、可视化特征，查看是否学到想要的信息，如果没有尝试增加attention或者传统方法。
25、增加半监督方法，比如mean teacher来增加数据，目前不好做。
26、使用soft_nms增加输出框数量，提高召回率。[y] 提高接近1个点
27、目前召回率拉满，定位精度差，尝试HRNet等检测骨干网替换分类网络提高定位精度。[y] 对大iou有效，提升很小，效率低
28、修改L1Loss为ioulloss，提高回归精度。
29、根据预测结果，针对性降低nms_score来提高召回率从而提高mAP，或者提高nms_iou_thr提高精度。[y]
30、在检测头增加iou分支，用来提高回归精度。
31、cascade使用4阶段且iou阈值为0.5-0.8。
32、50px内目标占大多数，提高s和m的分数，小目标产生的正样本少，精度低，可以优化阈值到0.4/0.5/0.6增大正样本数量[n]
33、需要实现一个定制AP评估器，和带权bottle分值计算器。因为cocoAP高的检测器，在比赛规则下不一定分数最高，看AP50。
34、修改rpn的iou阈值提高正负样本区分度，提高判别能力，从而提高精度。[n]
35、最后的挣扎，x101+swa+多类别thr预测集成
36、如何提高精度？即提高困难样本检测精度。增大图片尺度？多学习困难样本提高其置信度？更强的模型？切换正负样本划分比例？数据增强？Focalloss？


初步判断：
已经过拟合，需要增加数据[n]
召回性能不错，但是精度不高。[y]

# Metrics

| Model | Backbone | Lr Schd | Param(M) | FPS | AP | AP50 | AP75 | APs | APm | APl | Score | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FCOS | ResNet50 | 2x | - | --- | 0.647 | 0.891 | 0.697 | 0.530 | 0.570 | 0.714 | 0.66791 |
| FCOS | ResNet50 | e84 | - | --- | 0.924 | 0.983 | 0.957 | 0.881 | 0.901 | 0.956 | 0.67465 |
| FCOSTD | ResNet50 | 2x | - | --- | 0.600 | 0.869 | 0.645 | 0.342 | 0.677 | 0.458 | 0.66376 |
| CascadeRCNN | ResNet50 | 1x |  | 13.1 | 0.424 | 0.681 | 0.432 | 0.167 | 0.312 | 0.437 | - |
| CascadeRCNN | ResNet50 | 2x |  | 13.1 |  |  |  |  | | 0.68661 |
| CascadeRCNN | ResNet50 | 4x |  | 13.1 | 0.893 | 0.972 | 0.949 | 0.891 | 0.831 | 0.930 | 0.69899 |
| CascadeRCNN+DCN | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 0.58743 |
| CascadeRCNN+ms | ResNet50 | 2x | --- | 8.3 | 0.942 | 0.993 | 0.987 | 0.966 | 0.939 | 0.935 | 0.70828 |
| CascadeRCNN+DCN+1333+ms | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 0.72041 |

Cascade RCNN r50

val e12

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.460
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.739
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.460
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.279
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.347
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.552
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 29.818 | 喷码正常 | 37.436 | 瓶盖断点 | 18.336 |
| 瓶盖坏边 | 83.613 | 瓶盖打旋 | 30.361 | 瓶盖变形 | 68.917 |
| 标贴气泡 | 14.537 | 标贴歪斜 | 16.151 | 喷码异常 | 74.315 |
| 标贴起皱 | 86.258 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

val e24

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.739
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.285
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.474
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.365
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.491
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.576
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 29.683 | 喷码正常 | 33.347 | 瓶盖断点 | 21.131 |
| 瓶盖坏边 | 82.305 | 瓶盖打旋 | 31.859 | 瓶盖变形 | 68.875 |
| 标贴气泡 | 15.894 | 标贴歪斜 | 16.986 | 喷码异常 | 75.001 |
| 标贴起皱 | 83.116 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+


val r50+ms+1333+dcn

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.527
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.807
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.663
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.684
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.731
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 37.552 | 喷码正常 | 41.178 | 瓶盖断点 | 25.748 |
| 瓶盖坏边 | 86.488 | 瓶盖打旋 | 37.261 | 瓶盖变形 | 69.974 |
| 标贴气泡 | 34.068 | 标贴歪斜 | 32.415 | 喷码异常 | 78.573 |
| 标贴起皱 | 83.392 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

val r50+ms+1333/1000+dcn+coco 1.30

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.561
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.836
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.593
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.616
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.666
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.674
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.692
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 53.190 | 喷码正常 | 43.347 | 瓶盖断点 | 27.599 |
| 瓶盖坏边 | 85.737 | 瓶盖打旋 | 40.787 | 瓶盖变形 | 77.953 |
| 标贴气泡 | 42.790 | 标贴歪斜 | 25.664 | 喷码异常 | 78.132 |
| 标贴起皱 | 85.551 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+ 0.72

val r50+单尺度+638+fpn6+anchor5+dcn+coco 1.31
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 80.624 | 喷码正常 | 16.567 | 瓶盖断点 | 17.160 |
| 瓶盖坏边 | 27.402 | 瓶盖打旋 | 82.134 | 瓶盖变形 | 17.169 |
| 标贴气泡 | 17.201 | 标贴歪斜 | 31.070 | 喷码异常 | 69.415 |
| 标贴起皱 | 70.938 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+  坏边88 打旋 82  破损 80 起皱87  变形77
因为anchor调整 破损、打旋ap高

+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 31.969 | 喷码正常 | 29.897 | 瓶盖断点 | 12.895 |
| 瓶盖坏边 | 88.143 | 瓶盖打旋 | 36.359 | 瓶盖变形 | 73.208 |
| 标贴气泡 | 11.141 | 标贴歪斜 | 23.750 | 喷码异常 | 79.009 |
| 标贴起皱 | 86.734 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+ 

val baseline r50+ms+2048/1200+dcn+coco 2.1

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.585
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.869
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.605
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.556
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.709
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.725
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.696
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.747
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 43.438 | 喷码正常 | 37.894 | 瓶盖断点 | 23.672 |
| 瓶盖坏边 | 90.143 | 瓶盖打旋 | 41.433 | 瓶盖变形 | 73.497 |
| 标贴气泡 | 51.124 | 标贴歪斜 | 49.762 | 喷码异常 | 84.373 |
| 标贴起皱 | 90.120 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+ 0.1952/0.2778/0.7994
检测特大特小目标效果差，能检测到，定位性能还行，精度不高

val r50+ms+1333/1000+dcn+fpn6+coco +nms0.7 2.2

+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 41.482 | 喷码正常 | 34.112 | 瓶盖断点 | 20.653 |
| 瓶盖坏边 | 87.066 | 瓶盖打旋 | 40.944 | 瓶盖变形 | 73.691 |
| 标贴气泡 | 28.590 | 标贴歪斜 | 37.107 | 喷码异常 | 80.978 |
| 标贴起皱 | 85.857 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+ 0.1373 /0.1945/0.67882 1000短边还是不行，fpn6无效

val r50+ms+1333/1200+dcn+fpn+coco+iou0.6  2.2

+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 42.035 | 喷码正常 | 35.873 | 瓶盖断点 | 21.806 |
| 瓶盖坏边 | 87.498 | 瓶盖打旋 | 36.822 | 瓶盖变形 | 74.838 |
| 标贴气泡 | 49.612 | 标贴歪斜 | 47.637 | 喷码异常 | 84.822 |
| 标贴起皱 | 90.837 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+ 0.78442 降低，说明iou0.6无效

val res50+ms+s1200/1800+1500+coco+softnms0.01/0.5 2.9

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.572
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.862
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.468
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.530
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.619
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.695
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.720
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.676
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.705
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.734
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 43.678 | 喷码正常 | 37.436 | 瓶盖断点 | 22.666 |
| 瓶盖坏边 | 88.087 | 瓶盖打旋 | 41.494 | 瓶盖变形 | 73.968 |
| 标贴气泡 | 43.482 | 标贴歪斜 | 49.394 | 喷码异常 | 83.667 |
| 标贴起皱 | 87.708 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

val res50+ms+s1200/1800+1500+coco+softnms0.001/0.5 2.9

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.597
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.885
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.623
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.471
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.551
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.647
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.713
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.744
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.728
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.775
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 45.920 | 喷码正常 | 40.718 | 瓶盖断点 | 24.344 |
| 瓶盖坏边 | 88.210 | 瓶盖打旋 | 43.729 | 瓶盖变形 | 78.680 |
| 标贴气泡 | 48.994 | 标贴歪斜 | 52.565 | 喷码异常 | 84.798 |
| 标贴起皱 | 88.779 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

val res50+ms+s1200/1800+1500+coco+softnms0.001/0.5+alldata 2.9

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.604
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.882
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.639
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.475
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.560
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.746
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.684
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.732
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.776
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 47.643 | 喷码正常 | 41.995 | 瓶盖断点 | 24.812 |
| 瓶盖坏边 | 89.213 | 瓶盖打旋 | 44.080 | 瓶盖变形 | 80.853 |
| 标贴气泡 | 49.643 | 标贴歪斜 | 52.615 | 喷码异常 | 84.459 |
| 标贴起皱 | 88.682 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

1 0.445 0.723 0.482 0.349 0.447 0.514
2 0.518 0.800 0.547 0.380 0.502 0.561
3 0.527 0.824 0.564 0.429 0.499 0.561
4 0.547 0.835 0.579 0.373 0.517 0.578
5 0.572 0.861 0.602 0.419 0.537 0.606
6 0.575 0.863 0.602 0.427 0.531 0.620
7 0.572 0.862 0.591 0.468 0.530 0.619 +all lr
8 0.588 0.879 0.613 0.452 0.542 0.634 +all
9 0.597 0.885 0.623 0.471 0.551 0.647 +all
10 0.604 0.882 0.639 0.475 0.560 0.649 +all
11 0.607 0.887 0.639 0.475 0.562 0.650 +all
12 0.608 0.890 0.644 0.481 0.566 0.653 +all
13 0.609 0.890 0.644 0.482 0.569 0.650 
14 0.608 0.891 0.641 0.487 0.568 0.655
15 0.610 0.890 0.647 0.493 0.573 0.655
16 0.609 0.893 0.647 0.478 0.571 0.655
17 0.609 0.893 0.645 0.486 0.573 0.652
18 0.612 0.891 0.645 0.485 0.569 0.656
19 0.612 0.893 0.650 0.487 0.573 0.656
20 0.613 0.893 0.651 0.486 0.574 0.657
21 0.612 0.893 0.651 0.487 0.573 0.656
es 0.613 0.894 0.651 0.488 0.574 0.658

val hrnet+ms+s1000/1200+1200+coco+softnms0.02/0.5 2.5

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.590
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.867
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.601
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.532
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.743
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.585
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.720
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.785
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 44.179 | 喷码正常 | 38.109 | 瓶盖断点 | 21.894 |
| 瓶盖坏边 | 88.601 | 瓶盖打旋 | 42.901 | 瓶盖变形 | 74.588 |
| 标贴气泡 | 51.103 | 标贴歪斜 | 51.473 | 喷码异常 | 86.227 |
| 标贴起皱 | 90.700 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

0 0.499 0.778 0.530 0.371 0.488 0.556
0 0.513 0.787 0.537 0.357 0.470 0.551
1 0.564 0.837 0.602 0.383 0.509 0.637 *
2 0.563 0.845 0.595 0.387 0.508 0.627 *
3 0.559 0.833 0.603 0.347 0.515 0.637
4 0.562 0.846 0.592 0.373 0.516 0.634
5 0.585 0.857 0.611 0.368 0.530 0.666
6 0.590 0.867 0.601 0.374 0.532 0.680 *
7 0.587 0.856 0.600 0.361 0.529 0.674
8 0.587 0.860 0.604 0.360 0.519 0.681
9 0.592 0.859 0.613 0.363 0.525 0.691
10 0.592 0.862 0.612 0.357 0.524 0.690
11 0.594 0.861 0.617 0.357 0.524 0.692 
12 0.590 0.859 0.607 0.361 0.525 0.678 + OHEM

val baseline hrnet+ms+s1000/1200+1200+coco+ensemble 2.6

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.594
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.867
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.605
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.383
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.535
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.681
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.557
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.719
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.742
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.567
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.785
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 44.418 | 喷码正常 | 39.476 | 瓶盖断点 | 23.088 |
| 瓶盖坏边 | 89.906 | 瓶盖打旋 | 43.804 | 瓶盖变形 | 75.247 |
| 标贴气泡 | 50.960 | 标贴歪斜 | 51.470 | 喷码异常 | 85.730 |
| 标贴起皱 | 90.259 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

类别序号	类别名称	类别权重 标签数量 总得分
1	瓶盖破损	0.15 1619 242.85 *
2	瓶盖变形	0.09 705 63.45 *
3	瓶盖坏边	0.09 656 59.04
4	瓶盖打旋	0.05 480 24
5	瓶盖断点	0.13 614 79.82 *
6	标贴歪斜	0.05 186 9.3
7	标贴起皱	0.12 384 46.08
8	标贴气泡	0.13 443 57.59 *
9	正常喷码	0.07 489 34.23
10	异常喷码	0.12 199 23.88

50px内目标占大多数，提高s和m的分数

val x101 e12

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.584
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.850
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.598
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.370
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.506
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.681
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.716
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.729
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.610
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.683
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.792
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 43.250 | 喷码正常 | 38.973 | 瓶盖断点 | 19.549 |
| 瓶盖坏边 | 87.823 | 瓶盖打旋 | 43.753 | 瓶盖变形 | 77.766 |
| 标贴气泡 | 47.346 | 标贴歪斜 | 49.902 | 喷码异常 | 84.467 |
| 标贴起皱 | 90.963 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+


r50+ms+e48

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.942
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.993
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.987
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.966
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.939
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.935
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.775
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.958
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.958
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.971
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.951
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.949
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 93.368 | 喷码正常 | 96.186 | 瓶盖断点 | 94.998 |
| 瓶盖坏边 | 91.327 | 瓶盖打旋 | 95.781 | 瓶盖变形 | 93.638 |
| 标贴气泡 | 90.789 | 标贴歪斜 | 93.997 | 喷码异常 | 95.556 |
| 标贴起皱 | 96.723 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

FCOS r50 

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.647
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.891
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.697
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.530
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.570
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.714
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.715
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.721
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.686
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.778
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 52.293 | 喷码正常 | 76.231 | 瓶盖断点 | 64.714 |
| 瓶盖坏边 | 88.151 | 瓶盖打旋 | 40.018 | 瓶盖变形 | 85.881 |
| 标贴气泡 | 35.245 | 标贴歪斜 | 34.671 | 喷码异常 | 82.657 |
| 标贴起皱 | 87.480 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

e84

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.923
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.984
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.957
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.881
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.901
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.956
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.771
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.938
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.938
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.900
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.918
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.963
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 86.623 | 喷码正常 | 97.465 | 瓶盖断点 | 96.312 |
| 瓶盖坏边 | 98.560 | 瓶盖打旋 | 80.963 | 瓶盖变形 | 99.416 |
| 标贴气泡 | 88.400 | 标贴歪斜 | 82.004 | 喷码异常 | 96.235 |
| 标贴起皱 | 96.586 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

FCOSTD r50

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.600
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.869
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.645
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.342
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.658
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.678
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.685
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.583
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.660
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.739
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 50.231 | 喷码正常 | 70.739 | 瓶盖断点 | 52.955 |
| 瓶盖坏边 | 80.011 | 瓶盖打旋 | 37.330 | 瓶盖变形 | 79.590 |
| 标贴气泡 | 30.180 | 标贴歪斜 | 31.553 | 喷码异常 | 82.203 |
| 标贴起皱 | 84.898 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

FCOSTD x101

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.490
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.767
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.535
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.280
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.597
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.495
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.588
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 35.832 | 喷码正常 | 57.827 | 瓶盖断点 | 42.896 |
| 瓶盖坏边 | 82.423 | 瓶盖打旋 | 30.479 | 瓶盖变形 | 59.544 |
| 标贴气泡 | 13.477 | 标贴歪斜 | 16.308 | 喷码异常 | 77.731 |
| 标贴起皱 | 73.563 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+