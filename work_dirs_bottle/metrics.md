# Plan

使用FCOS baseline
1、使用FCOSTD [n]
2、使用Cascade [y]
3、增加epoch [y]
4、使用大size做test [y]
5、使用数据增强提高泛化性能，包括多尺度训练 [y]
6、使用可变卷积，因为目标占框比例很小，用可变卷积减少背景影响。
7、分出val进行验证调参、查看train和test的分布情况，得出差异，使用9:1 [y]
8、分析数据源、背景不算分，不需要背景，清洗掉。[y]
9、修数据，很多标注不准确，大尺寸标注不好修改小，因为test可能也标注的很大且评价iou为0.8。
10、模型集成，数据分为658x492的rgb图和4096x3000的灰度图，且为两种不同环境，使用多模型集成。
11、通过人工对test进行部分标注，扩大训练数据集。
12、使用resnet的前几层做fpn [n]
13、使用densenet做骨干网
14、修改FPN出来的层数，使用第一层

初步判断：
已经过拟合，需要增加数据

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

val r50+ms+1333+dcn+coco

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