# Plan

1、使用FCOSTD
2、使用Cascade
3、增加epoch
4、使用大size做test
5、使用数据增强提高泛化性能
6、使用可变卷积
7、分出val

# Metrics

| Model | Backbone | Lr Schd | Param(M) | FPS | AP | AP50 | AP75 | APs | APm | APl | Score | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CascadeRCNN | ResNet50 | 1x |  | 13.1 | 0.424 | 0.681 | 0.432 | 0.167 | 0.312 | 0.437 | - |
| CascadeRCNN | ResNet50 | 4x |  | 13.1 | 0.893 | 0.972 | 0.949 | 0.891 | 0.831 | 0.930 | - |
| FCOS | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 0.67465 |
| FCOS | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 0.66791 |
| FCOSTD | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 0.66376 |

Cascade RCNN r50

e12

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.424
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.681
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.432
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.321
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.518
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.523
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 18.639 | 喷码正常 | 41.748 | 瓶盖断点 | 22.283 |
| 瓶盖坏边 | 80.676 | 瓶盖打旋 | 24.762 | 瓶盖变形 | 71.450 |
| 标贴气泡 | 4.908  | 标贴歪斜 | 5.658  | 喷码异常 | 74.702 |
| 标贴起皱 | 79.647 | None     | None   | None     | None   |
+----------+--------+----------+--------+----------+--------+

e48

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.893
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.972
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.949
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.891
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.831
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.930
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.757
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.912
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.912
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.909
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.848
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.944
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| 瓶盖破损 | 85.687 | 喷码正常 | 95.497 | 瓶盖断点 | 93.621 |
| 瓶盖坏边 | 93.204 | 瓶盖打旋 | 82.607 | 瓶盖变形 | 97.364 |
| 标贴气泡 | 83.135 | 标贴歪斜 | 71.735 | 喷码异常 | 94.155 |
| 标贴起皱 | 96.094 | None     | None   | None     | None   |
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