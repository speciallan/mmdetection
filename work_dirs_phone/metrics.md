# Plan

所有图片的数量： 315
所有标注的数量： 31518
s/m/l 5504 15015 10999

原始数据：
train/val imgs 284 31 
train/val annos 28415 3103

除去错误图片后：
train/val annos 28978 2381

除去错误标注后：
train/val annos 28116 3228

图片数据：
基本上宽高比为2：1，宽范围3900-10375，高范围2200-5040

{'TT': 263, 'QZ': 260, 'HJ': 511, 'FC': 397, 'HH': 23068, 'M': 1271, 'BHH': 1009, 'GG': 369, 'SL': 47, 'HM': 218, 'XK': 409, 'KP': 249, 'DQ': 205, 'QP': 33, 'LF': 18, 'QJ': 14, 'QS': 9, 'BX': 1}

最多的类别 HH BHH M

最小HH 9x8


| Model | Backbone | Lr Schd | Param(M) | FPS | AP | AP50 | AP75 | APs | APm | APl | Score | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ReitnaNet | ResNet50 | 1x | | 20.9 | 


baseline retinanet + 1333,800 + e12

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.245
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.008
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.063
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.228
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.269
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.027
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.085
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.313
+----------+--------+----------+--------+----------+-------+
| category | AP     | category | AP     | category | AP    |
+----------+--------+----------+--------+----------+-------+
| TT       | 66.527 | QZ       | 64.692 | HJ       | 0.862 |
| FC       | 83.972 | HH       | 2.324  | M        | 1.429 |
| BHH      | 2.744  | GG       | 53.283 | SL       | 0.000 |
| HM       | 88.635 | XK       | 2.519  | KP       | 0.000 |
| DQ       | 0.000  | QP       | 0.000  | LF       | 0.000 |
| QJ       | nan    | QS       | nan    | BX       | nan   |
+----------+--------+----------+--------+----------+-------+
听筒、前置摄像头、防拆标、感光器、返回键 检测效果还行

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.491
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.685
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.549
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.561
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.464
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.578
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.403
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.634
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638
+----------+--------+----------+--------+----------+--------+
| category | AP     | category | AP     | category | AP     |
+----------+--------+----------+--------+----------+--------+
| HH       | 45.105 | GG       | 81.930 | M        | 44.089 |
| KP       | 39.223 | QZ       | 84.041 | BHH      | 36.968 |
| HJ       | 37.283 | XK       | 32.253 | TT       | nan    |
| DQ       | 34.683 | FC       | 67.558 | QP       | 18.885 |
| QS       | nan    | SL       | 75.050 | HM       | nan    |
| QJ       | 0.000  | LF       | 90.000 | BX       | nan    |
+----------+--------+----------+--------+----------+--------+

