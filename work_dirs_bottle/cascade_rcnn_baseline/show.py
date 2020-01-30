#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:Speciallan



# 可视化gt
#     all_boxes = annotations[id]
#
#     for classid, boxes in enumerate(all_boxes):
#
#         if len(boxes) == 0:
#             continue
#
#         # print(classid, boxes)
#
#         for k,v in enumerate(boxes):
#
#             if len(v) == 0:
#                 continue
#
#             x1, y1, x2, y2 = v
#
#             show_img = cv2.rectangle(show_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
#             show_img = cv2.putText(show_img, classes[classid], (x1 - 2, y1 - 2), font, 0.5, (0, 255, 0), 1)
#
#     combine = np.array(np.zeros((256, 256*3, 3)))
#     combine[:, 0:256, :] = origin_img
#     combine[:, 256:256*2, :] = show_img
#     combine[:, 256*2:256*3, :] = results_img
#     # combine = cv2.vconcat(origin_img, show_img)
#     cv2.imwrite(show_path, combine)
#     print('生成{}完毕'.format(show_path))