from .registry import DATASETS
from .xml_style import XMLDataset
from .coco import CocoDataset

@DATASETS.register_module
class SARDataset(CocoDataset):

    CLASSES = ('ship',)

# @DATASETS.register_module
# class SARDataset(XMLDataset):
#
#     CLASSES = ['ship']
#
#     def __init__(self, **kwargs):
#         super(SARDataset, self).__init__(**kwargs)
#         if 'VOC2007' in self.img_prefix:
#             self.year = 2007
#         elif 'VOC2012' in self.img_prefix:
#             self.year = 2012
#         elif 'SAR' in self.img_prefix:
#             self.year = 2007
#         else:
#             raise ValueError('Cannot infer dataset year from img_prefix')
