from .registry import DATASETS
from .xml_style import XMLDataset
from .coco import CocoDataset

@DATASETS.register_module
class PhoneDataset(CocoDataset):

    # CLASSES = ('SL', 'HH', 'BHH', 'DQ', 'QJ', 'XK', 'KP', 'QS', 'LF', 'PFL', 'QP', 'M', 'BX', 'HJ',
    #            'FC', 'TT', 'QZ', 'GG', 'PP', 'AJ', 'HM', 'LH', 'QM', 'CCC', 'HZ', 'HZW', 'SG', 'NFC', 'LGO', 'XZF', 'ZF', 'SSS', 'JD', 'CJ', 'JJ', 'DJ', 'KT', 'KZ', 'KQ', 'YJ', 'WC', 'SH', 'RK', 'SX', 'MK', 'JG', 'HD', 'NGC', 'BQ', 'LS')

    # {'TT': 263, 'QZ': 260, 'HJ': 511, 'FC': 397, 'HH': 23068, 'M': 1271, 'BHH': 1009, 'GG': 369, 'SL': 47, 'HM': 218, 'XK': 409, 'KP': 249, 'DQ': 205, 'QP': 33, 'LF': 18, 'QJ': 14, 'QS': 9, 'BX': 1}

    CLASSES = ('TT', 'QZ', 'HJ', 'FC', 'HH', 'M', 'BHH', 'GG', 'SL', 'HM', 'XK', 'KP', 'DQ', 'QP', 'LF', 'QJ', 'QS', 'BX')

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
