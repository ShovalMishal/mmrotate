# Copyright (c) OpenMMLab. All rights reserved.
from .dota_metric import DOTAMetric
from .rotated_coco_metric import RotatedCocoMetric
from .bbox_regressor_metric import BBoxRegressorMetric

__all__ = ['DOTAMetric', 'RotatedCocoMetric', 'BBoxRegressorMetric']
