from __future__ import annotations
from ultralytics.utils.loss import v8DetectionLoss

from .tmp_custom_loss import TmpCustomLoss


# THS, Copied from ultralytics.utils.loss


class TrainLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        # Only override bbox_loss
        self.bbox_loss = TmpCustomLoss(model.model[-1].reg_max).to(self.device)
