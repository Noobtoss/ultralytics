from __future__ import annotations
from ultralytics.utils.loss import E2ELoss
from ultralytics.nn.tasks import DetectionModel

from .train_loss import TrainLoss


# THS, Copied from ultralytics.nn.tasks


class DetectionModel(DetectionModel):
    def init_criterion(self):
        return E2ELoss(self, TrainLoss) if getattr(self, "end2end", False) else TrainLoss(self)
