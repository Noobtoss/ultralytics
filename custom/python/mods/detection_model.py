from ultralytics.utils.loss import E2ELoss
from ultralytics.utils import LOGGER
from ultralytics.nn.tasks import DetectionModel as BaseDetectionModel

from .train_loss import TrainLoss


class DetectionModel(BaseDetectionModel):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DetectionModel")
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        return E2ELoss(self, TrainLoss) if getattr(self, "end2end", False) else TrainLoss(self)
