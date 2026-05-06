from ultralytics.utils.loss import E2ELoss
from ultralytics.utils import LOGGER
from ultralytics.nn.tasks import DetectionModel as _DetectionModel

from .train_loss import TrainLoss


class DetectionModel(_DetectionModel):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DetectionModel")
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        return E2ELoss(self, TrainLoss) if getattr(self, "end2end", False) else TrainLoss(self)
