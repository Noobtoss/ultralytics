from ultralytics.utils.loss import E2ELoss
from ultralytics.utils import LOGGER
from ultralytics.nn.tasks import DetectionModel as _DetectionModel

from .v8_detection_loss import v8DetectionLoss


class DetectionModel(_DetectionModel):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DetectionModel")
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        return E2ELoss(self, v8DetectionLoss) if getattr(self, "end2end", False) else v8DetectionLoss(self)
