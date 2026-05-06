from ultralytics.models.yolo.detect import DetectionValidator as _DetectionValidator
from ultralytics.utils import LOGGER


class DetectionValidator(_DetectionValidator):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DetectionValidator")
        super().__init__(*args, **kwargs)
