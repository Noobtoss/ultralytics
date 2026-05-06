from ultralytics.models.yolo.detect import DetectionValidator as BaseDetectionValidator
from ultralytics.utils import LOGGER

class DetectionValidator(BaseDetectionValidator):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DetectionValidator")
        super().__init__(*args, **kwargs)
