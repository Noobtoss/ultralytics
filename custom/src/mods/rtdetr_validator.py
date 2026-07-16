from ultralytics.utils import LOGGER
from ultralytics.models.rtdetr.val import RTDETRValidator as _RTDETRValidator

from .detection_validator import DetectionValidator


_RTDETRValidator.__bases__ = (DetectionValidator,)


class RTDETRValidator(_RTDETRValidator):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("[Modded] RTDETRValidator")
        super().__init__(*args, **kwargs)
