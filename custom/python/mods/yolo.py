from ultralytics.models.yolo import YOLO as _YOLO
from ultralytics.utils import LOGGER

from .model import Model

_YOLO.__bases__ = (Model,)


class YOLO(_YOLO):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] YOLO")
        super().__init__(*args, **kwargs)
