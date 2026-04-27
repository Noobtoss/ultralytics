from ultralytics.models.yolo import YOLO as BaseYOLO
from ultralytics.utils import LOGGER

from .model import Model

BaseYOLO.__bases__ = (Model,)


class YOLO(BaseYOLO):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("YOLO __init__ called")
        super().__init__(*args, **kwargs)
