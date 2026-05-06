from ultralytics.engine.model import Model as _Model
from ultralytics.models.yolo import YOLO as _YOLO
from ultralytics.utils import LOGGER


class Model(_Model):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] Model")
        super().__init__(*args, **kwargs)


_YOLO.__bases__ = (Model,)


class YOLO(_YOLO):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] YOLO")
        super().__init__(*args, **kwargs)
