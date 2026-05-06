from ultralytics.engine.model import Model as _Model
from ultralytics.utils import LOGGER


class Model(_Model):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] Model")
        super().__init__(*args, **kwargs)
