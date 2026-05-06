from ultralytics.utils import LOGGER
from ultralytics.engine.trainer import BaseTrainer as _BaseTrainer


class BaseTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] BaseTrainer")
        super().__init__(*args, **kwargs)
