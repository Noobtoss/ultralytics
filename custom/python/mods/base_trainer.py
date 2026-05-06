from ultralytics.utils import LOGGER
from ultralytics.engine.trainer import BaseTrainer as BaseBaseTrainer


class BaseTrainer(BaseBaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] BaseTrainer")
        super().__init__(*args, **kwargs)
