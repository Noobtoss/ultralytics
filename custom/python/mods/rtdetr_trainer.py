from ultralytics.utils import LOGGER
from ultralytics.models.rtdetr.train import RTDETRTrainer as _RTDETRTrainer


class RTDETRTrainer(_RTDETRTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] RTDETRTrainer")
        super().__init__(*args, **kwargs)
