from ultralytics.utils import RANK, LOGGER
from ultralytics.models.rtdetr.train import RTDETRTrainer as _RTDETRTrainer

from .rtdetr_detection_model import RTDETRDetectionModel


class RTDETRTrainer(_RTDETRTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] RTDETRTrainer")
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
