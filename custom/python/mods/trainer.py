from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK

from .detection_model import DetectionModel


# THS, Copied from ultralytics.models.yolo.detect.DetectionTrainer
# THS, Copied from ultralytics.engine.trainer.BaseTrainer


class Trainer(DetectionTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
