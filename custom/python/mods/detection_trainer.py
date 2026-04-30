from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer as BaseDetectionTrainer
from ultralytics.models import yolo
from ultralytics.utils import RANK

from .detection_model import DetectionModel


# THS, Copied from ultralytics.models.yolo.detect.DetectionTrainer
# THS, Copied from ultralytics.engine.trainer.BaseTrainer


class DetectionTrainer(BaseDetectionTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "cls_feat_loss", "tmp_num_preds", "tmp_num_good_preds", "tmp_num_targets"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
