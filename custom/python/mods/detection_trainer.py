from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer as BaseDetectionTrainer
from ultralytics.models import yolo
from ultralytics.utils import RANK

from .detection_model import DetectionModel
from .proj_heads import ProjHeadFactory


# THS, Copied from ultralytics.models.yolo.detect.DetectionTrainer
# THS, Copied from ultralytics.engine.trainer.BaseTrainer


class DetectionTrainer(BaseDetectionTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        kwargs = {k[len("cls_feat_"):]: v for k, v in vars(self.args).items() if k.startswith("cls_feat_")}
        kwargs['nl'] = model.model[-1].nl
        kwargs['in_channels'] = model.model[-1].cv3[0][-2][-1].conv.out_channels
        model.model[-1].cls_feat_proj_head = ProjHeadFactory.get(**kwargs)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "cls_feat_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
