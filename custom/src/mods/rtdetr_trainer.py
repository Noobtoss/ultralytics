from copy import copy
from ultralytics.utils import RANK, LOGGER
from ultralytics.models.rtdetr.train import RTDETRTrainer as _RTDETRTrainer
from .coco_dataset import build_coco_dataset

from .rtdetr_detection_model import RTDETRDetectionModel
from .rtdetr_validator import RTDETRValidator
from .cls_feat_proj_head import ClsFeatProjHead
from .cls_feat_scheduler import ClsFeatScheduler


class RTDETRTrainer(_RTDETRTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] RTDETRTrainer")
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        if "train_ann" in self.data:
            return build_coco_dataset(self, img_path, mode, batch, rtdetr=True)
        return super().build_dataset(img_path, mode, batch)

    def _setup_train(self):
        super()._setup_train()
        if getattr(self.model, "criterion", None) is None:  # Default done in BaseModel forward
            self.model.criterion = self.model.init_criterion()
        if hasattr(self.args, "cls_feat_scheduler"):
            self.add_callback("on_train_epoch_start", ClsFeatScheduler(self).on_train_epoch_start)
            self.add_callback("on_train_epoch_end", ClsFeatScheduler(self).on_train_epoch_end)

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if hasattr(self.args, "cls_feat_proj_head"):
            LOGGER.warning("[Modded] adding cls_feat_proj_head")
            kwargs = {
                k.removeprefix("cls_feat_"): v
                for k, v in vars(self.args).items()
                if k.startswith("cls_feat_")
            }
            kwargs['dim'] = model.model[-1].dec_score_head[0][-1].in_features
            model.cls_feat_proj_head = ClsFeatProjHead(**kwargs) if kwargs["proj_head"] is not None else None
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return an RTDETRValidator suitable for RT-DETR model validation."""
        self.loss_names = "giou_loss", "cls_loss", "l1_loss", "cls_feat_loss"
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
