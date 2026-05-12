from copy import copy
from ultralytics.engine.trainer import BaseTrainer as _BaseTrainer
from ultralytics.models.yolo.detect import DetectionTrainer as _DetectionTrainer
from ultralytics.utils import RANK, LOGGER
from ultralytics.utils.torch_utils import unwrap_model

from .detection_model import DetectionModel
from .proj_heads import ProjHeadFactory
from .detection_validator import DetectionValidator


class BaseTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] BaseTrainer")
        super().__init__(*args, **kwargs)


_DetectionTrainer.__bases__ = (BaseTrainer,)


class DetectionTrainer(_DetectionTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DetectionTrainer")
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if hasattr(self.args, "cls_feat_proj_head"):
            kwargs = {k[len("cls_feat_"):]: v for k, v in vars(self.args).items() if k.startswith("cls_feat_")}
            kwargs['nl'] = model.model[-1].nl
            kwargs['in_channels'] = model.model[-1].cv3[0][-2][-1].conv.out_channels
            model.cls_feat_proj_head = ProjHeadFactory.get(**kwargs)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "cls_feat_loss"
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        optimizer = super().build_optimizer(model, name=name, lr=lr, momentum=momentum, decay=decay,
                                            iterations=iterations)
        if hasattr(self.args, "cls_feat_proj_head_lr"):
            proj_head_param_ids = set()
            for k, v in unwrap_model(model).named_parameters():
                if not v.requires_grad:
                    continue
                if k.startswith("cls_feat_proj_head"):
                    proj_head_param_ids.add(id(v))

            if proj_head_param_ids:
                for group in optimizer.param_groups:
                    group_params = group.get('params', [])
                    if any(id(p) in proj_head_param_ids for p in group_params):
                        group['lr'] = self.args.cls_feat_proj_head_lr
                        LOGGER.info(f"[Modded] cls_feat_proj_head param group lr to {group['lr']}")
        return optimizer
