from copy import copy
from ultralytics.engine.trainer import BaseTrainer as _BaseTrainer
from ultralytics.models.yolo.detect import DetectionTrainer as _DetectionTrainer
from ultralytics.utils import RANK, LOGGER
from ultralytics.utils.torch_utils import unwrap_model

from .detection_model import DetectionModel
from .detection_validator import DetectionValidator
from .cls_feat_proj_head import ClsFeatProjHeadFactory
from .cls_feat_scheduler import ClsFeatCallback


class BaseTrainer(_BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] BaseTrainer")
        super().__init__(*args, **kwargs)


_DetectionTrainer.__bases__ = (BaseTrainer,)


class DetectionTrainer(_DetectionTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DetectionTrainer")
        super().__init__(*args, **kwargs)

    def _setup_train(self):
        super()._setup_train()
        if hasattr(self.args, "cls_feat_scheduler"):
            self.add_callback("on_train_epoch_start", ClsFeatCallback(self))

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if hasattr(self.args, "cls_feat_proj_head"):
            kwargs = {
                k.removeprefix("cls_feat_"): v
                for k, v in vars(self.args).items()
                if k.startswith("cls_feat_")
            }
            # kwargs['dim'] = model.model[-1].cv3[0][-2][-1].conv.out_channels
            kwargs['dim'] = model.model[-1].cv3[0][-1].in_channels
            model.cls_feat_proj_head = ClsFeatProjHeadFactory.get(**kwargs)
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
        cls_feat_proj_head_params = [
            v for k, v in unwrap_model(model).named_parameters()
            if v.requires_grad and k.startswith("cls_feat_proj_head")
        ]

        if cls_feat_proj_head_params:
            lr = getattr(self.args, "cls_feat_proj_head_lr", None) or optimizer.param_groups[0]['lr']

            cls_feat_proj_head_ids = {id(p) for p in cls_feat_proj_head_params}
            for group in optimizer.param_groups:
                group['params'] = [p for p in group['params'] if id(p) not in cls_feat_proj_head_ids]
            optimizer.add_param_group({
                "params": cls_feat_proj_head_params,
                "lr": lr,
                "initial_lr": lr,
            })
            LOGGER.warning(f"[Modded] Moved cls_feat_proj_head to new group lr={lr}")

        return optimizer
