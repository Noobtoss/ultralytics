from ultralytics.utils import RANK, LOGGER
from ultralytics.models.rtdetr.train import RTDETRTrainer as _RTDETRTrainer

from .rtdetr_detection_model import RTDETRDetectionModel
from .cls_feat_proj_heads import ClsFeatProjHeadFactory


class RTDETRTrainer(_RTDETRTrainer):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] RTDETRTrainer")
        super().__init__(*args, **kwargs)

    def get_model(self, cfg: dict | None = None, weights: str | None = None, verbose: bool = True):
        model = RTDETRDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if hasattr(self.args, "cls_feat_proj_head"):
            cls_feat_kwargs = {
                k.removeprefix("cls_feat_"): v
                for k, v in vars(self.args).items()
                if k.startswith("cls_feat_")
            }
            cls_feat_kwargs['dim'] = model.model[-1].dec_score_head[0][-1].in_features
            model.cls_feat_proj_head = ClsFeatProjHeadFactory.get(**cls_feat_kwargs)
        if weights:
            model.load(weights)
        return model
