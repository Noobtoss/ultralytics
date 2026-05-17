import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers
from ultralytics.utils.metrics import bbox_iou


class UnpackReducer(reducers.BaseReducer):
    def element_reduction(self, losses, loss_indices, embeddings, labels):
        sorted_indices = torch.argsort(loss_indices)
        return losses[sorted_indices]


class NormalizeFeats(nn.Module):
    """Wraps any embedding loss and L2-normalizes embeddings before forwarding."""

    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, feats, *args, **kwargs):
        return self.loss(F.normalize(feats, dim=1), *args, **kwargs)


class ClsFeatLossFactory:
    @staticmethod
    def get(loss: str = None, **kwargs):
        if loss is None or loss == "None":
            return None
        elif loss == "sup_con_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#supconloss
            kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(losses.SupConLoss).parameters}
            return NormalizeFeats(losses.SupConLoss(**kwargs, reducer=UnpackReducer()))
        elif loss == "general_lifted_struct":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#generalizedliftedstructureloss
            kwargs = {k: v for k, v in kwargs.items() if
                      k in inspect.signature(losses.GeneralizedLiftedStructureLoss).parameters}
            return NormalizeFeats(losses.GeneralizedLiftedStructureLoss(**kwargs, reducer=UnpackReducer()))
        else:
            raise ValueError(f"Unknown feat loss type: '{loss}'")


class TALAlignWeight:
    def __init__(self, alpha: float = 1.0, beta: float = 6.0, **kwargs):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, pred_scores, target_scores, pred_bboxes, target_bboxes):
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=True).squeeze()
        pred_scores = pred_scores[torch.arange(len(pred_scores)), target_scores.argmax(-1)].sigmoid()
        align_metric = pred_scores.pow(self.alpha) * iou.pow(self.beta)
        return align_metric


class ConfWeight:
    def __call__(self, pred_scores, target_scores, pred_bboxes, target_bboxes):
        return pred_scores.sigmoid().max(-1).values


class WeightFactory:
    @staticmethod
    def get(weight: str = None, **kwargs):
        if weight is None or weight == "None":
            return None
        elif weight == "tal":
            # https://arxiv.org/abs/2108.07755
            return TALAlignWeight(**kwargs)
        elif weight == "conf":
            return ConfWeight()
        else:
            raise ValueError(f"Unknown weight type: '{weight}'")


class ConfFilter:
    def __init__(self, top_rel: float = 0.1, min_abs: float = None, **kwargs):
        self.top_rel = top_rel
        self.min_abs = min_abs

    def __call__(self, pred_scores, target_scores, pred_bboxes, target_bboxes):
        conf = pred_scores.sigmoid().max(-1).values

        mask = torch.ones(len(conf), dtype=torch.bool, device=conf.device)

        if self.min_abs is not None:
            mask = mask & (conf >= self.min_abs)

        if self.top_rel is not None:
            k = max(1, int(len(conf) * self.top_rel))
            thresh = conf.topk(k).values[-1]
            mask = mask & (conf >= thresh)

        return mask


class MaskFactory:
    @staticmethod
    def get(mask: str = None, **kwargs):
        if mask is None or mask == "None":
            return None
        elif mask == "tal":
            raise NotImplementedError
        elif mask == "conf":
            return ConfFilter(**kwargs)
        else:
            raise ValueError(f"Unknown mask type: '{mask}'")


class ClsFeatLoss(nn.Module):
    def __init__(self, loss: str, mask: str = None, weight: str = None, **kwargs):
        super().__init__()
        self.feat_loss = ClsFeatLossFactory.get(loss, **kwargs)
        self.mask = MaskFactory.get(mask, **kwargs)
        self.weight = WeightFactory.get(weight, **kwargs)

    def forward(
            self,
            cls_feats: torch.Tensor,
            pred_scores: torch.Tensor,
            target_scores: torch.Tensor,
            pred_bboxes: torch.Tensor,
            target_bboxes: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros(1, device=cls_feats.device)
        target_cls = target_scores.max(-1).indices

        if self.mask is not None:
            mask = self.mask(cls_feats, target_cls, pred_bboxes, target_bboxes)
            if not mask.sum():
                return loss
            cls_feats = cls_feats[mask]
            pred_scores = pred_scores[mask]
            target_scores = target_scores[mask]
            pred_bboxes = pred_bboxes[mask]
            target_bboxes = target_bboxes[mask]

        loss_per_element = self.feat_loss(cls_feats, target_cls)

        if self.weight is not None:
            weight = self.weight(pred_scores, target_scores, pred_bboxes, target_bboxes)
            weight = weight / weight.sum()
            loss += (loss_per_element * weight).sum()
        else:
            loss += loss_per_element.mean()

        return loss
