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

class TALAlignWeighting:
    def __init__(self, alpha: float = 1.0, beta: float = 6.0, **kwargs):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, pred_scores, target_scores, pred_bboxes, target_bboxes):
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=True).squeeze()
        pred_scores = pred_scores[torch.arange(len(pred_scores)), target_scores.argmax(-1)].sigmoid()
        align_metric = pred_scores.pow(self.alpha) * iou.pow(self.beta)
        return align_metric


class ConfWeighting:
    def __call__(self, pred_scores, target_scores, pred_bboxes, target_bboxes):
        return pred_scores.sigmoid().max(-1).values


class WeightingFactory:
    @staticmethod
    def get(weighting: str = None, **kwargs):
        if weighting is None or weighting == "None":
            return None
        elif weighting == "tal":
            # https://arxiv.org/abs/2108.07755
            return TALAlignWeighting(**kwargs)
        elif weighting == "conf":
            return ConfWeighting()
        else:
            raise ValueError(f"Unknown weighting type: '{weighting}'")


class ClsFeatLoss(nn.Module):
    def __init__(self, loss: str, weighting: str = None, **kwargs):
        super().__init__()
        self.feat_loss = ClsFeatLossFactory.get(loss, **kwargs)
        self.weighting = WeightingFactory.get(weighting, **kwargs)

    def forward(
            self,
            cls_feats: torch.Tensor,
            pred_scores: torch.Tensor,
            target_scores: torch.Tensor,
            pred_bboxes: torch.Tensor,
            target_bboxes: torch.Tensor) -> torch.Tensor:

        target_cls = target_scores.max(-1).indices

        loss = self.feat_loss(cls_feats, target_cls)
        if self.weighting is not None:
            weighting = self.weighting(pred_scores, target_scores, pred_bboxes, target_bboxes)
            loss = loss * weighting / weighting.sum()

        return loss.mean()
