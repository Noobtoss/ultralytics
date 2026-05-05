import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers


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
        if loss is None:
            return None
        if loss == "sup_con_loss":
            kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(losses.SupConLoss).parameters}
            return NormalizeFeats(losses.SupConLoss(**kwargs, reducer=UnpackReducer()))
        if loss == "general_lifted_struct":
            kwargs = {k: v for k, v in kwargs.items() if
                      k in inspect.signature(losses.GeneralizedLiftedStructureLoss).parameters}
            return NormalizeFeats(losses.GeneralizedLiftedStructureLoss(**kwargs, reducer=UnpackReducer()))
        else:
            raise ValueError(f"Unknown feat loss type: '{loss}'")


class ClsFeatLoss(nn.Module):
    def __init__(self, loss: str, conf_thresh: float = 0.2, iou_thresh: float = 0.5, **kwargs):
        super().__init__()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.feat_loss = ClsFeatLossFactory.get(loss, **kwargs)

    def forward(
            self,
            cls_feats: torch.Tensor,
            target_scores: torch.Tensor,
            pred_scores: torch.Tensor,
            fg_mask: torch.Tensor,
    ) -> torch.Tensor:
        cls_feats = cls_feats[fg_mask]
        target_scores = target_scores[fg_mask]
        # pred_scores = pred_scores[fg_mask]

        loss = torch.zeros(1, device=cls_feats.device)  # tmp

        target_cls = target_scores.max(-1).indices
        # pred_conf = pred_scores.detach().sigmoid().max(-1).values

        # """
        # raw
        loss += self.feat_loss(cls_feats, target_cls).mean()
        # """
        """
        # masking
        pred_conf_mask = pred_conf > self.conf_thresh
        if pred_conf_mask.sum():
            loss += self.feat_loss(cls_feats[pred_conf_mask], target_cls[pred_conf_mask]).mean()
        """
        # """
        # weighting
        # loss += sum(self.feat_loss(cls_feats, target_cls) * pred_conf) / pred_conf.sum()
        # """

        return loss
