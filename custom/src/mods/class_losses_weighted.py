import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.loss import FocalLoss, VarifocalLoss

class ClassLossWeighted(nn.Module):
    def __init__(self,
                 loss: nn.Module = nn.BCEWithLogitsLoss(reduction="none"),
                 class_weights: torch.Tensor = None,
                 class_weights_matrix: torch.Tensor = None
                 ) -> None:
        super().__init__()
        self.loss = loss

        self.register_buffer("class_weights", class_weights)
        self.register_buffer("class_weights_matrix", class_weights_matrix)

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        loss = self.loss(pred_scores, target_scores, *args, **kwargs)
        if self.class_weights is not None:
            loss = loss * self.class_weights
        if self.class_weights_matrix is not None:
            labels = target_scores.argmax(dim=-1)
            weight_per_sample = self.class_weights_matrix[labels]
            loss = loss * weight_per_sample
        return loss


class VarifocalLossWeighted(VarifocalLoss):
    def __init__(self,
                 gamma: float = 2.0,
                 alpha: float = 0.75,
                 class_weights: torch.Tensor = None,
                 class_weights_matrix: torch.Tensor = None
                 ):
        super().__init__(gamma, alpha)
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("class_weights_matrix", class_weights_matrix)

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            # >>> MOD
            loss = F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight
            if self.class_weights is not None:
                loss = loss * self.class_weights
            if self.class_weights_matrix is not None:
                gt_cls = label.argmax(dim=-1)
                weight_per_sample = self.class_weights_matrix[gt_cls]
                loss = loss * weight_per_sample
            loss = loss.mean(1).sum()
            # <<< MOD
        return loss


class FocalLossWeighted(FocalLoss):
    def __init__(self,
                 gamma: float = 1.5,
                 alpha: float = 0.25,
                 class_weights: torch.Tensor = None,
                 class_weights_matrix: torch.Tensor = None
                 ):
        super().__init__(gamma, alpha)
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("class_weights_matrix", class_weights_matrix)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        # >>> MOD
        if self.class_weights is not None:
            loss = loss * self.class_weights
        if self.class_weights_matrix is not None:
            gt_cls = label.argmax(dim=-1)
            weight_per_sample = self.class_weights_matrix[gt_cls]
            loss = loss * weight_per_sample
        # <<< MOD
        return loss.mean(1).sum()
