import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils.torch_utils import autocast


class BCEWithLogitsLossWeighted(nn.Module):
    def __init__(self,
                 class_weights: torch.Tensor = None,
                 class_weights_matrix: torch.Tensor = None
                 ) -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction="none")

        self.register_buffer("class_weights", class_weights)
        self.register_buffer("class_weights_matrix", class_weights_matrix)

    def forward(self, pred_scores: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        loss = self.loss(pred_scores, target_scores)
        loss = loss.view(-1, loss.shape[-1])
        if self.class_weights is not None:
            loss = loss * self.class_weights
        if self.class_weights_matrix is not None:
            target_scores = target_scores.view(-1, target_scores.shape[-1])
            gt_cls = target_scores.argmax(dim=-1)
            weight_per_sample = self.class_weights_matrix[gt_cls]
            loss = loss * weight_per_sample
        return loss


class VarifocalLossWeighted(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: float = 0.75,
                 class_weights: torch.Tensor = None,
                 class_weights_matrix: torch.Tensor = None
                 ):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        # >>> MOD
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("class_weights_matrix", class_weights_matrix)
        # <<< MOD

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            # >>> MOD
            loss = F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight
            loss = loss.view(-1, loss.shape[-1])
            label = label.view(-1, label.shape[-1])
            if self.class_weights is not None:
                loss = loss * self.class_weights
            if self.class_weights_matrix is not None:
                gt_cls = label.argmax(dim=-1)
                weight_per_sample = self.class_weights_matrix[gt_cls]
                loss = loss * weight_per_sample
            loss = loss.mean(1).sum()
            # <<< MOD
        return loss


class FocalLossWeighted(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """
    def __init__(self,
                 gamma: float = 1.5,
                 alpha: float = 0.25,
                 class_weights: torch.Tensor = None,
                 class_weights_matrix: torch.Tensor = None
                 ):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
        # >>> MOD
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("class_weights_matrix", class_weights_matrix)
        # <<< MOD

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
        loss = loss.view(-1, loss.shape[-1])
        label = label.view(-1, label.shape[-1])
        if self.class_weights is not None:
            loss = loss * self.class_weights
        if self.class_weights_matrix is not None:
            gt_cls = label.argmax(dim=-1)
            weight_per_sample = self.class_weights_matrix[gt_cls]
            loss = loss * weight_per_sample
        # <<< MOD
        return loss.mean(1).sum()
