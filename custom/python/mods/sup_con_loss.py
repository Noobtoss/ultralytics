import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/google-research/google-research/blob/master/supcon/losses.py#L99
# https://github.com/HobbitLong/SupContrast/blob/master/losses.py
# https://github.com/GuillaumeErhard/Supervised_contrastive_loss_pytorch/blob/main/loss/spc.py
# https://github.com/huggingface/sentence-transformers/tree/master/sentence_transformers/losses


def divide_no_nan(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    PyTorch equivalent of tf.math.divide_no_nan.
    Returns x / y, with zeros where y == 0.
    Differentiable and safe for autograd.
    """
    y_safe = torch.where(y == 0, torch.ones_like(y), y)
    result = x / y_safe
    result = torch.where(y == 0, torch.zeros_like(result), result)
    return result


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning:
        https://arxiv.org/abs/2004.11362

        :param temperature: float — controls sharpness of the contrastive distribution.
            Lower values → sharper, harder contrast (sensitive to hardest negatives).
            Higher values → softer, more uniform distribution over negatives.

            Rough guidelines:
                batch_size ~32,  n_classes ~5-10  → 0.07 (paper default)
                batch_size ~128, n_classes ~10-20 → 0.1
                batch_size ~300, n_classes ~30    → 0.1 - 0.2
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, shape [batch_size] — per-sample loss (0 where no positives)
        """
        projections = F.normalize(projections, dim=1)  # L2 normalizes

        device = projections.device

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        exp_dot_tempered = torch.exp(
            dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0].detach()
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(
            exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True) + 1e-8)
        )
        loss_per_sample = divide_no_nan(
            torch.sum(log_prob * mask_combined, dim=1), cardinality_per_samples
        )

        return loss_per_sample
