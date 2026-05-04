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


class FeatLossFactory:
    @staticmethod
    def get(loss: str = None, **kwargs):
        if loss is None:
            return None
        if loss == "sup_con_loss":
            kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(losses.SupConLoss).parameters}
            return NormalizeFeats(losses.SupConLoss(**kwargs, reducer=UnpackReducer()))
        if loss == "general_lifted_struct":
            kwargs = {k: v for k, v in kwargs.items() if k in inspect.signature(losses.GeneralizedLiftedStructureLoss).parameters}
            return NormalizeFeats(losses.GeneralizedLiftedStructureLoss(**kwargs, reducer=UnpackReducer()))
        else:
            raise ValueError(f"Unknown cls_feat loss type: '{loss}'")
