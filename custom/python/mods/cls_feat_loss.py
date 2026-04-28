import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers


class _SupConLossWithNorm(nn.Module):
    def __init__(self, temperature: float):
        super().__init__()
        self._loss = losses.SupConLoss(temperature=temperature, reducer=reducers.SumReducer())

    def forward(self, emb, labels):
        return self._loss(F.normalize(emb, dim=1), labels)


class ClsFeatLossFactory:
    @staticmethod
    def get(loss: str, hyp):
        if loss is None:
            return None
        if loss == "sup_con_loss":
            temperature = getattr(hyp, "cls_feat_loss_temp", 0.07)
            return _SupConLossWithNorm(temperature)
        else:
            raise ValueError(f"Unknown cls_feat loss type: '{loss}'")
