import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers, miners
from ultralytics.utils.metrics import bbox_iou


class UnpackReducer(reducers.BaseReducer):
    def element_reduction(self, losses, loss_indices, embeddings, labels):
        sorted_indices = torch.argsort(loss_indices)
        return losses[sorted_indices]


class NormalizeEmbeddingsWrapper(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

    def forward(self, embeddings, *args, **kwargs):
        return self.loss(F.normalize(embeddings, dim=1), *args, **kwargs)

"""
class MinerWrapper(nn.Module):
    def __init__(self, miner, loss: nn.Module, **kwargs):
        super().__init__()
        self.miner = miner
        self.loss = loss

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        indices = self.miner(embeddings, labels)
        return self.loss(embeddings, labels, indices)
    
# https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss
params_miner = {
    "margin": 0.2,
    "type_of_triplets": "semihard",  # options: "all", "hard", "semihard", "easy"
}
params = {
    "margin": 0.2,
}
params_miner.update({k: v for k, v in kwargs.items() if k in inspect.signature(miners.TripletMarginMiner).parameters})
params.update({k: v for k, v in kwargs.items() if k in inspect.signature(losses.TripletMarginLoss).parameters})
return MinerWrapper(
    miner=miners.TripletMarginMiner(**params_miner),
    loss=losses.TripletMarginLoss(**params, reducer=UnpackReducer())
)
"""

class FeatLossFactory:
    @staticmethod
    def get(loss: str = None, **kwargs):
        if loss is None or loss == "None":
            return None
        elif loss == "sup_con_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#supconloss
            params = {
                "temperature": 0.07,
            }
            params.update({k: v for k, v in kwargs.items() if k in inspect.signature(losses.SupConLoss).parameters})
            return NormalizeEmbeddingsWrapper(losses.SupConLoss(**params, reducer=UnpackReducer()))

        elif loss == "circle_loss":
            # https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss
            params = {
                "m": 0.40,
                "gamma": 32,
            }
            params.update({k: v for k, v in kwargs.items() if k in inspect.signature(losses.CircleLoss).parameters})
            return NormalizeEmbeddingsWrapper(losses.CircleLoss(**params, reducer=UnpackReducer()))

        elif loss == "multi_sim_loss":
            params = {
                "alpha": 2.0,
                "beta": 20.0,
                "base": 0.5,
            }
            params.update({k: v for k, v in kwargs.items() if k in inspect.signature(losses.MultiSimilarityLoss).parameters})
            return NormalizeEmbeddingsWrapper(losses.MultiSimilarityLoss(**params, reducer=UnpackReducer()))

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
    def __init__(self, **kwargs):
        pass

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


class Masking:
    def __init__(self, mask_pct: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.mask_pct = mask_pct

    def _masking(self, metric):
        k = max(1, int(len(metric) * (1 - self.mask_pct)))
        thresh = metric.topk(k).values[-1]
        return metric >= thresh


class TALAlignMask(Masking, TALAlignWeight):
    def __call__(self, pred_scores, target_scores, pred_bboxes, target_bboxes):
        align_metric = super().__call__(pred_scores, target_scores, pred_bboxes, target_bboxes)
        return self._masking(align_metric)


class ConfMask(Masking, ConfWeight):
    def __call__(self, pred_scores, target_scores, pred_bboxes, target_bboxes):
        conf = super().__call__(pred_scores, target_scores, pred_bboxes, target_bboxes)
        return self._masking(conf)


class RandMask:
    def __init__(self, mask_pct: float = 0.4, **kwargs):
        self.mask_pct = mask_pct

    def __call__(self, pred_scores, target_scores):
        k = max(1, int(len(pred_scores) * self.mask_pct))
        mask = torch.zeros(len(pred_scores), dtype=torch.bool)
        indices = torch.randperm(len(pred_scores))[:k]
        mask[indices] = True
        return ~mask


class RandMaskBalanced:
    def __init__(self, mask_pct: float = 0.4, min_per_class: int = 4, **kwargs):
        self.mask_pct = mask_pct
        self.min_per_class = min_per_class

    def __call__(self, pred_scores, target_scores):
        target_cls = target_scores.max(-1).indices
        n = len(pred_scores)
        k = max(1, int(n * self.mask_pct))
        mask = torch.zeros(n, dtype=torch.bool)

        # Guarantee at least min_per_class per unique class
        for cls in target_cls.unique():
            cls_indices = (target_cls == cls).nonzero(as_tuple=True)[0]
            k_cls = min(self.min_per_class, len(cls_indices))
            chosen = cls_indices[torch.randperm(len(cls_indices))[:k_cls]]
            mask[chosen] = True

        # Fill remaining budget with random unchosen indices
        remaining = k - mask.sum().item()
        if remaining > 0:
            unmasked = (~mask).nonzero(as_tuple=True)[0]
            extra = unmasked[torch.randperm(len(unmasked))[:remaining]]
            mask[extra] = True

        return ~mask


class MaskFactory:
    @staticmethod
    def get(mask: str = None, **kwargs):
        if mask is None or mask == "None":
            return None
        elif mask == "conf":
            return ConfMask(**kwargs)
        elif mask == "rand":
            return RandMask(**kwargs)
        elif mask == "rand_balanced":
            return RandMaskBalanced(**kwargs)
        else:
            raise ValueError(f"Unknown mask type: '{mask}'")


class ClsFeatLoss(nn.Module):
    def __init__(self, loss: str, mask: str = None, weight: str = None, **kwargs):
        super().__init__()
        self.loss = FeatLossFactory.get(loss, **kwargs)
        self.mask = MaskFactory.get(mask, **kwargs)
        self.weight = WeightFactory.get(weight, **kwargs)

    def forward(
            self,
            cls_feats: torch.Tensor,
            pred_scores: torch.Tensor,
            target_scores: torch.Tensor,
            pred_bboxes: torch.Tensor,
            target_bboxes: torch.Tensor
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=cls_feats.device)
        target_cls = target_scores.max(-1).indices
        if self.mask is not None:
            mask = self.mask(cls_feats, target_scores, pred_bboxes, target_bboxes)
            if not mask.sum():
                return loss
            cls_feats = cls_feats[mask]
            target_cls = target_cls[mask]
            pred_scores = pred_scores[mask]
            target_scores = target_scores[mask]
            pred_bboxes = pred_bboxes[mask]
            target_bboxes = target_bboxes[mask]

        loss_per_element = self.loss(cls_feats, target_cls).squeeze(-1)

        if self.weight is not None:
            weight = self.weight(pred_scores, target_scores, pred_bboxes, target_bboxes)
            weight = weight / weight.sum()
            loss += (loss_per_element * weight).sum()
        else:
            loss += loss_per_element.mean()

        return loss
