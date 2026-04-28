import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors
from ultralytics.utils import LOGGER

from .sup_con_loss import SupConLoss


# THS, Copied from ultralytics.utils.loss


class ClsEmbLossFactory:
    @staticmethod
    def get(loss: str, hyp):
        if loss is None:
            LOGGER.warning("No cls_emb loss type specified, cls_emb loss disabled.")
            return None
        if loss == "sup_con_loss":
            kwargs = {}
            if hasattr(hyp, "cls_emb_temp"):
                kwargs["temperature"] = hyp.cls_emb_temp
            return SupConLoss(**kwargs)
        else:
            raise ValueError(f"Unknown cls_emb loss type: '{loss}'")


class TrainLoss(v8DetectionLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cls_emb_loss = ClsEmbLossFactory.get(getattr(self.hyp, "cls_emb_loss", None), self.hyp)
        self.hyp.cls_emb = getattr(self.hyp, "cls_emb", None)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        # >>> MOD
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, emb
        pred_distri, pred_scores, cls_embeds = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["cls_embs"].permute(0, 2, 1).contiguous(),
        )
        # <<< MOD
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE
        # >>> MOD
        if self.cls_emb_loss is not None and self.hyp.cls_emb is not None:
            if fg_mask.sum():
                matched_cls_embeds = cls_embeds[fg_mask]
                matched_target_labels = target_scores[fg_mask].clone().argmax(dim=1)
                cls_emb_loss = self.cls_emb_loss(matched_cls_embeds, matched_target_labels)
                loss[3] = cls_emb_loss.sum() / target_scores_sum
            loss[3] *= self.hyp.cls_emb
        # <<< MOD
        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl, emb)
