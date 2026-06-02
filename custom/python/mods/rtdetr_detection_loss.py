from typing import Any
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import bbox_iou
from ultralytics.models.utils.loss import RTDETRDetectionLoss as _RTDETRDetectionLoss
from ultralytics.models.utils.loss import DETRLoss as _DETRLoss


class DETRLoss(_DETRLoss):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DETRLoss")
        super().__init__(*args, **kwargs)


_RTDETRDetectionLoss.__bases__ = (DETRLoss,)


class RTDETRDetectionLoss(_RTDETRDetectionLoss):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("[Modded] RTDETRDetectionLoss")
        super().__init__(*args, **kwargs)

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        dn_cls_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_bboxes, pred_scores, cls_feats = preds
        total_loss = DETRLoss.forward(self, pred_bboxes, pred_scores, cls_feats, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            dn_loss = DETRLoss.forward(self, dn_bboxes, dn_scores, dn_cls_feats, batch, postfix="_dn", match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss
