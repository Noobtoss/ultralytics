import torch

from ultralytics.utils import LOGGER
from ultralytics.nn.tasks import RTDETRDetectionModel as _RTDETRDetectionModel


class RTDETRDetectionModel(_RTDETRDetectionModel):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] RTDETRDetectionModel")
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from .rtdetr_detection_loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True, model=self)

    def loss(self, batch, preds=None):
        """Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (tuple, optional): Precomputed model predictions.

        Returns:
            (torch.Tensor): Total loss value.
            (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        # >>> MOD
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta, dec_cls_feats = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores, dec_cls_feats = None, None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
            dn_cls_feats, dec_cls_feats = torch.split(dec_cls_feats, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        # slot 0 is a zero vector (no encoder cls features by design); keeps indexing symmetric with dec_scores
        dec_cls_feats = torch.cat(
            [torch.zeros(1, *dec_cls_feats.shape[1:], device=dec_cls_feats.device, dtype=dec_cls_feats.dtype),
             dec_cls_feats], dim=0)  # (7, bs, 300, 4)

        loss = self.criterion(
            (dec_bboxes, dec_scores, dec_cls_feats),
            targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta, dn_cls_feats=dn_cls_feats,
        )
        # <<< MOD
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )
