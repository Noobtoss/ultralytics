from typing import Any
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import bbox_iou
from ultralytics.models.utils.loss import RTDETRDetectionLoss as _RTDETRDetectionLoss
from ultralytics.models.utils.loss import DETRLoss as _DETRLoss

from .cls_feat_loss import ClsFeatLoss


class DETRLoss(_DETRLoss):
    def __init__(self, *args, model, **kwargs) -> None:
        LOGGER.warning("[Modded] DETRLoss")
        super().__init__(*args, **kwargs)
        hyp = model.args  # hyperparameters
        self.loss_gain["cls_feat"] = getattr(hyp, "cls_feat", 0)
        cls_feat_kwargs = {
            k.removeprefix("cls_feat_"): v
            for k, v in vars(hyp).items()
            if k.startswith("cls_feat_")
        }
        self.cls_feat_loss = ClsFeatLoss(**cls_feat_kwargs).to(self.device)
        n = getattr(hyp, "cls_feat_dec_layers", None)  # hard encoding 6 is bad
        assert n != 0
        self.cls_feat_dec_layers = range(5 - n, 5) if n is not None else range(0, 5)

    def _get_loss_cls_feat(
        self,
        cls_feats: torch.Tensor,
        pred_scores: torch.Tensor,
        targets: torch.Tensor,
        gt_scores: torch.Tensor,
        num_gts: int,
        postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        name_class = f"loss_cls_feat{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        cls_feats = cls_feats.reshape(bs * nq, -1)  # [bs*nq, feats]
        pred_scores = pred_scores.reshape(bs * nq, -1)  # [bs*nq, nc]
        gt_scores = gt_scores.reshape(bs * nq, -1)  # [bs*nq, nc]

        fg_mask = targets != self.nc
        fg_mask = fg_mask.view(-1)
        cls_feats = cls_feats[fg_mask]
        pred_scores = pred_scores[fg_mask]
        gt_scores = gt_scores[fg_mask]

        loss_cls_feat = self.cls_feat_loss(cls_feats=cls_feats, target_scores=gt_scores, pred_scores=pred_scores)
        # loss_cls_feat uses reduction="mean" over all elements (bs * nq * feats).
        # _get_loss_cls applies .mean(1).sum() over (bs * nq, nc+1), making loss_cls ~ (bs * nq) times larger.
        # Scale loss_cls_feat by (bs * nq) to match loss_cls magnitude might be needed.

        return {name_class: loss_cls_feat.squeeze() * self.loss_gain["cls_feat"]}

    # >>> MOD
    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
        cls_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
    # <<< MOD
        """Calculate losses for a single prediction layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            pred_scores (torch.Tensor): Predicted class scores.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.
            postfix (str, optional): String to append to loss names.
            match_indices (list[tuple], optional): Pre-computed matching indices.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of losses.
        """
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        # >>> MOD
        loss = {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
        }
        if cls_feats is not None:
            loss.update(
                self._get_loss_cls_feat(cls_feats, pred_scores, targets, gt_scores, len(gt_bboxes), postfix)
            )
        return loss
        # <<< MOD

    # >>> MOD
    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        match_indices: list[tuple] | None = None,
        postfix: str = "",
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        cls_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
    # <<< MOD
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        # >>> MOD
        loss = torch.cat([loss, torch.zeros(1, device=pred_bboxes.device)])
        # <<< MOD
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        # >>> MOD
        for i, (aux_bboxes, aux_scores, aux_cls_feats) in enumerate(zip(pred_bboxes, pred_scores, cls_feats)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
                cls_feats=aux_cls_feats if i in self.cls_feat_dec_layers else None,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']
            loss[-1] += loss_.get(f"loss_cls_feat{postfix}", torch.tensor(0.0))

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
            f"loss_cls_feat_aux{postfix}": loss[-1],
        }
        # <<< MOD
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    # >>> MOD
    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        cls_feats: torch.Tensor = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
    # <<< MOD
        """Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores, shape (L, B, N, C).
            batch (dict[str, Any]): Batch information containing cls, bboxes, and gt_groups.
            postfix (str, optional): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict[str, torch.Tensor]): Computed losses, including main and auxiliary (if enabled).

        Notes:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        # >>> MOD
        total_loss = self._get_loss(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, postfix=postfix, match_indices=match_indices, cls_feats=cls_feats[-1]
        )
        # <<< MOD

        if self.aux_loss:
            # >>> MOD
            total_loss.update(
                self._get_loss_aux(
                    pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix, cls_feats=cls_feats[:-1]
                )
            )
            # <<< MOD

        return total_loss

_RTDETRDetectionLoss.__bases__ = (DETRLoss,)


class RTDETRDetectionLoss(_RTDETRDetectionLoss):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("[Modded] RTDETRDetectionLoss")
        super().__init__(*args, **kwargs)

    # >>> MOD
    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
        dn_cls_feats: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
    # <<< MOD
        """Forward pass to compute detection loss with optional denoising loss.

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): Tuple containing predicted bounding boxes and scores.
            batch (dict[str, Any]): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes.
            dn_scores (torch.Tensor, optional): Denoising scores.
            dn_meta (dict[str, Any], optional): Metadata for denoising.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing total loss and denoising loss if applicable.
        """
        # >>> MOD
        pred_bboxes, pred_scores, cls_feats = preds
        total_loss = DETRLoss.forward(self, pred_bboxes, pred_scores, batch, cls_feats=cls_feats)
        # <<< MOD

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            # Compute the denoising training loss
            # >>> MOD
            dn_loss = DETRLoss.forward(self, dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices, cls_feats=dn_cls_feats)
            # <<< MOD
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss.keys()})

        return total_loss
