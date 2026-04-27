import torch
from ultralytics.nn.modules.head import Detect as BaseDetect
from ultralytics.utils import LOGGER


# THS, Copied from ultralytics.nn.modules.head


class ClsFeatsReturnDetect(BaseDetect):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("FeatsReturnDetect __init__ called")

        super().__init__(*args, **kwargs)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        # >>> MOD
        scores = []
        cls_embs_raw = []
        cls_embs = []
        for i in range(self.nl):
            cls_feats_raw_i = x[i]
            cls_feats_i = cls_head[i][0](cls_feats_raw_i)

            h = cls_feats_i
            for layer in list(cls_head[i])[1:]:
                h = layer(h)

            score_i = h.view(bs, self.nc, -1)  # conv → (bs, nc, H, W)
            cls_feats_raw_i = cls_feats_raw_i.view(bs, cls_feats_raw_i.shape[1], -1)
            cls_feats_i = cls_feats_i.view(bs, cls_feats_i.shape[1], -1)

            scores.append(score_i)
            cls_embs_raw.append(cls_feats_raw_i)
            cls_embs.append(cls_feats_i)

        scores = torch.cat(scores, dim=-1)
        cls_embs = torch.cat(cls_embs, dim=-1)  # cls_embs.shape == scores.shape do not be alarmed
        return dict(boxes=boxes, scores=scores, feats=x, cls_embs=cls_embs, cls_embs_raw=cls_embs_raw)
        # <<< MOD
