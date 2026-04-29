import torch
from ultralytics.nn.modules.head import Detect as BaseDetect
from ultralytics.utils import LOGGER


# THS, Copied from ultralytics.nn.modules.head


class ClsFeatsDetect(BaseDetect):
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
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        # >>> MOD
        """
        # Insanity check
        for i in range(self.nl):
            x_i = x[i].detach().clone().requires_grad_(True)
            with torch.enable_grad():
                scores_i = cls_head[i](x_i).view(bs, self.nc, -1)
                h, w = x[i].shape[2], x[i].shape[3]
                for hi in range(h):
                    for wi in range(w):
                        if x_i.grad is not None:
                            x_i.grad.zero_()
                        scores_i[0, 0, hi * w + wi].backward(retain_graph=True)
                        affected = x_i.grad[0, 0, :, :].nonzero(as_tuple=False)
                        print(f"scale {i} input ({hi},{wi}) affects outputs: {affected.tolist()}")
        raise SystemExit("Insanity check done")
        """
        # <<< MOD
        """
        # >>> MOD
        scores = []
        cls_feats_raw = []
        cls_feats = []
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
            cls_feats_raw.append(cls_feats_raw_i)
            cls_feats.append(cls_feats_i)

        scores = torch.cat(scores, dim=-1)
        cls_feats = torch.cat(cls_feats, dim=-1)  # cls_feats.shape can be == scores.shape, do not be alarmed
        return dict(boxes=boxes, scores=scores, feats=x, cls_feats=cls_feats, cls_feats_raw=cls_feats_raw)
        # <<< MOD        
        """
        return dict(boxes=boxes, scores=scores, feats=x)
