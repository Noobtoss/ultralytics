import torch
import torch.nn as nn
from ultralytics.nn.modules.head import Detect as BaseDetect
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.utils import LOGGER


# THS, Copied from ultralytics.nn.modules.head


class ClsFeatsDetect(BaseDetect):
    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()) -> None:
        LOGGER.warning("FeatsReturnDetect __init__ called")

        super().__init__(nc, reg_max, end2end, ch)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        # >>> MOD
        c3 *=2
        # <<< MOD
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)  # k=1
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),  # k=1
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),  # k=1
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )

    def forward_head(
            self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        # scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
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
        print("-------------------------------------------------------------------------------------------------------")
        for i in range(self.nl):
            h = x[i]
            for layer in list(cls_head[i])[:-1]:
                h = layer(h)
            cls_feats_i = h.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                scores_i = cls_head[i][-1](cls_feats_i).view(bs, self.nc, -1)
                h, w = x[i].shape[2], x[i].shape[3]
                for hi in range(h):
                    for wi in range(w):
                        if cls_feats_i.grad is not None:
                            cls_feats_i.grad.zero_()
                        scores_i[0, 0, hi * w + wi].backward(retain_graph=True)
                        affected = cls_feats_i.grad[0, 0, :, :].nonzero(as_tuple=False)
                        print(f"scale {i} input ({hi},{wi}) affects outputs: {affected.tolist()}")
        raise SystemExit("Insanity check done")
        """

        scores = []
        cls_feats = []
        for i in range(self.nl):
            h = x[i]
            for layer in list(cls_head[i])[:-1]:
                h = layer(h)
            cls_feats_i = h
            score_i = cls_head[i][-1](h).view(bs, self.nc, -1)  # conv → (bs, nc, H, W)
            scores.append(score_i)
            cls_feats.append(cls_feats_i)
        scores = torch.cat(scores, dim=-1)

        return dict(boxes=boxes, scores=scores, feats=x, cls_feats=cls_feats)
        # <<< MOD
