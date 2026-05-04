import torch
import torch.nn as nn
from ultralytics.engine.model import Model as BaseModel
from ultralytics.utils import LOGGER

class Model(BaseModel):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("Model __init__ called")
        super().__init__(*args, **kwargs)
        self.model.cls_feats_proj_head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(80,80, 1),
                nn.SiLU(),
                nn.Conv2d(80, 128, 1)
            ) for _ in range(3)
        ])
