import torch.nn as nn


class ProjHeadFactory:
    @staticmethod
    def get(proj_head: str = None, in_channels: int = 80, nl: int = 3, **kwargs):
        if proj_head is None:
            return None
        if proj_head == "s":
            return nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, 128, 1),
                    nn.SiLU(),
                    nn.Conv2d(128, 128, 1)
                ) for _ in range(nl)
            ])
        if proj_head == "m":
            return nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, 256, 1),
                    nn.SiLU(),
                    nn.Conv2d(256, 256, 1)
                ) for _ in range(nl)
            ])
        else:
            raise ValueError(f"Unknown proj head type: '{proj_head}'")
