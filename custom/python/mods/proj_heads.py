import torch.nn as nn


class ProjHeadFactory:
    @staticmethod
    def get(proj_head: str = None, in_channels: int = 256, nl: int = 3, **kwargs):
        if proj_head is None or proj_head == "None":
            return None

        if proj_head == "s":
            # 1-layer linear (weak baseline)
            return nn.Sequential(
                nn.Linear(in_channels, 128, bias=False)
            )

        if proj_head == "m":
            # 2-layer MLP (SupCon paper choice)
            return nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, 128, bias=False)
            )

        if proj_head == "l":
            # 2-layer MLP (SupCon paper choice)
            return nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, 256, bias=False)
            )

        raise ValueError(f"Unknown proj head type: '{proj_head}'. Choose from: 's', 'm', 'l'")
