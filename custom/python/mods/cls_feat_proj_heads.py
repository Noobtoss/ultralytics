import torch.nn as nn


class ClsFeatProjHeadFactory:
    @staticmethod
    def get(proj_head: str = None, in_channels: int = 256, **kwargs):
        if proj_head is None or proj_head == "None":
            return None

        if proj_head == "n":
            return nn.Sequential(
                nn.Linear(in_channels, 64, bias=False)
            )

        if proj_head == "s":
            return nn.Sequential(
                nn.Linear(in_channels, 128, bias=False)
            )

        if proj_head == "m":
            return nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, 128, bias=False)
            )

        if proj_head == "l":
            return nn.Sequential(
                nn.Linear(in_channels, in_channels, bias=False),
                nn.BatchNorm1d(in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, 256, bias=False)
            )

        raise ValueError(f"Unknown proj head type: '{proj_head}'. Choose from: 'n', 's', 'm', 'l'")
