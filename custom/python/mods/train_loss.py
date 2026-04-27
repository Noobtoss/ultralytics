import torch
from ultralytics.utils.loss import v8DetectionLoss


# THS, Copied from ultralytics.utils.loss


class TrainLoss(v8DetectionLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        print(preds.keys())
        return self.loss(self.parse_output(preds), batch)
