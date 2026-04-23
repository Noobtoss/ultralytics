from ultralytics.utils.loss import v8DetectionLoss


# THS, Copied from ultralytics.utils.loss


class TrainLoss(v8DetectionLoss):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
