from ultralytics.utils import LOGGER
from ultralytics.nn.tasks import RTDETRDetectionModel as _RTDETRDetectionModel


class RTDETRDetectionModel(_RTDETRDetectionModel):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] RTDETRDetectionModel")
        super().__init__(*args, **kwargs)

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from .rtdetr_detection_loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)
