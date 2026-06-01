from ultralytics.utils import LOGGER
from ultralytics.models.utils.loss import RTDETRDetectionLoss as _RTDETRDetectionLoss

class RTDETRDetectionLoss(_RTDETRDetectionLoss):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("[Modded] RTDETRDetectionLoss")
        super().__init__(*args, **kwargs)
