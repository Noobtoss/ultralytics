from ultralytics.utils import LOGGER
from ultralytics.models.utils.loss import RTDETRDetectionLoss as _RTDETRDetectionLoss
from ultralytics.models.utils.loss import DETRLoss as _DETRLoss



class DETRLoss(_DETRLoss):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("[Modded] DETRLoss")
        super().__init__(*args, **kwargs)


_RTDETRDetectionLoss.__bases__ = (DETRLoss,)


class RTDETRDetectionLoss(_RTDETRDetectionLoss):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("[Modded] RTDETRDetectionLoss")
        super().__init__(*args, **kwargs)
