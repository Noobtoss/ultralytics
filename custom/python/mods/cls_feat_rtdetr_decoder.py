from ultralytics.utils import LOGGER
from ultralytics.nn.modules.head import RTDETRDecoder as _RTDETRDecoder

class ClsFeatRTDETRDecoder(_RTDETRDecoder):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("[Modded] RTDETRDecoder")
        super().__init__(*args, **kwargs)
