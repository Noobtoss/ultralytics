from ultralytics.utils import LOGGER
from ultralytics.models.rtdetr.val import RTDETRValidator as _RTDETRValidator
from .coco_dataset import build_coco_dataset

from .detection_validator import DetectionValidator


_RTDETRValidator.__bases__ = (DetectionValidator,)


class RTDETRValidator(_RTDETRValidator):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("[Modded] RTDETRValidator")
        super().__init__(*args, **kwargs)

    def build_dataset(self, img_path, mode="val", batch=None):
        if "train_ann" in self.data:
            return build_coco_dataset(self, img_path, mode, batch, rtdetr=True)
        return super().build_dataset(img_path, mode, batch)
