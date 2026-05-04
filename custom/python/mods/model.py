import torch
import torch.nn as nn
from ultralytics.engine.model import Model as BaseModel
from ultralytics.utils import LOGGER

class Model(BaseModel):
    def __init__(self, *args, **kwargs) -> None:
        LOGGER.warning("Model __init__ called")
        super().__init__(*args, **kwargs)
