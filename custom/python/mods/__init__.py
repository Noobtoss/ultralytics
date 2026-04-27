import ultralytics.nn.modules as nn_modules
import ultralytics.nn.tasks as tasks

from .model import Model
from .yolo import YOLO
from .cls_feats_return_detect import ClsFeatsReturnDetect

# Monkey-patch namespaces
nn_modules.ClsFeatsReturnDetect = ClsFeatsReturnDetect
tasks.ClsFeatsReturnDetect = ClsFeatsReturnDetect

# Patch parse_model to treat Detect2 like Detect
_original_parse_model = tasks.parse_model


def _patched_parse_model(d, ch, verbose=True):
    import ultralytics.nn.tasks as t
    _orig_detect = t.Detect
    t.Detect = ClsFeatsReturnDetect  # trick parse_model into handling ClsFeatsReturnDetect like Detect
    result = _original_parse_model(d, ch, verbose)
    t.Detect = _orig_detect  # restore
    return result


tasks.parse_model = _patched_parse_model
