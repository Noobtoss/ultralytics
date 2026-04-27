import ultralytics.cfg as ucfg
import ultralytics.engine.trainer as utrainer
import ultralytics.nn.modules as nn_modules
import ultralytics.nn.tasks as tasks

from .cls_feats_return_detect import ClsFeatsReturnDetect
from .yolo import YOLO
from .detection_trainer import DetectionTrainer

# ── CFG validation ────────────────────────────────────────────────────────────
def _patched_check_dict_alignment(*args, **kwargs):
    pass

ucfg.check_dict_alignment    = _patched_check_dict_alignment
utrainer.check_dict_alignment = _patched_check_dict_alignment

# ── Namespace patches ─────────────────────────────────────────────────────────
nn_modules.ClsFeatsReturnDetect = ClsFeatsReturnDetect
tasks.ClsFeatsReturnDetect      = ClsFeatsReturnDetect

# ── parse_model patch ─────────────────────────────────────────────────────────
_original_parse_model = tasks.parse_model

def _patched_parse_model(d, ch, verbose=True):
    _orig_detect  = tasks.Detect
    tasks.Detect  = ClsFeatsReturnDetect
    result        = _original_parse_model(d, ch, verbose)
    tasks.Detect  = _orig_detect
    return result

tasks.parse_model = _patched_parse_model
