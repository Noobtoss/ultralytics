import ultralytics.cfg as ucfg
import ultralytics.engine.trainer as utrainer
import ultralytics.nn.modules as nn_modules
import ultralytics.nn.tasks as tasks
from ultralytics.utils import LOGGER

from .base_trainer import BaseTrainer
from .cls_feat_loss import ClsFeatLossFactory, ClsFeatLoss
from .cls_feats_detect import ClsFeatsDetect
from .detection_model import DetectionModel
from .detection_trainer import DetectionTrainer
from .loss_gain_scheduler import LossGainScheduler
from .model import Model
from .proj_heads import ProjHeadFactory
from .train_loss import TrainLoss
from .yolo import YOLO


# ── CFG validation ────────────────────────────────────────────────────────────
def patched_check_dict_alignment(*args, **kwargs):
    LOGGER.warning("[MonkeyPatch] check_dict_alignment skipped")
    pass


ucfg.CFG_FLOAT_KEYS |= {"cls_feat", "cls_feat_temperature"}  # just add your keys to the set
_base_check_cfg = ucfg.check_cfg


def patched_check_cfg(cfg: dict) -> None:
    LOGGER.warning("[MonkeyPatch] check_cfg forced hard=False")
    _base_check_cfg(cfg, hard=False)


ucfg.check_dict_alignment = patched_check_dict_alignment
ucfg.check_cfg = patched_check_cfg
utrainer.check_cfg = patched_check_cfg
utrainer.check_dict_alignment = patched_check_dict_alignment

# ── Namespace patches ─────────────────────────────────────────────────────────
nn_modules.ClsFeatsDetect = ClsFeatsDetect
tasks.ClsFeatsDetect = ClsFeatsDetect

# ── parse_model patch ─────────────────────────────────────────────────────────
_original_parse_model = tasks.parse_model


def _patched_parse_model(d, ch, verbose=True):
    _orig_detect = tasks.Detect
    tasks.Detect = ClsFeatsDetect
    result = _original_parse_model(d, ch, verbose)
    tasks.Detect = _orig_detect
    return result


tasks.parse_model = _patched_parse_model
