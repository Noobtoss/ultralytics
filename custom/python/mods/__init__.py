import ultralytics.cfg as _cfg
import ultralytics.engine.trainer as _trainer
import ultralytics.nn.modules as _modules
import ultralytics.nn.tasks as _tasks
from ultralytics.utils import LOGGER

from .cls_feat_loss import ClsFeatLossFactory, ClsFeatLoss
from .cls_feat_detect import ClsFeatDetect
from .detection_model import DetectionModel
from .detection_trainer import DetectionTrainer
from .detection_validator import DetectionValidator
from .loss_gain_scheduler import LossGainScheduler
from .cls_feat_proj_heads import ClsFeatProjHeadFactory
from .train_loss import TrainLoss
from .yolo import YOLO


# ── CFG validation ────────────────────────────────────────────────────────────
def patched_check_dict_alignment(*args, **kwargs):
    LOGGER.warning("[MonkeyPatch] check_dict_alignment skipped")
    pass


_cfg.CFG_FLOAT_KEYS |= {"cls_feat", "cls_feat_temperature", "cls_feat_proj_head_lr",
                        "cls_feat_top_rel", "cls_feat_alpha", "cls_feat_beta"}
_cfg.CFG_INT_KEYS |= {"tal_topk"}
_base_check_cfg = _cfg.check_cfg


def patched_check_cfg(cfg: dict) -> None:
    LOGGER.warning("[MonkeyPatch] check_cfg forced hard=False")
    _base_check_cfg(cfg, hard=False)


_cfg.check_dict_alignment = patched_check_dict_alignment
_cfg.check_cfg = patched_check_cfg
_trainer.check_cfg = patched_check_cfg
_trainer.check_dict_alignment = patched_check_dict_alignment

# ── Namespace patches ─────────────────────────────────────────────────────────
_modules.ClsFeatDetect = ClsFeatDetect
_tasks.ClsFeatDetect = ClsFeatDetect

# ── parse_model patch ─────────────────────────────────────────────────────────
_original_parse_model = _tasks.parse_model


def _patched_parse_model(d, ch, verbose=True):
    _orig_detect = _tasks.Detect
    _tasks.Detect = ClsFeatDetect
    result = _original_parse_model(d, ch, verbose)
    _tasks.Detect = _orig_detect
    return result


_tasks.parse_model = _patched_parse_model
