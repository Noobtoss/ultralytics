from argparse import Namespace
from pathlib import Path

from mods.coco_dataset import resolve_data


TRAIN_CFG = Namespace(
    data="",
    epochs=100,
    imgsz=1280,  # 640
    seed=88888,
    batch=8,
    single_cls=False,
    project="runs",
    name="",
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # class_weights=None,
    cls_feat=0,  # 0.5,
    # cls_feat_loss="sup_con_loss",
    # cls_feat_proj_head="s",
    # cls_feat_proj_head_lr=0.001,
)

CFG = Namespace(
    ckpt="checkpoints/yolo11x.pt",  # "checkpoints/rtdetr-x.pt"
    train_cfg=TRAIN_CFG
)

ALIASES = {}

VALUE_MAPPING = {}


def merge_cfg(cfg: Namespace, args):
    it = iter(args.opts)
    for key, value in zip(it, it):
        for target in ALIASES.get(key, (key,)):
            mapped_value = VALUE_MAPPING[target](value) if target in VALUE_MAPPING else value
            setattr(cfg.train_cfg, target, mapped_value)

def tag(cfg: Namespace):
    train_cfg = cfg.train_cfg
    train_cfg.model_name = Path(getattr(cfg, 'ckpt', None) or getattr(cfg, 'model', None)).stem
    train_ann = train_cfg.data.get("train_ann") if isinstance(train_cfg.data, dict) else None
    if not train_ann:
        train_cfg.dataset_name = Path(train_cfg.data).stem
    else:
        train_cfg.dataset_name = Path(train_ann).resolve().parent.name
    train_cfg.run_name = train_cfg.name

def parse_cfg(args: Namespace) -> Namespace:
    args = Namespace(**{"data": None, **vars(args)})
    if getattr(args, "data_dir", None):
        data_dir = Path(args.data_dir)
        for key in ("data", "train_ann", "val_ann", "test_ann"):
            path = getattr(args, key, None)
            if path:
                setattr(args, key, str(data_dir / Path(path)))
    args.val_ann = getattr(args, "val_ann", None) or getattr(args, "test_ann", None)
    cfg = CFG

    cfg.train_cfg.name = args.run_name;  delattr(args, "run_name")
    cfg.train_cfg.save_dir = args.save_dir; delattr(args, "save_dir")
    cfg.model = args.model; delattr(args, "model")
    cfg.ckpt = args.ckpt; delattr(args, "ckpt")
    if args.opts:
        merge_cfg(cfg, args)
    cfg.train_cfg.data = resolve_data(args); delattr(args, "data")
    tag(cfg)
    return cfg
