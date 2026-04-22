import sys
import os
import argparse
from argparse import Namespace
import warnings
from ultralytics import YOLO
from ultralytics.cfg import CFG_FLOAT_KEYS, CFG_FRACTION_KEYS, CFG_INT_KEYS, CFG_BOOL_KEYS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_eval_metrics import get_eval_metrics
from mods.trainer import Trainer

DEFAULT_TRAIN_CFG = Namespace(
    data="",
    epochs=100,
    imgsz=640,
    seed=88888,
    batch=8,
    single_cls=False,
    project="runs",
    name="",
)

DEFAULT_CFG = Namespace(
    model="checkpoints/yolo11x.pt",
    train_cfg=DEFAULT_TRAIN_CFG
)


def train(cfg: Namespace):
    model = YOLO(cfg.model)
    model.train(trainer=Trainer, **vars(cfg.train_cfg))


def parse_args():
    parser = argparse.ArgumentParser("ultralytics train parser")
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--save_dir", type=str, help="save dir")
    parser.add_argument("--model", type=str, help="path to model file")
    parser.add_argument("--data", type=str, help="path to data file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def parse_cfg(args: Namespace) -> Namespace:
    cfg = DEFAULT_CFG

    cfg.train_cfg.name = args.exp_name
    cfg.train_cfg.save_dir = args.save_dir
    cfg.model = args.model
    cfg.train_cfg.data = args.data

    if args.opts:
        it = iter(args.opts)
        for k, v in zip(it, it):
            if v is not None:
                if k in CFG_FLOAT_KEYS:
                    setattr(cfg.train_cfg, k, float(v))
                elif k in CFG_FRACTION_KEYS:
                    setattr(cfg.train_cfg, k, float(v))
                elif k in CFG_INT_KEYS:
                    setattr(cfg.train_cfg, k, int(v))
                elif k in CFG_BOOL_KEYS:
                    setattr(cfg.train_cfg, k, bool(v))
                else:
                    warnings.warn(f"Skipping unknown key: '{k}'")

    print(cfg)

    return cfg


def main():
    args = parse_args()
    cfg = parse_cfg(args)
    train(cfg)
    get_eval_metrics(cfg)


if __name__ == '__main__':
    main()
