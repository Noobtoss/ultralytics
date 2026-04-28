import os
import sys
import site
import argparse
from argparse import Namespace

# When running from inside a local ultralytics repo clone, Python would normally
# pick up the local folder instead of the conda-installed package. We fix this by
# forcing the conda site-packages to the front of the search path.
sys.path.insert(0, site.getsitepackages()[0])
# Also ensure the directory of this script itself is on the path for local imports.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_eval_metrics import get_eval_metrics
from mods import YOLO, DetectionTrainer, LossGainScheduler

DEFAULT_TRAIN_CFG = Namespace(
    data="",
    epochs=100,
    imgsz=1280, # 640
    seed=88888,
    batch=8,
    single_cls=False,
    project="runs",
    name="",
    box=7.5,
    cls=0.5,
    dfl=1.5,
    cls_feat_loss="sup_con_loss",
    cls_feat=0.09, # 0.5,
    cls_feat_loss_temp=0.1,
)

DEFAULT_CFG = Namespace(
    ckpt="checkpoints/yolo11x.pt",
    train_cfg=DEFAULT_TRAIN_CFG
)


def train(cfg: Namespace):
    if cfg.model is not None:
        model = YOLO(cfg.model).load(cfg.ckpt)
    else:
        model = YOLO(cfg.ckpt)
    model.add_callback("on_train_epoch_start", LossGainScheduler())
    model.train(**vars(cfg.train_cfg), trainer=DetectionTrainer)


def parse_args():
    parser = argparse.ArgumentParser("ultralytics train parser")
    parser.add_argument("--exp_name", type=str, help="exp name")
    parser.add_argument("--save_dir", type=str, help="save dir")
    parser.add_argument("--model", type=str, default=None, help="path to model")
    parser.add_argument("--ckpt", type=str, help="path to ckpt")
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

    cfg.train_cfg.name = args.exp_name; delattr(args, "exp_name")
    cfg.train_cfg.save_dir = args.save_dir; delattr(args, "save_dir")
    cfg.model = args.model; delattr(args, "model")
    cfg.ckpt = args.ckpt; delattr(args, "ckpt")
    cfg.train_cfg.data = args.data; delattr(args, "data")

    if args.opts:
        it = iter(args.opts)
        for k, v in zip(it, it):
            setattr(cfg.train_cfg, k, v)
    return cfg


def main():
    FLAG = False
    # FLAG = True
    if FLAG:
        args = Namespace(
            exp_name="unnamed_experiment",
            save_dir="/Users/noobtoss/code_nexus/ultralytics/runs/unnamed_experiment",
            model="/Users/noobtoss/code_nexus/ultralytics/custom/cfg/cls_feats_return_yolo11n.yaml",
            ckpt="/Users/noobtoss/code_nexus/ultralytics/checkpoints/yolo11n.pt",
            data="/Users/noobtoss/code_nexus/ultralytics/datasets/semmel/Images05MetaFood2026_local.yaml",
            opts="",
        )
    else:
        args = parse_args()

    cfg = parse_cfg(args)
    train(cfg)
    get_eval_metrics(cfg)


if __name__ == '__main__':
    main()
