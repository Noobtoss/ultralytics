import os
import sys
import site
import warnings
import argparse
from argparse import Namespace
import csv

# When running with a local ultralytics/ directory present, Python would normally
# pick up the local folder instead of the conda-installed package. We fix this by
# manipulating sys.path explicitly — since insert(0, ...) is a stack operation,
# entries are added in reverse priority order so that conda site-packages lands
# at index 0 and takes precedence over the local directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # index 1 — local imports
sys.path.insert(0, site.getsitepackages()[0])                   # index 0 — conda site-packages (priority)

from ultralytics.utils import LOGGER
from get_eval_metrics import get_eval_metrics
from mods import YOLO, DetectionTrainer, DetectionValidator, LossGainScheduler

DEFAULT_TRAIN_CFG = Namespace(
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
    cls_feat=0.09,  # 0.5,
    cls_feat_loss="sup_con_loss",
    cls_feat_temperature=0.07,
    # cls_feat_mask="conf",
    # cls_feat_top_rel=0.4,
    # cls_feat_weight="conf",
    # cls_feat_alpha=1.0,
    # cls_feat_beta=6.0,
    # cls_feat_proj_head="s",
    # cls_feat_proj_head_lr=0.001,
    # tal_topk=10,
)

DEFAULT_CFG = Namespace(
    ckpt="checkpoints/yolo11x.pt",
    train_cfg=DEFAULT_TRAIN_CFG
)


def val_last(trainer):
    if trainer.last.exists():
        LOGGER.info(f"\nValidating {trainer.last}...")
        metrics = trainer.validator(model=trainer.last)

        for csv_path in [trainer.save_dir.parent / "results.csv",
                         trainer.save_dir.parent.parent / "results/results.csv",
                         trainer.save_dir.parent.parent / "results.csv"]:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            row = {"name": trainer.args.name, **{k: round(v, 3) for k, v in metrics.items()}}
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            LOGGER.info(f"Results saved to {csv_path}")


def train(cfg: Namespace):
    if cfg.model is not None:
        model = YOLO(cfg.model).load(cfg.ckpt)
    else:
        model = YOLO(cfg.ckpt)
    model.add_callback("on_train_epoch_start", LossGainScheduler())
    model.add_callback("on_train_end", val_last)
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

    cfg.train_cfg.name = args.exp_name;  delattr(args, "exp_name")
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
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        warnings.warn("⚠️ Running with hardcoded test args")
        args = Namespace(
            exp_name="unnamed_experiment",
            save_dir="/Users/noobtoss/code_nexus/ultralytics/runs/unnamed_experiment",
            model="/Users/noobtoss/code_nexus/ultralytics/custom/cfg/cls_feat_yolo26n.yaml",
            ckpt="/Users/noobtoss/code_nexus/ultralytics/checkpoints/yolo26n.pt",
            data="/Users/noobtoss/code_nexus/ultralytics/datasets/semmel/Images05ACCV2026_local.yaml",
            opts="",
        )

    cfg = parse_cfg(args)
    train(cfg)
    get_eval_metrics(cfg)


if __name__ == '__main__':
    main()
