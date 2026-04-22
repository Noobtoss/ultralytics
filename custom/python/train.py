import os
import argparse
from argparse import Namespace
import yaml
import pandas as pd
import warnings

from ultralytics import YOLO
from ultralytics.cfg import CFG_FLOAT_KEYS, CFG_FRACTION_KEYS, CFG_INT_KEYS, CFG_BOOL_KEYS

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
    model.train(**vars(cfg.train_cfg))


def eval(cfg: Namespace) -> None:
    eval_cfg = cfg.train_cfg
    original_name = eval_cfg.name
    results_dir = os.path.join(eval_cfg.project, eval_cfg.name)

    model = YOLO(f"{results_dir}/weights/last.pt")

    with open(eval_cfg.data) as f:
        data_cfg = yaml.safe_load(f)

    results = model.val(**vars(eval_cfg), split="test", exist_ok=True)
    evaluation_results = {
        "test-full": {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
        }
    }

    tmp_yaml = os.path.join(results_dir, "tmp.yaml")
    for subset in data_cfg.get("test", {}):
        set_name = os.path.basename(os.path.dirname(subset))

        with open(tmp_yaml, "w") as f:
            yaml.dump({**data_cfg, "test": subset}, f, default_flow_style=False)

        eval_cfg.data = tmp_yaml
        eval_cfg.name = os.path.join(original_name, f"test-{set_name}")  # ← use original_name
        results = model.val(**vars(eval_cfg), split="test")

        evaluation_results[set_name] = {
            "mAP50": round(results.box.map50, 3),
            "mAP75": round(results.box.map75, 3),
            "mAP50-95": round(results.box.map, 3),
        }
        os.remove(tmp_yaml)

    pd.DataFrame.from_dict(evaluation_results, orient="index").to_csv(os.path.join(results_dir, "metrics.csv"))


def parse_args():
    parser = argparse.ArgumentParser("ultralytics train parser")
    parser.add_argument("--exp_name", type=str, help="exp name")
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

    return cfg


def main():
    args = parse_args()
    cfg = parse_cfg(args)
    train(cfg)
    eval(cfg)


if __name__ == '__main__':
    main()
