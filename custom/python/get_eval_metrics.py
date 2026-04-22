import os
from argparse import Namespace
import yaml
import pandas as pd
from ultralytics import YOLO


def get_eval_metrics(cfg: Namespace) -> None:
    eval_cfg = cfg.train_cfg
    original_name = eval_cfg.name
    results_dir = os.path.join(eval_cfg.project, eval_cfg.name)

    model = YOLO(f"{results_dir}/weights/last.pt")

    with open(eval_cfg.data) as f:
        data_cfg = yaml.safe_load(f)

    results = model.val(**vars(eval_cfg), split="test", exist_ok=True)
    metrics = {
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

        metrics[set_name] = {
            "mAP50": round(results.box.map50, 3),
            "mAP75": round(results.box.map75, 3),
            "mAP50-95": round(results.box.map, 3),
        }
        os.remove(tmp_yaml)

    pd.DataFrame.from_dict(metrics, orient="index").to_csv(os.path.join(results_dir, "metrics.csv"))
