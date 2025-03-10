import os
from datetime import datetime
import argparse
from argparse import Namespace
import yaml
import pandas as pd
from ultralytics import YOLO
from typing import Any

# === Configuration ===
MODEL = "models/yolo11x.pt"
DATA = ["datasets/semmel/04/semmel61.yaml",
        "datasets/semmel/04/semmel64.yaml",
        "datasets/semmel/04/semmel65.yaml",
        "datasets/semmel/04/semmel66.yaml",
        "datasets/semmel/04/semmel67.yaml",
        "datasets/semmel/04/semmel68.yaml"]
EPOCHS = 200
SEEDS = 6666
IMGSZ = 640

# === Configuration ===
MODEL = "models/yolo11x.pt"
DATA = ["datasets/semmel/04/semmel66.yaml",
        "datasets/semmel/04/semmel67.yaml",
        "datasets/semmel/04/semmel68.yaml"]
EPOCHS = 100
SEEDS = [886666, 881313, 888888, 884040, 881919]
IMGSZ = 640

# === Configuration ===
MODEL = "models/yolov8x.pt"
DATA = ["datasets/semmel/03/semmel69.yaml",
        "datasets/semmel/03/semmel70.yaml"]
EPOCHS = 300
SEEDS = [886666, 881313, 888888, 884040, 881919]
IMGSZ = 640

# === Configuration ===
MODEL = "yolo11x.yaml"
DATA = ["datasets/coco/coco05.yaml",
        "datasets/coco/coco06.yaml"]
EPOCHS = 80
SEEDS = [886666, 881313, 888888, 884040]
IMGSZ = 640

# === Configuration ===
MODEL = "yolo11x.yaml"
DATA = ["datasets/semmel/machine/semmel75.yaml",
        "datasets/semmel/machine/semmel76.yaml",
        "datasets/semmel/machine/semmel77.yaml",
        "datasets/semmel/machine/semmel78.yaml"]
EPOCHS = 100
SEEDS = 6666
IMGSZ = 1280

# === Configuration ===
MODEL = "yolo11x.yaml"
DATA = ["datasets/semmel/04/semmel61.yaml",
        "datasets/semmel/04/semmel80.yaml",
        "datasets/semmel/04/semmel81.yaml",
        "datasets/semmel/04/semmel82.yaml",
        "datasets/semmel/04/semmel83.yaml"]
EPOCHS = 100
SEEDS = [886666, 881313, 888888, 884040]
IMGSZ = 1280

def training(config: Namespace):
    model = YOLO(config.model)
    train_config = config.train_config
    model.train(**vars(train_config))

def evaluation(config: Namespace) -> None:
    eval_config = config.train_config
    with open(eval_config.data, "r") as f:
        data_cfg = yaml.safe_load(f)

    results_dir = str(os.path.join(eval_config.project, eval_config.name))
    last_model_path = f"{results_dir}/weights/last.pt"
    model = YOLO(last_model_path)

    evaluation_results = {}

    results = model.val(**vars(eval_config), split="test", exist_ok=True)
    evaluation_results["test-full"] = {
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
    }

    test_sets = data_cfg.get("test", {})
    for set in test_sets:
        set_name = os.path.basename(os.path.dirname(set))
        tmp_file_name = os.path.join(results_dir, 'tmp.yaml')
        with open(tmp_file_name, 'w') as file:
            data = {"path": data_cfg["path"],
                    "train": data_cfg["train"],
                    "test": set,
                    "val": data_cfg["val"],
                    "names": data_cfg["names"]}
            yaml.dump(data, file, default_flow_style=False)

        eval_config.data=tmp_file_name
        eval_config.name = os.path.join(eval_config.name, f"test-{set_name}")
        results = model.val(**vars(eval_config), split="test")
        eval_config.name = os.path.dirname(eval_config.name)
        evaluation_results[set_name] = {
            "mAP50": results.box.map50,
            "mAP75": results.box.map75,
            "mAP50-95": results.box.map
        }
        os.remove(tmp_file_name)
    df = pd.DataFrame.from_dict(evaluation_results, orient='index')
    df.to_csv(os.path.join(results_dir, "evaluation.csv"))
    return

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    return args

def parse_index(index: int) -> dict[str, Any]:
    num_models = len(MODEL) if isinstance(MODEL, list) else 1
    num_datas = len(DATA) if isinstance(DATA, list) else 1
    num_seeds = len(SEEDS) if isinstance(SEEDS, list) else 1

    # Calculate indices based on id (mimicking SLURM_ARRAY_TASK_ID logic)
    index = index - 1  # Zero-based index (SLURM_ARRAY_TASK_ID is 1-based)

    # Reversed calculation: Seed first
    model_index = index // (num_seeds * num_datas)  # Calculate model last
    data_index = (index // num_seeds) % num_datas  # Calculate data second
    seed_index = index % num_seeds  # Calculate seed first

    # Get the corresponding configuration values
    model = MODEL[model_index] if isinstance(MODEL, list) else MODEL
    data = DATA[data_index]  if isinstance(DATA, list) else DATA
    epochs = EPOCHS[data_index] if isinstance(EPOCHS, list) else EPOCHS
    imgsz = IMGSZ[data_index] if isinstance(IMGSZ, list) else IMGSZ
    seed = SEEDS[seed_index] if isinstance(SEEDS, list) else SEEDS

    config_dict = {
        "model": model,
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "seed": seed
    }

    print(config_dict)
    return config_dict

def parse_config(config_dict: dict[str, Any]) -> Namespace:
    config = argparse.Namespace()
    model = config_dict.pop("model")
    data = config_dict["data"]
    seed = config_dict["seed"]
    config.model = model
    config.train_config = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(config.train_config, key, value)
    config.train_config.project = "runs"
    config.train_config.name = (f"{os.path.splitext(os.path.basename(model))[0]}-"
                                f"{os.path.splitext(os.path.basename(data))[0].lower()}-"
                                f"{seed}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}")

    return config

def main():
    args = parse_args()
    config_dict = parse_index(args.index)
    config = parse_config(config_dict)
    training(config)
    evaluation(config)

if __name__ == '__main__':
    main()
