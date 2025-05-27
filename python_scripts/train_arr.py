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
MODEL = "models/yolo11x.pt"
DATA = ["datasets/semmel/05Machine/semmel75.yaml",
        "datasets/semmel/05Machine/semmel76.yaml",
        "datasets/semmel/05Machine/semmel77.yaml",
        "datasets/semmel/05Machine/semmel78.yaml"]
EPOCHS = 100
SEEDS = 6666
IMGSZ = 1280

# === Configuration ===
MODEL = "models/yolo11x.pt"
DATA = ["datasets/semmel/04/semmel65.yaml",
        "datasets/semmel/04/semmel80.yaml",
        "datasets/semmel/04/semmel81.yaml",
        "datasets/semmel/04/semmel82.yaml",
        "datasets/semmel/04/semmel83.yaml"]
EPOCHS = 100
SEEDS = [886666, 881313, 888888, 884040]
IMGSZ = 1280

# === Configuration ===
MODEL = "yolo11x.yaml"
DATA =  ["datasets/coco/coco05.yaml",
         "datasets/coco/coco06.yaml",
         "datasets/coco/coco07.yaml",
         "datasets/coco/coco08.yaml",
         "datasets/coco/coco09.yaml"]
EPOCHS = 70
SEEDS = 6666
IMGSZ = 640

# === Configuration ===
MODEL = "models/yolo11x.pt"
DATA = ["datasets/semmel/05Machine/semmel75.yaml",
        "datasets/semmel/05Machine/semmel76.yaml",
        "datasets/semmel/05Machine/semmel77.yaml",
        "datasets/semmel/05Machine/semmel78.yaml"]
EPOCHS = 100
SEEDS = 6666
IMGSZ = 1280

# === Configuration ===
MODEL = "models/yolo11s.pt"
DATA = ["datasets/semmel/05Machine/semmel75.yaml",
        "datasets/semmel/05Machine/semmel76.yaml",
        "datasets/semmel/05Machine/semmel77.yaml",
        "datasets/semmel/05Machine/semmel78.yaml"]
EPOCHS = 200
SEEDS = 6666
IMGSZ = 640

# === Configuration ===
MODEL = ["models/yolo11s.pt",
         "models/yolo11m.pt",
         "models/yolo11l.pt",
         "models/yolo11x.pt"]
DATA = "datasets/lebensmittel/lebensmittel01.yaml"
EPOCHS = 300
SEEDS = 6666
IMGSZ = 640

# === Configuration ===
MODEL = "yolo11x.yaml"
DATA =  ["datasets/coco/coco10.yaml",
         "datasets/coco/coco11.yaml"]
EPOCHS = 70
SEEDS = 6666
IMGSZ = 640

# === Configuration ===
MODEL = "models/yolo11x.pt"
DATA = ["datasets/semmel/05fzi2025/semmel84.yaml",
        "datasets/semmel/05fzi2025/semmel85.yaml",
        "datasets/semmel/05fzi2025/semmel86.yaml",
        "datasets/semmel/05fzi2025/semmel87.yaml",
        "datasets/semmel/05fzi2025/semmel88.yaml",
        "datasets/semmel/05fzi2025/semmel89.yaml",
        "datasets/semmel/05fzi2025/semmel90.yaml",
        "datasets/semmel/05fzi2025/semmel91.yaml"]
EPOCHS = 100
SEEDS = [886666, 881313, 888888, 884040]
IMGSZ = 1280

# === Configuration ===
MODEL = "models/yolo11x.pt"
DATA = ["datasets/semmel/05fzi2025/semmel92.yaml",
        "datasets/semmel/05fzi2025/semmel93.yaml",
        "datasets/semmel/05fzi2025/semmel94.yaml"]
EPOCHS = 100
SEEDS = [886666, 881313, 888888, 884040]
IMGSZ = 1280

# === Configuration ===
MODEL = "models/yolo11x.pt"
DATA = ["datasets/semmel/05Ostern/semmel95.yaml",
        "datasets/semmel/05Ostern/semmel96.yaml",
        "datasets/semmel/05Ostern/semmel97.yaml",
        "datasets/semmel/05Ostern/semmel98.yaml",
        "datasets/semmel/05Ostern/semmel99.yaml",
        "datasets/semmel/05Ostern/semmelDemo04.yaml",
        "datasets/semmel/05Ostern/semmel100.yaml",
        "datasets/semmel/05Ostern/semmel101.yaml",
        "datasets/semmel/05Ostern/semmel102.yaml"]
EPOCHS = 100
SEEDS = [886666, 881313, 888888, 884040]
IMGSZ = 1280

# === Configuration ===
MODEL = ["models/yolo11s.pt", "models/yolo11m.pt", "models/yolo11l.pt", "models/yolo11x.pt"]
DATA = ["datasets/semmel/05Machine/semmel75.yaml"]
EPOCHS = 100
SEEDS = 6666
IMGSZ = 1280

# === Configuration ===
MODEL = ["models/yolo11s.pt", "models/yolo11m.pt", "models/yolo11l.pt", "models/yolo11x.pt"]
DATA = ["datasets/semmel/05Machine/semmel75.yaml"]
EPOCHS = 100
SEEDS = 6666
IMGSZ = 640

# === Configuration ===
MODEL = "models/yolo11x.pt"
DATA = ["datasets/papier/papier00.yaml",
        "datasets/papier/papier01.yaml"]
EPOCHS = [100, 2000]
SEEDS = [886666, 881313, 888888, 884040]
IMGSZ = 1280

# === Configuration ===
MODEL = "models/yolo11x.pt"
DATA = ["datasets/semmel/05Zucker/semmel103.yaml",
        "datasets/semmel/05Zucker/semmel104.yaml",
        "datasets/semmel/05Zucker/semmel105.yaml",
        "datasets/semmel/05Zucker/semmel106.yaml",
        "datasets/semmel/05Zucker/semmel107.yaml",
        "datasets/semmel/05Zucker/semmel108.yaml",
        "datasets/semmel/05Zucker/semmel109.yaml",
        "datasets/semmel/05Zucker/semmel110.yaml"
        ]
EPOCHS = 100
SEEDS = [886666, 881313, 888888, 884040]
IMGSZ = 1280


def training(cfg: Namespace):
    model = YOLO(cfg.model)
    train_cfg = cfg.train_cfg
    # train_cfg.save_period = 10
    # train_cfg.time = 42
    model.train(**vars(train_cfg))

def evaluation(cfg: Namespace) -> None:
    eval_cfg = cfg.train_cfg
    with open(eval_cfg.data, "r") as f:
        data_cfg = yaml.safe_load(f)

    results_dir = str(os.path.join(eval_cfg.project, eval_cfg.name))
    last_model_path = f"{results_dir}/weights/last.pt"
    model = YOLO(last_model_path)

    evaluation_results = {}

    results = model.val(**vars(eval_cfg), split="test", exist_ok=True)
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

        eval_cfg.data=tmp_file_name
        eval_cfg.name = os.path.join(eval_cfg.name, f"test-{set_name}")
        results = model.val(**vars(eval_cfg), split="test")
        eval_cfg.name = os.path.dirname(eval_cfg.name)
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

    model_index = index // (num_seeds * num_datas)
    seed_index = (index // num_datas) % num_seeds
    data_index = index % num_datas

    # Get the corresponding configuration values
    model = MODEL[model_index] if isinstance(MODEL, list) else MODEL
    seed = SEEDS[seed_index] if isinstance(SEEDS, list) else SEEDS
    data = DATA[data_index]  if isinstance(DATA, list) else DATA
    epochs = EPOCHS[data_index] if isinstance(EPOCHS, list) else EPOCHS
    imgsz = IMGSZ[data_index] if isinstance(IMGSZ, list) else IMGSZ

    cfg_dict = {
        "model": model,
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "seed": seed,
        "batch": 8
    }

    print(cfg_dict)
    return cfg_dict

def parse_cfg(cfg_dict: dict[str, Any]) -> Namespace:
    cfg = argparse.Namespace()
    model = cfg_dict.pop("model")
    data = cfg_dict["data"]
    seed = cfg_dict["seed"]
    cfg.model = model
    cfg.train_cfg = argparse.Namespace()
    for key, value in cfg_dict.items():
        setattr(cfg.train_cfg, key, value)
    cfg.train_cfg.project = "runs"
    cfg.train_cfg.name = (f"{os.path.splitext(os.path.basename(model))[0]}-"
                                f"{os.path.splitext(os.path.basename(data))[0].lower()}-"
                                f"{seed}-{datetime.now().strftime('%Y-%m-%d_%H-%M')}")

    return cfg

def train_arr():
    args = parse_args()
    cfg_dict = parse_index(args.index)
    cfg = parse_cfg(cfg_dict)
    training(cfg)
    evaluation(cfg)

if __name__ == '__main__':
    train_arr()
