import os
from datetime import datetime
import argparse
from argparse import Namespace
import yaml
import pandas as pd
from ultralytics import YOLO

# === Configuration ===
MODEL = ["models/yolo11x.pt"]
DATA = ["datasets/semmel/04/semmel61.yaml",
        "datasets/semmel/04/semmel64.yaml",
        "datasets/semmel/04/semmel65.yaml",
        "datasets/semmel/04/semmel66.yaml",
        "datasets/semmel/04/semmel67.yaml",
        "datasets/semmel/04/semmel68.yaml"]
EPOCHS = [200, 200, 200, 200, 200, 200]
SEEDS = [6666]

# === Configuration ===
MODEL = ["models/yolo11x.pt"]
DATA = ["datasets/semmel/04/semmel66.yaml",
        "datasets/semmel/04/semmel67.yaml",
        "datasets/semmel/04/semmel68.yaml"]
EPOCHS = [100, 100, 100]
SEEDS = [886666, 881313, 888888, 884040, 881919]

# === Configuration ===
MODEL = ["models/yolo11x.pt"]
DATA = ["datasets/semmel/03/semmel69.yaml",
        "datasets/semmel/03/semmel70.yaml"]
EPOCHS = [300, 300]
SEEDS = [886666, 881313, 888888, 884040, 881919]


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

def parse_index(index: int) -> tuple[str, str, int, int]:
    num_models = len(MODEL)
    num_datas = len(DATA)
    num_seeds = len(SEEDS)

    # Calculate indices based on id (mimicking SLURM_ARRAY_TASK_ID logic)
    index = index - 1  # Zero-based index (SLURM_ARRAY_TASK_ID is 1-based)

    # Reversed calculation: Seed first
    seed_index = index % num_seeds  # Calculate seed first
    data_index = (index // num_seeds) % num_datas  # Calculate data second
    config_index = index // (num_seeds * num_datas)  # Calculate model last

    # Get the corresponding configuration values
    model = MODEL[config_index]
    data = DATA[data_index]
    epochs = EPOCHS[data_index]  # Assuming epochs vary per dataset
    seed = SEEDS[seed_index]

    print(f"Seed: {seed}, Model: {model}, Data: {data}, Epochs: {epochs}")
    return model, data, epochs, seed

def parse_config(model: str, data:str, epochs:int, seed:int) -> Namespace:
    config = argparse.Namespace()
    config.model = model
    config.train_config = argparse.Namespace()
    config.train_config.data = data
    config.train_config.epochs = epochs
    config.train_config.seed = seed
    config.train_config.project = (f"runs/{os.path.splitext(os.path.basename(model))[0]}-"
                                   f"{os.path.splitext(os.path.basename(data))[0].lower()}")
    config.train_config.name = f"{seed}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    return config

def main():
    args = parse_args()
    model, data, epochs, seed = parse_index(args.index)
    config = parse_config(model, data, epochs, seed)
    training(config)
    evaluation(config)

if __name__ == '__main__':
    main()
