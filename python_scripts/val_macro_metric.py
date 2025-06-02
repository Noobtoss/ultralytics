import argparse
from argparse import Namespace
import yaml
from ultralytics import YOLO
from pathlib import Path
import shutil
import numpy as np


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/Users/noobtoss/codeNexus/ultralytics')
    parser.add_argument('--dir', type=str, default='results/semmel/05Ostern')
    args = parser.parse_args()
    return args

def unpack_metrics(metrics):
    names = metrics.names
    map = metrics.box.map
    map50 = metrics.box.map50
    ap = [0] * len(names)
    ap50 = [0] * len(names)
    for idx, first, second in zip(metrics.box.ap_class_index, metrics.box.ap, metrics.box.ap50):
        ap[idx] = first
        ap50[idx] = second
    return {
        "names": names,
        "map": map,
        "map50": map50,
        "ap": ap,
        "ap50": ap50
    }

def macro_metrics(metrics):
    maps = [d["map"] for d in metrics]
    map50s = [d["map50"] for d in metrics]
    aps = np.array([d["ap"] for d in metrics])
    ap50s = np.array([d["ap50"] for d in metrics])

    return {
        "names": next(iter(metrics))["names"],
        "map_m": round(np.mean(maps), 3),
        "map_s": round(np.std(maps), 3),
        "map50_m": round(np.mean(map50s), 3),
        "map50_s": round(np.std(map50s), 3),
        "ap_m": np.round(np.mean(aps, axis=0), 3).tolist(),
        "ap_s": np.round(np.std(aps, axis=0), 3).tolist(),
        "ap50_me": np.round(np.mean(ap50s, axis=0), 3).tolist(),
        "ap50_s": np.round(np.std(ap50s, axis=0), 3).tolist()
    }

def print_macro_metrics(macro):
    names = macro["names"]
    ap_m = macro["ap_m"]
    ap_s = macro["ap_s"]
    ap50_m = macro["ap50_me"]
    ap50_s = macro["ap50_s"]

    print(f"mAP:     {macro['map_m']:.3f} ± {macro['map_s']:.3f}")
    print(f"mAP@50:  {macro['map50_m']:.3f} ± {macro['map50_s']:.3f}")
    print()
    print("Per-class AP:")
    for i, name in names.items():
        if ap_m[i] > 0:
            print(f" - {name:<24} {ap_m[i]:.3f} ± {ap_s[i]:.3f}")
    print()
    print("Per-class AP@50:")
    for i, name in names.items():
        if ap50_m[i] > 0:
            print(f" - {name:<24} {ap50_m[i]:.3f} ± {ap50_s[i]:.3f}")


def val_macro_metric():
    args = parse_args()
    root = Path(args.root)
    dir = root / args.dir

    model_args_current = None
    metrics_list = []
    name_list = []
    for model_dir in sorted(dir.iterdir()):
        args_path = model_dir / 'args.yaml'
        if args_path.exists():
            with (open(args_path, 'r') as f):
                model_args = yaml.safe_load(f)
                name_list.append(model_args.pop("name"))
                model_args.pop("seed")
                model_args.pop("save_dir")

            if model_args_current is None:
                model_args_current = model_args
            elif model_args_current != model_args:
                model_args_current = model_args
                macro = macro_metrics(metrics_list)
                print("\n" + "=" * 50 + "\n")
                print(name_list[-2])
                print("\n" + "=" * 50 + "\n")
                print_macro_metrics(macro)
                print("\n" + "=" * 50 + "\n")
                metrics_list = []
            last_model_path = model_dir / "weights/last.pt"
            model = YOLO(last_model_path)
            metrics = model.val(split="test", verbose=False)
            metrics = unpack_metrics(metrics)
            metrics_list.append(metrics)
    macro = macro_metrics(metrics_list)
    print("\n" + "=" * 50 + "\n")
    print(name_list[-2])
    print("\n" + "=" * 50 + "\n")
    print_macro_metrics(macro)
    print("\n" + "=" * 50 + "\n")
    if Path('runs').exists():
        shutil.rmtree('runs')

if __name__ == '__main__':
    val_macro_metric()
