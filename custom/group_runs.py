#
# Example call:
#   conda activate conda-ultralytics
#   python custom/group_runs.py
#
import argparse
import re
import shutil
import sys
from pathlib import Path
import yaml


IGNORED_ARGS = {"seed", "name", "run_name", "save_dir"}


def read_args(args_path):
    args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if not isinstance(args, dict):
        raise ValueError(f"Expected a YAML mapping: {args_path}")
    return {key: value for key, value in args.items() if key not in IGNORED_ARGS}


def group_runs(runs_dir, target_dir=None, valid_only=True, tag_runs=False):
    runs_dir = Path(runs_dir).resolve()
    target_dir = Path(target_dir).resolve() if target_dir is not None else runs_dir.parent / "results"
    if not runs_dir.is_dir():
        raise ValueError(f"Not a directory: {runs_dir}")

    groups = {}
    for run in sorted(runs_dir.iterdir()):
        if valid_only and not (run / "TRAIN_SUCCESS").is_file():
            continue
        match = re.fullmatch(r"(.+)_\d+_\d+", run.name)
        if run.is_dir() and match:
            groups.setdefault(match[1], []).append(run)
    if not groups:
        raise ValueError(f"No run directories ending in _<number>_<number> in {runs_dir}")

    # Validate every group before moving anything.
    plans = []
    completed = []
    for group, runs in groups.items():
        target = target_dir / group
        reference = None
        moves = {}
        for run in runs:
            args_path = run / "args.yaml"
            checkpoints = sorted(path for path in run.rglob("*.pt") if path.is_file())
            if not args_path.is_file() or not checkpoints:
                print(f"Warning: skipping {run}: missing args.yaml or .pt checkpoints")
                continue
            args = read_args(args_path)
            if reference is None:
                if tag_runs:
                    model_name = str(args.get("model_name", "")).lower()
                    match = re.search(r"yolo\d+|rtdetr", model_name)
                    if match:
                        target = target_dir / match[0] / group
                    else:
                        print(f"Warning: leaving {target} untagged: unsupported model_name: {model_name}")
                reference_path = target / "args.yaml" if target.exists() or target.is_symlink() else args_path
                reference = read_args(reference_path)
            if args != reference:
                raise ValueError(f"Arguments differ: {reference_path} vs {args_path}")

            for checkpoint in checkpoints:
                name = f"{run.name}.pt" if len(checkpoints) == 1 else f"{run.name}_{checkpoint.stem}.pt"
                existing = moves.get(name, target / name)
                if existing.exists() or existing.is_symlink():
                    print(f"Skipping {checkpoint}: destination name already in use: {name}")
                    continue
                moves[name] = checkpoint
            if all(checkpoint in moves.values() for checkpoint in checkpoints):
                completed.append(run)
        if moves:
            plans.append((target, {"args.yaml": reference_path, **moves}))

    for target, moves in plans:
        print(f"Organizing into {target}")
        target.mkdir(parents=True, exist_ok=True)
        for name, source in moves.items():
            if source != target / name:
                print(f"  {source} -> {name}")
                transfer = shutil.copy2 if name == "args.yaml" else shutil.move
                transfer(str(source), target / name)

    for run in completed:
        (run / "TRAIN_SUCCESS").unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir",
                        type=Path,
                        default=Path(__file__).resolve().parents[1] / "runs",
                        help="Directory containing the individual run folders")
    parser.add_argument("--target_dir",
                        type=Path,
                        default=Path(__file__).resolve().parents[3] / "data_hub/results",
                        help="Output directory; defaults to the repository's results folder")
    parser.add_argument("--valid-only",
                        type=str.lower,
                        choices=("true", "false"),
                        default="true",
                        help="Only group runs containing TRAIN_SUCCESS (default: true)")
    args = parser.parse_args()
    args.valid_only = args.valid_only == "true"
    return args


if __name__ == "__main__":
    if len(sys.argv) > 1:
        args = parse_args()
    else:
        args = argparse.Namespace(
            runs_dir=Path(__file__).resolve().parents[1] / "runs",
            target_dir=Path(__file__).resolve().parents[3] / "data_hub/results",
            valid_only=True,
        )
    group_runs(args.runs_dir, args.target_dir, args.valid_only, tag_runs=args.runs_dir.parent != args.target_dir.parent)
