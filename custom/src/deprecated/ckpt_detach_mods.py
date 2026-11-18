import os
import site
import sys
import torch
from pathlib import Path

# When running with a local ultralytics/ directory present, Python would normally
# pick up the local folder instead of the conda-installed package. We fix this by
# manipulating sys.path explicitly — since insert(0, ...) is a stack operation,
# entries are added in reverse priority order so that conda site-packages lands
# at index 0 and takes precedence over the local directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # index 1 — local imports
sys.path.insert(0, site.getsitepackages()[0])  # index 0 — conda site-packages (priority)

from ultralytics import YOLO


def ckpt_detach_mods(ckpt):
    import subprocess
    import ultralytics

    def detach_yaml(model_yaml):
        pkg_dir = Path(ultralytics.__file__).parent
        yamls = [f.stem for f in pkg_dir.glob("cfg/models/**/*.yaml")]
        return next((y + ".yaml" for y in yamls if y in model_yaml), None)

    ckpt_detached = Path(ckpt)
    ckpt_detached = ckpt_detached.with_stem(ckpt_detached.stem + "_detached")

    model = YOLO(ckpt)
    model_nc = model.model.nc

    model_yaml = model.model.yaml.get("yaml_file")
    model_scale = model.model.yaml.get("scale")
    model_names = model.names
    assert model_yaml is not None, f"Could not find yaml_file in model.model.yaml, keys: {list(model.model.yaml.keys())}"
    model_yaml = detach_yaml(model_yaml)
    assert model_yaml is not None, f"Could not match yaml_file to any ultralytics default yaml"

    model_state = model.model.state_dict()
    model_state = {k: v for k, v in model_state.items() if "proj_head" not in k and "cls_feat" not in k}

    torch.save(model_state, ckpt_detached)

    script = (
        "import sys, torch\n"
        "from ultralytics import YOLO, RTDETR\n"
        f"model = YOLO('{model_yaml}')\n"
        f"model.overrides['nc'] = {model_nc}\n"
        f"model.model = model.model.__class__(model.model.yaml | {{'nc': {model_nc}, 'scale': '{model_scale}'}})\n"
        f"state = torch.load('{str(ckpt_detached)}')\n"
        "model.model.load_state_dict(state, strict=True)\n"
        f"model.model.names = {model_names!r}\n"
        "model.overrides['names'] = model.model.names\n"
        f"model.save('{str(ckpt_detached)}')\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def main():
    ckpt = "last.pt"
    ckpt_detach_mods(ckpt)


if __name__ == "__main__":
    main()
