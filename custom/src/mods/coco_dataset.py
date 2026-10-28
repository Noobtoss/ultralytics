import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics.data import YOLODataset
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.utils import TQDM
from ultralytics.utils.ops import clip_boxes, ltwh2xyxy
from ultralytics.utils.torch_utils import unwrap_model


# Source: https://docs.ultralytics.com/guides/coco-json-training


class COCODataset(YOLODataset):
    """Dataset that reads COCO JSON annotations directly without conversion to .txt files."""

    def __init__(self, *args, ann_file="", **kwargs):
        """Initialize the dataset with a COCO JSON annotation file."""
        self.ann_file = ann_file
        # >>> MOD
        super().__init__(*args, data=kwargs.pop("data", {"channels": 3}), **kwargs)
        # <<< MOD

    def get_img_files(self, img_path):
        """Image paths are resolved from the JSON file, not from scanning a directory."""
        # >>> MOD
        # self.fraction = 1.0  # fraction is applied while scanning a directory, which this dataset skips
        # <<< MOD
        return []

    def cache_labels(self, path=Path("./labels.cache")):
        """Parse COCO JSON and convert annotations to YOLO format in memory."""
        x = {"labels": []}
        with open(self.ann_file) as f:
            coco = json.load(f)

        categories = {cat["id"]: cat["id"] - 1 for cat in coco["categories"]}

        # >>> MOD: check that category IDs and names match the training metadata.
        names = self.data.get("coco_names", self.data.get("names"))
        if names is not None:
            if any(names.get(cat["id"] - 1) != cat["name"] for cat in coco["categories"]):
                raise ValueError(f"{self.ann_file}: category IDs and names do not match the training metadata")
        # <<< MOD

        img_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            # >>> MOD
            if not ann.get("iscrowd", False) and ann["category_id"] not in categories:
                raise KeyError(ann["category_id"])
            # <<< MOD
            img_to_anns[ann["image_id"]].append(ann)

        # >>> MOD: honor fraction because get_img_files() does not scan the directory.
        images = coco["images"]
        if self.fraction < 1:
            images = images[:round(len(images) * self.fraction)]
        # <<< MOD

        for img_info in TQDM(images, desc="reading annotations"):
            h, w = img_info["height"], img_info["width"]
            im_file = Path(self.img_path) / img_info["file_name"]
            # >>> MOD
            if not im_file.is_file():
                raise FileNotFoundError(f"{self.ann_file}: image not found: {im_file}")
            # <<< MOD

            self.im_files.append(str(im_file))
            bboxes = []
            for ann in img_to_anns.get(img_info["id"], []):
                if ann.get("iscrowd", False):
                    continue
                if not ann.get("bbox", False):
                    continue
                box = np.array(ann["bbox"], dtype=np.float32)
                box[:2] += box[2:] / 2
                box[[0, 2]] /= w
                box[[1, 3]] /= h
                if box[2] <= 0 or box[3] <= 0:
                    continue
                cls = categories[ann["category_id"]]
                bboxes.append([cls, *box.tolist()])

            lb = np.array(bboxes, dtype=np.float32) if bboxes else np.zeros((0, 5), dtype=np.float32)
            x["labels"].append(
                {
                    "im_file": str(im_file),
                    "shape": (h, w),
                    "cls": lb[:, 0:1],
                    "bboxes": lb[:, 1:],
                    "segments": [],
                    "normalized": True,
                    "bbox_format": "xywh",
                }
            )

        if not x["labels"]:
            raise ValueError(f"{self.ann_file}: no images found")
        # MOD: omit the hash and save_dataset_cache_file() to keep annotations in memory.
        return x

    def get_labels(self):
        """Read annotations each time, without loading or writing a disk cache."""
        cache = self.cache_labels()
        self.im_files = [lb["im_file"] for lb in cache["labels"]]
        return cache["labels"]


class COCORTDETRDataset(COCODataset, RTDETRDataset):
    """Use COCO annotations with RT-DETR's image resizing and transforms."""


def resolve_data(args):
    """Return the YAML path or an in-memory COCO dataset configuration."""
    if not getattr(args, "train_ann", None):
        return args.data
    paths = [Path(path).resolve() for path in (args.train_ann, args.val_ann)]
    imgs_dir = getattr(args, "data_dir", None)
    imgs_dir = str(Path(imgs_dir).resolve() if imgs_dir else paths[0].parent)
    with paths[0].open(encoding="utf-8") as file:
        categories = sorted(json.load(file)["categories"], key=lambda category: category["id"])
    names = {category["id"] - 1: category["name"] for category in categories}
    names = {i: names.get(i, f"unused_{i}") for i in range(max(names) + 1)}
    return dict(train=imgs_dir, val=imgs_dir, train_ann=str(paths[0]), val_ann=str(paths[1]),
                names=names, coco_names=names, nc=len(names), channels=3)


def build_coco_dataset(owner, img_path, mode="val", batch=None, rtdetr=False):
    """Build the appropriate COCO dataset for a trainer or validator."""
    model = getattr(owner, "model", None)
    stride = max(int(unwrap_model(model).stride.max()), 32) if model is not None else (getattr(owner, "stride", None) or 32)
    dataset = COCORTDETRDataset if rtdetr else COCODataset
    return dataset(
        img_path=img_path,
        ann_file=owner.data[f"{mode}_ann"],
        data=owner.data,
        imgsz=owner.args.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=owner.args,
        rect=False if rtdetr else owner.args.rect or mode == "val",
        cache=owner.args.cache or None,
        single_cls=owner.args.single_cls or False,
        stride=stride,
        pad=0.0 if mode == "train" else 0.5,
        classes=owner.args.classes,
        fraction=owner.args.fraction if mode == "train" else 1.0,
    )
