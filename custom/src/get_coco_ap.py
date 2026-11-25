import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _image_id(im_file):
    """Unique image ID: the full path. Must be used on both GT and prediction side."""
    return str(im_file)


def _write_coco_preds(trainer, directory):
    validator = trainer.validator
    orig = validator.pred_to_json

    def pred_to_json(predn, second):
        n = len(validator.jdict)
        orig(predn, second)
        # newer Ultralytics passes pbatch (dict), older versions pass the filename
        im_file = second["im_file"] if isinstance(second, dict) else second
        for d in validator.jdict[n:]:
            d["image_id"] = _image_id(im_file)

    with patch.object(validator, "save_dir", directory), \
         patch.object(validator.args, "save_json", True), \
         patch.object(validator, "eval_json", lambda stats: stats), \
         patch.object(validator, "pred_to_json", pred_to_json):
        validator(model=trainer.last)
    return directory / "predictions.json"


def _write_coco_gt(validator, directory):
    gt = {
        "info": {}, "images": [], "annotations": [],
        "categories": [{"id": validator.class_map[i], "name": name} for i, name in validator.names.items()],
    }
    dataset = validator.dataloader.dataset
    for index, label in enumerate(dataset.labels):
        # Rectangular batching removes "shape" from the stored labels.
        _, (height, width), _ = dataset.load_image(index)
        image_id = _image_id(label["im_file"])
        gt["images"].append({
            "id": image_id, "file_name": label["im_file"], "height": height, "width": width,
        })
        if not label["normalized"] or label["bbox_format"] != "xywh":
            raise ValueError("Expected normalized xywh validation labels.")
        boxes = label["bboxes"] * [width, height, width, height]
        for cls, (x, y, w, h) in zip(label["cls"].reshape(-1), boxes.tolist()):
            gt["annotations"].append({
                "id": len(gt["annotations"]) + 1, "image_id": image_id,
                "category_id": validator.class_map[0 if validator.args.single_cls else int(cls)],
                "bbox": [x - w / 2, y - h / 2, w, h],
                "area": w * h, "iscrowd": 0,
            })
    if len({im["id"] for im in gt["images"]}) != len(gt["images"]):
        raise ValueError("Validation images must have unique paths.")
    path = directory / "gt.json"
    path.write_text(json.dumps(gt), encoding="utf-8")
    return path


def get_coco_ap(trainer):
    with TemporaryDirectory(dir=trainer.save_dir) as tmp:
        preds_path = _write_coco_preds(trainer, Path(tmp))
        if not preds_path.exists() or json.loads(preds_path.read_text(encoding="utf-8")) == []:
            return {"eval/mAP50-95": 0.0, "eval/mAP50": 0.0}
        gt_path = _write_coco_gt(trainer.validator, Path(tmp))
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(preds_path))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return {"eval/mAP50-95": float(evaluator.stats[0]), "eval/mAP50": float(evaluator.stats[1])}
