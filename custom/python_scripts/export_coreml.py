import argparse
from ultralytics import YOLO
import coremltools as ct


def make_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description="YOLO CoreML Export Script")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size used during CoreML export")
    parser.add_argument("--ckpt", type=str, default="last.pt", help="Path to the YOLO model checkpoint (.pt file)")
    return parser


def main():
    # yolo_model = YOLO('../../models/demo_models/SemmelDemo04.pt')  # load yolo_model
    # yolo_model.export(format="coreml", imgsz=1280, nms=True)  # export yolo_model
    # coreml_model = ct.models.MLModel('resources/Semmel51.mlpackage')  # load coreml_model
    args = make_parser().parse_args()
    yolo_model = YOLO(args.ckpt)
    yolo_model.export(format="coreml", imgsz=args.imgsz, nms=True)  # export yolo_model


if __name__ == "__main__":
    main()
