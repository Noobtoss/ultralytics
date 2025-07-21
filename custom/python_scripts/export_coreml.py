from ultralytics import YOLO
import coremltools as ct

yolo_model = YOLO('../../models/demo_models/SemmelDemo04.pt')  # load yolo_model

yolo_model.export(format="coreml", imgsz=1280, nms=True)  # export yolo_model
# coreml_model = ct.models.MLModel('resources/Semmel51.mlpackage')  # load coreml_model
