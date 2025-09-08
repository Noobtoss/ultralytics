from ultralytics import YOLO

model = YOLO("checkpoints/yolo11x-seg.pt")

results = model.train(data="datasets/holz/holz00baseline.yaml", epochs=2, imgsz=640)
