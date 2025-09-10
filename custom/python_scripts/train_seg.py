from ultralytics import YOLO

model = YOLO("checkpoints/yolo11x-seg.pt")

results = model.train(data="datasets/holz/holz01noSK.yaml", epochs=100, imgsz=1280)
