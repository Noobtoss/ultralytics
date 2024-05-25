from ultralytics import YOLO

data = 2 * ["datasets/v1.9/yoloSetups/Semmel52.yaml","datasets/v1.9/yoloSetups/Semmel53.yaml","datasets/v1.9/yoloSetups/Semmel54.yaml","datasets/v1.9/yoloSetups/Semmel56.yaml","datasets/v1.9/yoloSetups/Semmel57.yaml"]

models = ["runs/cfgv8X1280pxSemmel52-1654/train/weights/last.pt", "runs/cfgv8X1280pxSemmel53-1653/train/weights/last.pt", "runs/cfgv8X1280pxSemmel54-1652/train/weights/last.pt", "runs/cfgv8X1280pxSemmel56-1657/train/weights/last.pt", "runs/cfgv8X1280pxSemmel57-1658/train/weights/last.pt", "runs/cfgv9E1280pxSemmel52-1651/train/weights/last.pt", "runs/cfgv9E1280pxSemmel53-1650/train/weights/last.pt", "runs/cfgv9E1280pxSemmel54-1649/train/weights/last.pt", "runs/cfgv9E1280pxSemmel56-1659/train/weights/last.pt", "runs/cfgv9E1280pxSemmel57-1660/train/weights/last.pt",
          "runs/cfgv8X1280pxSemmel52-1654/train/weights/best.pt", "runs/cfgv8X1280pxSemmel53-1653/train/weights/best.pt", "runs/cfgv8X1280pxSemmel54-1652/train/weights/best.pt", "runs/cfgv8X1280pxSemmel56-1657/train/weights/best.pt", "runs/cfgv8X1280pxSemmel57-1658/train/weights/best.pt", "runs/cfgv9E1280pxSemmel52-1651/train/weights/best.pt", "runs/cfgv9E1280pxSemmel53-1650/train/weights/best.pt", "runs/cfgv9E1280pxSemmel54-1649/train/weights/best.pt", "runs/cfgv9E1280pxSemmel56-1659/train/weights/best.pt", "runs/cfgv9E1280pxSemmel57-1660/train/weights/best.pt"]

names = ["runs/cfgv8X1280pxSemmel52-1654/testLast", "runs/cfgv8X1280pxSemmel53-1653/testLast", "runs/cfgv8X1280pxSemmel54-1652/testLast", "runs/cfgv8X1280pxSemmel56-1657/testLast", "runs/cfgv8X1280pxSemmel57-1658/testLast", "runs/cfgv9E1280pxSemmel52-1651/testLast", "runs/cfgv9E1280pxSemmel53-1650/testLast", "runs/cfgv9E1280pxSemmel54-1649/testLast", "runs/cfgv9E1280pxSemmel56-1659/testLast", "runs/cfgv9E1280pxSemmel57-1660/testLast",
         "runs/cfgv8X1280pxSemmel52-1654/testBest", "runs/cfgv8X1280pxSemmel53-1653/testBest", "runs/cfgv8X1280pxSemmel54-1652/testBest", "runs/cfgv8X1280pxSemmel56-1657/testBest", "runs/cfgv8X1280pxSemmel57-1658/testBest", "runs/cfgv9E1280pxSemmel52-1651/testBest", "runs/cfgv9E1280pxSemmel53-1650/testBest", "runs/cfgv9E1280pxSemmel54-1649/testBest", "runs/cfgv9E1280pxSemmel56-1659/testBest", "runs/cfgv9E1280pxSemmel57-1660/testBest"]

for data, model, name in zip(data, models, names):
    # Load a model
    model = YOLO(model)

    # Customize validation settings
    validation_results = model.val(data=data, imgsz=1280, batch=8, project=".", name=name)
