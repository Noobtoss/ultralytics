from ultralytics import YOLO
import coremltools as ct
from PIL import Image, ImageDraw
import numpy as np


def find_crop(img, imgsz):
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)

    scale = max(img.size[0] / imgsz[0], img.size[1] / imgsz[1])
    return (int(img.size[0] / scale), int(img.size[1] / scale)), imgsz


def yolo_padding(img, resize_size=(960, 540), padding_size=(960, 960), padding_color=(128, 128, 128)):
    img = img.resize(resize_size)
    padded_img = Image.new("RGB", padding_size, padding_color)
    padded_img.paste(img, ((padding_size[0] - resize_size[0]) // 2, (padding_size[1] - resize_size[1]) // 2))
    return padded_img


def yolo_inference(model, img):
    # model.predict(img, imgsz=960, save=True, conf=0.5, iou=0.75)  # Test inference yolo_model
    for result in model(img):  # Test yolo_model response to image (somewhat) inference
        result.show()
    #    result.save(filename='results.jpg')


def coreml_inference(model, img):
    names = eval(model.get_spec().description.metadata.userDefined["names"])
    height, width = eval(model.get_spec().description.metadata.userDefined["imgsz"])
    draw = ImageDraw.Draw(img)

    results = model.predict({'image': img, "iouThreshold": 0.45, "confidenceThreshold": 0.25})
    for bbox, logits in zip(results["coordinates"], results["confidence"]):
        x_center, y_center, box_width, box_height = bbox
        left = int((x_center - box_width / 2) * width)
        top = int((y_center - box_height / 2) * height)
        right = int((x_center + box_width / 2) * width)
        bottom = int((y_center + box_height / 2) * height)
        draw.rectangle([left, top, right, bottom], outline="red", width=5)
        draw.text((left, top - 2), anchor="lb", text=f"{names[np.argmax(logits)]} {np.max(logits):.2f}", font_size=20,
                  fill="red")
    img.show()


def main():
    img = Image.open("resources/test03.jpg")
    yolo_model = YOLO('../../models/demo_models/SemmelDemo04.pt')
    coreml_model = ct.models.MLModel('../../models/demo_models/SemmelDemo04.mlpackage')

    resize_size, padding_size = find_crop(img, imgsz=1280)
    img = yolo_padding(img, resize_size=resize_size, padding_size=padding_size)
    yolo_inference(yolo_model, img)
    coreml_inference(coreml_model, img)


if __name__ == "__main__":
    main()
