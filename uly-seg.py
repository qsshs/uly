import ultralytics as uly

model = uly.YOLO("yolov8m-seg.pt")

model.export(format = "onnx", opset = 9)

res = model.predict(
    "uly-img-2.jpg", conf=0.5, save=True, show_labels=False, show_conf=False, retina_masks = True, imgsz = 1024
)

for i in res:
    i.show()
