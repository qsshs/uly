import ultralytics as uly

model = uly.YOLO("yolov8m-obb.pt")

res = model.predict(
    "uly-img-1.jpg", conf=0.5, save=True, show_labels=False, show_conf=False, imgsz=640
)

for i in res:
    i.show()
