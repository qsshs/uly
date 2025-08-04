from ultralytics import YOLO

# Load a model
model_XL = YOLO("yolo11x-cls.pt")
model_L = YOLO("yolo11l-cls.pt")
model_M = YOLO("yolo11m-cls.pt")
model_S = YOLO("yolo11s-cls.pt")

# Export the model
model_XL.export(format="onnx", opset=9, simplify=True, imgsz=(640, 640))
model_L.export(format="onnx", opset=9, simplify=True, imgsz=(640, 640))
model_M.export(format="onnx", opset=9, simplify=True, imgsz=(640, 640))
model_S.export(format="onnx", opset=9, simplify=True, imgsz=(640, 640))
