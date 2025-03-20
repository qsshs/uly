from ultralytics import YOLO

# Load a model
model = YOLO("yolo11l-cls.pt")

# Export the model
model.export(format="onnx", opset = 9, simplify = True)