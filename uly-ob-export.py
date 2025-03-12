from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m.pt")

# Export the model
model.export(format="onnx", opset = 9, simplify = True)