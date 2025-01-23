from ultralytics import YOLO

# Load a model
model = YOLO("yolov8m-obb.pt")

# Export the model
model.export(format="onnx", opset = 9, simplify = True)