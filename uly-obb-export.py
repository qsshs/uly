from ultralytics import YOLO

# Load a model
model_XL = YOLO("yolo11x-obb.pt")
model_L = YOLO("yolo11l-obb.pt")
model_M = YOLO("yolo11m-obb.pt")
model_S = YOLO("yolo11s-obb.pt")

# Export the model
model_XL.export(format="onnx", opset=9, simplify=True, imgsz=(1024, 1024))
model_L.export(format="onnx", opset=9, simplify=True, imgsz=(1024, 1024))
model_M.export(format="onnx", opset=9, simplify=True, imgsz=(1024, 1024))
model_S.export(format="onnx", opset=9, simplify=True, imgsz=(1024, 1024))
