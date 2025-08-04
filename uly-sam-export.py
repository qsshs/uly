from ultralytics import FastSAM

# Load a model
model_XL = FastSAM("FastSAM-x.pt")
model_S = FastSAM("FastSAM-s.pt")

# Export the model
model_XL.export(format="onnx", opset=9, simplify=True, imgsz=(1024, 1024))
model_S.export(format="onnx", opset=9, simplify=True, imgsz=(1024, 1024))
