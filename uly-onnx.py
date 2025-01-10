import onnx


model = onnx.load("yolov8m-seg.onnx")


for output in model.graph.output:
    print(output.name, output.type.tensor_type.shape)