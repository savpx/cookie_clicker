from ultralytics import YOLO

model = YOLO("bestV1.pt")
model.export(format="onnx", opset=12, simplify=True)