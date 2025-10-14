from ultralytics import YOLO

model = YOLO("yolo11s.yaml")

results = model.train(data="dataset/data.yaml", epochs=15, device="mps", resume="true")

results = model.val()

results = model("dataset/test/images/00504_jpg.rf.83a7e15ff43c0f937313abc601afa6c5.jpg")

success = model.export(format="onnx")