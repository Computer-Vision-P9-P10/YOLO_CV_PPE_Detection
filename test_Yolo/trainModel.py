from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolo12m.pt")

# Train the model on Construction-PPE dataset
model.train(data="dataset/data.yaml", epochs=100, imgsz=640, save=True)