# import cv2
# from ultralytics import YOLO

# # Load your trained model
# model = YOLO("best.pt")

# # Open the video file
# video_path = "data/DJI_0020.MP4"
# cap = cv2.VideoCapture(video_path)

# # Loop through video frames
# while cap.isOpened():
#     success, frame = cap.read()
    
#     if success:
#         # Run YOLO inference on the frame (force CPU)
#         results = model(frame)
        
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()
        
#         # Display the annotated frame
#         cv2.imshow("YOLO Detection", annotated_frame)
        
#         # Press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break

# # Cleanup
# cap.release()
# cv2.destroyAllWindows()


import time
import os
import cv2
import torch
from ultralytics import YOLO

# Configuration
MODEL_PATH = "best.pt"
ENGINE_PATH = "best.engine"        # will be created if missing
IMG_SIZE = 640                     # lower to 512/480 for more speed
USE_TENSORRT = True                # set False to skip engine export
SHOW_WINDOW = True
PRINT_FPS = True
USE_GSTREAMER = False              # set True to use HW decode pipeline
GST_PIPELINE = (
    f"filesrc location=data/DJI_0020.MP4 ! qtdemux ! h264parse ! nvv4l2decoder ! "
    "video/x-raw(memory:NVMM),format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! "
    "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 sync=0"
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
half = torch.cuda.is_available()   # Jetson supports FP16

def load_model():
    if USE_TENSORRT and os.path.isfile(ENGINE_PATH):
        return YOLO(ENGINE_PATH)
    model = YOLO(MODEL_PATH)
    if USE_TENSORRT and device.startswith("cuda"):
        # Export FP16 TensorRT engine (runs once)
        model.export(format="engine", half=half, device=device, imgsz=IMG_SIZE)
        return YOLO(ENGINE_PATH)
    return model

model = load_model()

# Warmup (small dummy tensor)
if device.startswith("cuda"):
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    _ = model(dummy, imgsz=IMG_SIZE, device=device, half=half, verbose=False)

# Video capture
if USE_GSTREAMER:
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture("data/DJI_0020.MP4")

frame_count = 0
t0 = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # Inference
    results = model(
        frame,
        device=device,
        imgsz=IMG_SIZE,
        half=half,
        verbose=False,
    )

    annotated = results[0].plot()  # Ultralytics GPU drawing is fine

    if SHOW_WINDOW:
        cv2.imshow("YOLO (Jetson)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_count += 1
    if PRINT_FPS and frame_count % 30 == 0:
        dt = time.time() - t0
        fps = frame_count / dt
        print(f"Frames: {frame_count} | Avg FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()