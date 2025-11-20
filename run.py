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
MODEL_PATH = "best.pt"              # replace with yolov8n.pt for max speed if acceptable
ENGINE_PATH = "best_fp16.engine"    # TensorRT engine output/reuse
IMG_SIZE = 512                      # reduce from 640 for speed (try 416 too)
USE_TENSORRT = True
SHOW_WINDOW = False                 # disable to measure pure speed
PRINT_FPS = True
FRAME_SKIP = 0                      # set to 1 to process every other frame, etc.
USE_GSTREAMER = True

# Hardware decode at 1280x720 (adjust lower if still slow)
GST_PIPELINE = (
    "filesrc location=data/DJI_0020.MP4 ! qtdemux ! h264parse ! nvv4l2decoder ! "
    "video/x-raw(memory:NVMM),format=NV12 ! nvvidconv ! "
    "video/x-raw,width=1280,height=720,format=BGRx ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1 sync=0"
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
half = device.startswith("cuda")    # Jetson supports FP16

def load_model():
    if USE_TENSORRT and os.path.isfile(ENGINE_PATH):
        return YOLO(ENGINE_PATH)
    model = YOLO(MODEL_PATH)
    if USE_TENSORRT and device.startswith("cuda"):
        model.export(format="engine", half=half, device=device, imgsz=IMG_SIZE)
        return YOLO(ENGINE_PATH)
    return model

model = load_model()

# Optional warmup
if device.startswith("cuda"):
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    _ = model(dummy, imgsz=IMG_SIZE, device=device, half=half, verbose=False)

# Video capture
if USE_GSTREAMER:
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture("data/DJI_0020.MP4")

frame_count = 0
proc_count = 0
start_time = time.time()

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    # Optional skip
    if FRAME_SKIP and (frame_count % (FRAME_SKIP + 1)) != 0:
        frame_count += 1
        continue

    # Inference (no plotting to save time)
    results = model(frame, device=device, imgsz=IMG_SIZE, half=half, verbose=False)

    # If you need boxes, access results[0].boxes (avoid plot)
    # boxes = results[0].boxes

    if SHOW_WINDOW:
        annotated = results[0].plot()
        cv2.imshow("YOLO Fast", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    frame_count += 1
    proc_count += 1

    if PRINT_FPS and proc_count % 60 == 0:
        elapsed = time.time() - start_time
        fps = proc_count / elapsed
        print(f"Processed frames: {proc_count} | Avg FPS: {fps:.2f}")

cap.release()
cv2.destroyAllWindows()