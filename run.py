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

# Config
MODEL_PATH = "yolov8n.pt"           # use a small model for testing; replace with best.pt later
ENGINE_PATH = "best_fp16.engine"
IMG_SIZE = 416                      # smaller size for speed
USE_TENSORRT = False                # start False; enable after base works
SHOW_WINDOW = False
PRINT_FPS = True
FRAME_SKIP = 0
USE_GSTREAMER = True

GST_PIPELINE = (
    "filesrc location=data/DJI_0020.MP4 ! qtdemux ! h264parse ! nvv4l2decoder ! "
    "video/x-raw(memory:NVMM),format=NV12 ! nvvidconv ! "
    "video/x-raw,width=960,height=540,format=BGRx ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink drop=1 sync=0"
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
half = device.startswith("cuda")

def load_model():
    try:
        if USE_TENSORRT and os.path.isfile(ENGINE_PATH):
            print(f"[INFO] Loading existing engine: {ENGINE_PATH}")
            return YOLO(ENGINE_PATH)
        print(f"[INFO] Loading weights: {MODEL_PATH}")
        m = YOLO(MODEL_PATH)
        if USE_TENSORRT and device.startswith("cuda"):
            print("[INFO] Exporting TensorRT engine (this may take a while)...")
            m.export(format="engine", half=half, device=device, imgsz=IMG_SIZE)
            print("[INFO] Engine export done.")
            return YOLO(ENGINE_PATH)
        return m
    except Exception as e:
        print(f"[ERROR] Model load/export failed: {e}")
        return None

model = load_model()
if model is None:
    raise SystemExit("Model failed to initialize.")

# Warmup (small)
try:
    if device.startswith("cuda"):
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        _ = model(dummy, imgsz=IMG_SIZE, device=device, half=half, verbose=False)
        print("[INFO] Warmup done.")
except Exception as e:
    print(f"[WARN] Warmup failed: {e}")

# Video capture
if USE_GSTREAMER:
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[WARN] GStreamer pipeline failed. Falling back to standard capture.")
        USE_GSTREAMER = False

if not USE_GSTREAMER:
    cap = cv2.VideoCapture("data/DJI_0020.MP4")
    if not cap.isOpened():
        raise SystemExit("[ERROR] Could not open video file.")

print("[INFO] Starting inference loop.")
frame_count = 0
proc_count = 0
start_time = time.time()

try:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("[INFO] End of stream or read failure.")
            break

        if FRAME_SKIP and (frame_count % (FRAME_SKIP + 1)) != 0:
            frame_count += 1
            continue

        # Inference
        try:
            results = model(frame, device=device, imgsz=IMG_SIZE, half=half, verbose=False)
        except Exception as e:
            print(f"[ERROR] Inference error: {e}")
            break

        # (Avoid plotting for speed; uncomment if needed)
        # if SHOW_WINDOW:
        #     annotated = results[0].plot()
        #     cv2.imshow("YOLO Fast", annotated)
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break

        frame_count += 1
        proc_count += 1

        if PRINT_FPS and proc_count % 60 == 0:
            elapsed = time.time() - start_time
            fps = proc_count / elapsed
            print(f"[INFO] Processed: {proc_count} | Avg FPS: {fps:.2f}")

except KeyboardInterrupt:
    print("[INFO] Interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Finished.")
