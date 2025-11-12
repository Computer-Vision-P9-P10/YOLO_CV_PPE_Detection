import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best.pt")

# Open the video file
video_path = "data/DJI_0020.MP4"
cap = cv2.VideoCapture(video_path)

# Loop through video frames
while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # Run YOLO inference on the frame (force CPU)
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLO Detection", annotated_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
