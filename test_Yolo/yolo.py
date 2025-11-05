from ultralytics import YOLO
import os
import glob
import numpy as np


def validate_model(weights_path):
    """Validate a trained YOLO model"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    model = YOLO(weights_path)
    val_results = model.val()
    print("Validation done")
    return val_results

def run_inference(weights_path, image_path, conf=0.4, iou=0.5, save=True, save_crop=True, show=False):
    """Run inference and simple postprocessing prints"""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    model = YOLO(weights_path)

    # Inference
    results = model(
        image_path,
        conf=conf,
        iou=iou,
        save=save,
        save_crop=save_crop,
        show=show,
    )

    # Post processing
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            # Typical attributes boxes.xyxy, boxes.cls, boxes.conf 
            try:
                print(f"Boxes xyxy: {boxes.xyxy.cpu().numpy()}")
            except Exception:
                print(f"Boxes xyxy: {getattr(boxes, 'xyxy', None)}")
            try:
                print(f"Classes: {boxes.cls.cpu().numpy()}")
                print(f"Confidence: {boxes.conf.cpu().numpy()}")
            except Exception:
                # fallback to printing raw attrs
                print(f"Classes: {getattr(boxes, 'cls', None)}")
                print(f"Confidence: {getattr(boxes, 'conf', None)}")

        masks = getattr(result, "masks", None)
        if masks is not None:
            # masks.xy or masks.data may be available depending on model
            print(f"Masks: {getattr(masks, 'xy', getattr(masks, 'data', None))}")

        keypoints = getattr(result, "keypoints", None)
        if keypoints is not None:
            print(f"Detected keypoints: {getattr(keypoints, 'xy', None)}")

        probs = getattr(result, "probs", None)
        if probs is not None:
            print(f"Predicted Class: {getattr(probs, 'top1', None)}")

        # CSV output if supported
        try:
            print(f"CSV Output: {result.to_csv()}")
        except Exception:
            pass

    print("Inference and post-processing complete")


# Main script execution
if __name__ == "__main__":
    # Configuration
    model_config = "yolo11s.yaml"
    data_yaml = "./dataset/data.yaml"
    device = "mps" # for mac
    
    # Insert path to weigths 
    weights_path = "./tilt/train9/weights/best.pt"
    
    if os.path.exists(weights_path):
        # Validate the model
        val_results = validate_model(weights_path)
        
        # Run inference
        run_inference(
            weights_path=weights_path,
            image_path="./dataset/test/images/",
            conf=0.25,
            iou=0.45,
            save=True,
            save_crop=False,
            show=False
        )
   
        
    else:
        print(f"Error: weights file '{weights_path}' not found. Please train the model or provide a valid weights file.")