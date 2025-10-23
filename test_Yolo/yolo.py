from ultralytics import YOLO
import os
import glob
import numpy as np

class YOLO_trainer:
    def __init__(self, model_config, data_yaml, device="cpu"):
        self.model_config = model_config  # yolo model yolo11s.yaml
        self.data_yaml = data_yaml  # dataset yaml file
        self.device = device  # cpu or mps (or gpu ids)
        self.trained_weights = None  # Updated after training

    def train_model(self, epochs=100, resume=True):
         # parameter can be adjust based on https://docs.ultralytics.com/usage/cfg/
        model = YOLO(self.model_config)
        train = model.train(
            data=self.data_yaml,
            epochs=epochs,
            device=self.device,
            resume=resume,
        )

        #  find best .pt model or pick most recent (weights dir)
        save_dir = getattr(train, "save_dir", None)
        best_path_pt = None
        if save_dir:
            weights_dir = os.path.join(save_dir, "weights")
            candidate = os.path.join(weights_dir, "best.pt")
            if os.path.exists(candidate):
                best_path_pt = candidate
            else:
                # fallback to newest .pt in weights_dir
                if os.path.isdir(weights_dir):
                    pts = sorted(
                        glob.glob(os.path.join(weights_dir, "*.pt")),
                        key=os.path.getmtime,
                        reverse=True,
                    )
                    if pts:
                        best_path_pt = pts[0]

        if best_path_pt is None:
            raise FileNotFoundError("did not find best .pt ")
        
        self.trained_weights = best_path_pt
        print(f"Training finished, saved weights at {best_path_pt}")

    def validate(self):
        assert self.trained_weights is not None, "Model needs to be trained first"
        model = YOLO(self.trained_weights)
        val_results = model.val()  # passing data is possible if needed
        print("Validation done")
        return val_results

    def postprocess_and_inference(self, image_path, conf=0.4, iou=0.5, save=True, save_crop=True, show=False):
        #Run inference and simple postprocessing prints
        assert self.trained_weights is not None, "Model needs to be trained first"
        model = YOLO(self.trained_weights)

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
                # masks.xy or masks.data may be available depending on model -> proberly need to find out if it's .xy or data
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

"""
Example from github copilot :)
if __name__ == "__main__":
    # Minimal example — adjust model/data paths and device ("mps" on Mac, "cpu" or GPU id)
    trainer = YOLO_trainer(
        model_config="yolo11s.yaml",
        data_yaml="dataset/data.yaml",
        device="mps"
    )

    # To train (uncomment to run)
    # trainer.train_model(epochs=15, resume=True)

    # Or set trained_weights if you already have a .pt file
    # trainer.trained_weights = "runs/train/exp/weights/best.pt"

    # Validate (requires trainer.trained_weights)
    # val_results = trainer.validate()
    # print(val_results)

    # Inference example (will run even without training if you set trainer.trained_weights)
    out = trainer.postprocess_and_inference(
        "dataset/test/images/00504_jpg.rf.83a7e15ff43c0f937313abc601afa6c5.jpg",
        conf=0.25,
        iou=0.45,
        save=True,
        save_crop=False,
        show=False
    )

    print("Done")
```# filepath: /Users/nicolaibergulff/Desktop/AAU/9.Semester/P9/YOLOv11_CV_PPE_Detection/test_Yolo/yolo.py
# ...existing code...
if __name__ == "__main__":
    # Minimal example — adjust model/data paths and device ("mps" on Mac, "cpu" or GPU id)
    trainer = YOLO_trainer(
        model_config="yolo11s.yaml",
        data_yaml="dataset/data.yaml",
        device="mps"
    )

    # To train (uncomment to run)
    # trainer.train_model(epochs=15, resume=True)

    # Or set trained_weights if you already have a .pt file
    # trainer.trained_weights = "runs/train/exp/weights/best.pt"

    # Validate (requires trainer.trained_weights)
    # val_results = trainer.validate()
    # print(val_results)

    # Inference example (will run even without training if you set trainer.trained_weights)
    out = trainer.postprocess_and_inference(
        "dataset/test/images/00504_jpg.rf.83a7e15ff43c0f937313abc601afa6c5.jpg",
        conf=0.25,
        iou=0.45,
        save=True,
        save_crop=False,
        show=False
    )

    print("Done")
"""