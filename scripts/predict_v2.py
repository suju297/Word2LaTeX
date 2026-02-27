import sys
from ultralytics import YOLO
from pathlib import Path
import cv2

def predict():
    # Load model
    model_path = "models/model_v2.pt"
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Image
    img_path = "data/merged_dataset/val/images/doclaynet_train_1007.png"
    if not Path(img_path).exists():
        print(f"Error: Image not found at {img_path}")
        return
        
    print(f"Running inference on {img_path}...")
    results = model(img_path)
    
    # Print results
    for r in results:
        print("\nDetected Objects:")
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            print(f" - {name}: {conf:.2f} (Box: {box.xyxy[0].tolist()})")
            
    print("\nSuccess! Model V2 is working locally.")

if __name__ == "__main__":
    predict()
