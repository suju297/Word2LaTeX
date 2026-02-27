import argparse
from pathlib import Path
from ultralytics import YOLO

def train_yolo(
    data_path: str = "data/merged_dataset/dataset.yaml",
    model_size: str = "n",  # n, s, m, l, x
    epochs: int = 10,
    batch_size: int = 16,
    device: str = "mps",  # use mps for Mac if available, else cpu
    project: str = "runs/detect",
    name: str = "train_merged_v1"
):
    """Train YOLOv8 model on document layout dataset."""
    
    # Load a model
    # load a pretrained model (recommended for training)
    model = YOLO(f"yolov8{model_size}.pt")  
    
    print(f"Training YOLOv8{model_size} on {data_path} for {epochs} epochs...")
    
    # Train the model
    # We use 'mps' for Apple Silicon acceleration if available
    try:
        results = model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            plots=True,
            save=True,
            # Augmentation settings for documents
            fliplr=0.0,  # document text shouldn't be flipped horizontally
            mosaic=0.5,  # less mosaic for documents to preserve structure
            scale=0.2,   # less scale variation
            degrees=0.0, # no rotation for documents usually
        )
        
        print(f"Training complete. Results saved to {project}/{name}")
        
    except Exception as e:
        print(f"Training failed with device={device}. Retrying with device='cpu'...")
        results = model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            device='cpu',
            project=project,
            name=name,
            plots=True,
            save=True,
            fliplr=0.0,
            mosaic=0.5,
            scale=0.2,
            degrees=0.0,
        )

    # Validate the model
    print("Validating model...")
    metrics = model.val()
    print(f"Validation MAP50-95: {metrics.box.map}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/merged_dataset/dataset.yaml")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--name", type=str, default="train_merged_v1")
    
    args = parser.parse_args()
    
    train_yolo(
        data_path=args.data,
        model_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        name=args.name
    )
