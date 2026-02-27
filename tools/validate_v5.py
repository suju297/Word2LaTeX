from ultralytics import YOLO

# Load model
model = YOLO('models/model_v5.pt')

# Run validation
# Using 'train' data as 'val' proxy since real val is missing in download
results = model.val(
    data='kaggle_export/version12/data/merged_dataset/dataset.yaml',
    imgsz=1024,
    batch=4,
    device='mps' # Use Mac GPU if available, else 'cpu'
)

# Print results
print("\n" + "="*50)
print("PER-CLASS METRICS (Approximate on subset)")
print("="*50)
print(f"{'Class':<15} {'mAP50':<10} {'mAP50-95':<10}")
print("-" * 35)

for i, c in enumerate(results.names.values()):
    # Ultralytics maps might be indexed by ID
    map50 = results.box.map50[i]
    map5095 = results.box.map75[i] # Approximate or access map50-95 vector
    # Accessing specific map50-95 for class i
    # results.box.maps is array of shape (nc, ) containing map50-95 for each class
    map5095_real = results.box.maps[i]
    
    print(f"{c:<15} {map50:.3f}      {map5095_real:.3f}")
