# Kaggle Training Script - Fine-Tune V17 (From V23 Epoch 37)
# ============================================================
# This script FINE-TUNES from V23 checkpoint using lower learning rate
# Target: 20 more epochs with proper fine-tuning settings
#
# DATASETS NEEDED:
# - version23-checkpoint (your V23 checkpoint)
# - custom-dataset (merged_dataset)
# - doclaynet-v1-2-yolo (public DocLayNet)

import os

try:
    import ultralytics
    print(f"ultralytics already installed: {ultralytics.__version__}")
except ImportError:
    print("Installing dependencies...")
    os.system('pip install "numpy==2.0.2" "ultralytics>=8.3.0,<9" --no-warn-conflicts')

import zipfile
import shutil
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

# ============================================================================
# CONFIGURATION
# ============================================================================
CHECKPOINT_PATHS = [
    "/kaggle/input/version23-checkpoint/train_v5_stage_b_resume/weights/best.pt",
    "/kaggle/input/version23-checkpoint/train_v5_stage_b_resume/weights/last.pt",
    "/kaggle/input/version23-checkpoint/weights/best.pt",
]

FINETUNE_EPOCHS = 20  # Additional epochs to train
LEARNING_RATE = 0.0001  # 100x lower than default (0.01) for fine-tuning
WARMUP_EPOCHS = 0.5  # Minimal warmup

# ============================================================================
# Dataset Configuration
# ============================================================================
import yaml

working_dir = '/kaggle/working/data'
os.makedirs(working_dir, exist_ok=True)

print("Searching for datasets...")

custom_dataset_paths = [
    '/kaggle/input/custom-dataset/merged_dataset',
    '/kaggle/input/doclayout-dataset/data/merged_dataset',
]

public_doclaynet_paths = [
    '/kaggle/input/doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection',
    '/kaggle/input/doclaynet-v1-2-yolo',
]

# Find Custom Dataset
custom_root = None
for p in custom_dataset_paths:
    if os.path.exists(p):
        custom_root = p
        break

print(f"Custom Dataset Root: {custom_root}")

# Find DocLayNet
def resolve_doclaynet_root(paths):
    for p in paths:
        if not os.path.exists(p):
            continue
        candidate = p
        nested = os.path.join(candidate, "DocLayNet-v1.2-YOLODetection")
        if os.path.isdir(nested):
            candidate = nested
        if os.path.isdir(os.path.join(candidate, "images", "train")):
            return candidate
    return None

doclaynet_root = resolve_doclaynet_root(public_doclaynet_paths)
print(f"DocLayNet Root: {doclaynet_root}")

# Copy Custom to Working
if custom_root:
    dest_path = f"{working_dir}/merged_dataset"
    if not os.path.exists(dest_path):
        print(f"Copying custom data to {dest_path}...")
        shutil.copytree(custom_root, dest_path)
    custom_root = dest_path

if not custom_root:
    raise FileNotFoundError("Custom dataset not found!")

# Update YAML
yaml_path = os.path.join(custom_root, "dataset.yaml")
with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

data_config['path'] = working_dir
data_config['train'] = [f"{working_dir}/merged_dataset/train/images"]
data_config['val'] = [f"{working_dir}/merged_dataset/val/images"]

if doclaynet_root:
    dl_train = os.path.join(doclaynet_root, "images", "train")
    if os.path.exists(dl_train):
        data_config['train'].append(dl_train)
        dl_val = os.path.join(doclaynet_root, "images", "val")
        if os.path.exists(dl_val):
            data_config['val'].append(dl_val)

# Ensure 11 classes
if data_config.get('nc', 0) < 11:
    data_config['nc'] = 11
    names = data_config.get('names', {})
    if isinstance(names, dict):
        max_id = max(names.keys()) if names else -1
        for i in range(max_id + 1, 11):
            names[i] = f"class_{i}"
    data_config['names'] = names

with open(yaml_path, 'w') as f:
    yaml.safe_dump(data_config, f)

print("Dataset config updated.")

# ============================================================================
# Find Checkpoint
# ============================================================================
checkpoint = None
for path in CHECKPOINT_PATHS:
    if os.path.exists(path):
        checkpoint = path
        print(f"✅ Found checkpoint: {path}")
        break

if not checkpoint:
    print("❌ Checkpoint not found!")
    print("Searched paths:")
    for p in CHECKPOINT_PATHS:
        print(f"  - {p}")
    raise FileNotFoundError("Checkpoint not found")

# Copy to working directory
working_ckpt = Path("/kaggle/working/checkpoint.pt")
shutil.copy2(checkpoint, working_ckpt)
print(f"Copied to: {working_ckpt}")

# ============================================================================
# Fine-Tune Training
# ============================================================================
from ultralytics import YOLO

print("\n" + "="*80)
print("FINE-TUNING FROM V23 CHECKPOINT")
print("="*80)
print(f"Checkpoint    : {checkpoint}")
print(f"Epochs        : {FINETUNE_EPOCHS}")
print(f"Learning Rate : {LEARNING_RATE} (100x lower than default)")
print(f"Warmup Epochs : {WARMUP_EPOCHS}")
print(f"Device        : 0,1")
print("="*80 + "\n")

model = YOLO(str(working_ckpt))
start_time = time.perf_counter()

# Fine-tuning with conservative settings
results = model.train(
    data=yaml_path,
    epochs=FINETUNE_EPOCHS,
    imgsz=1024,
    batch=8,
    lr0=LEARNING_RATE,           # Low LR for fine-tuning
    lrf=0.1,                     # Final LR = lr0 * lrf = 0.00001
    warmup_epochs=WARMUP_EPOCHS, # Minimal warmup
    project="/kaggle/working/runs/detect",
    name="finetune_v17",
    device="0,1",
    resume=False,
    exist_ok=True,
    patience=50,                 # Early stopping patience
    save_period=5,               # Save every 5 epochs
)

elapsed = time.perf_counter() - start_time
hours = int(elapsed // 3600)
mins = int((elapsed % 3600) // 60)
print(f"\nTraining completed in {hours}h {mins}m")

# ============================================================================
# Results Summary
# ============================================================================
run_dir = "/kaggle/working/runs/detect/finetune_v17"
results_csv = os.path.join(run_dir, "results.csv")

if os.path.exists(results_csv):
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if rows:
        latest = rows[-1]
        print("\n📊 Final Metrics:")
        print(f"  mAP50     : {latest.get('metrics/mAP50(B)', '?')}")
        print(f"  mAP50-95  : {latest.get('metrics/mAP50-95(B)', '?')}")
        print(f"  Precision : {latest.get('metrics/precision(B)', '?')}")
        print(f"  Recall    : {latest.get('metrics/recall(B)', '?')}")

# Zip outputs
print("\n📦 Zipping outputs...")
os.system('zip -r -1 -q results_v17.zip /kaggle/working/runs -x "/kaggle/working/data/*"')
print("✅ Download 'results_v17.zip' from Output.")
