# Kaggle Training Script - Resume V15 (Continue from V14 Epoch 19)
# ==================================================================
# This script resumes training from the version_14 checkpoint
# Target: Continue toward the long-run total (run in chunks per session)
#
# INSTRUCTIONS:
# 1. Upload version_14/runs/detect/train_v5_stage_b_resume2 as a Dataset:
#    - Name it "v14-checkpoint" (slug: v14-checkpoint)
#    - Structure: v14-checkpoint/train_v5_stage_b_resume2/weights/last.pt
# 2. Add these datasets to your notebook:
#    - v14-checkpoint (your checkpoint)
#    - custom-dataset (your merged_dataset)
#    - doclaynet-v1-2-yolo (public DocLayNet YOLO format)
# 3. Run this script with T4 x2 GPU

import os
print("Installing dependencies...")
os.system('pip install "numpy==2.0.2" "ultralytics>=8.3.0,<9" --no-warn-conflicts')

import zipfile
import shutil
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# ============================================================================
# RESUME CONFIGURATION - UPDATE THESE PATHS IF YOUR DATASET SLUG IS DIFFERENT
# ============================================================================
# For TRUE RESUME, we need the entire run directory (not just weights)
RESUME_RUN_DIR_PATHS = [
    "/kaggle/input/v14-checkpoint/train_v5_stage_b_resume2",
    "/kaggle/input/v14-checkpoint/runs/detect/train_v5_stage_b_resume2",
    "/kaggle/input/v14-checkpoint",
]

# Fallback to weights-only if run dir not found
RESUME_WEIGHTS_PATHS = [
    "/kaggle/input/v14-checkpoint/train_v5_stage_b_resume2/weights/last.pt",
    "/kaggle/input/v14-checkpoint/runs/detect/train_v5_stage_b_resume2/weights/last.pt",
    "/kaggle/input/v14-checkpoint/weights/last.pt",
]

PREVIOUS_EPOCHS_COMPLETED = 19  # From version_14 train_v5_stage_b_resume2
TARGET_TOTAL_EPOCHS = 37        # Target total - YOLO will use this when resume=True
USE_TRUE_RESUME = True          # Set to True for proper epoch continuation (19/37, 20/37...)

def sanitize_checkpoint_for_ddp(ckpt_path: str) -> str:
    """
    Strip optimizer state from checkpoint to avoid DDP mismatch errors.
    Keeps model weights and epoch info for proper resume.
    """
    import torch
    
    print(f"Sanitizing checkpoint for DDP compatibility: {ckpt_path}")
    
    try:
        # PyTorch 2.6+ defaults to weights_only=True, but Ultralytics checkpoints
        # contain custom classes that need weights_only=False
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Warning: failed to load checkpoint: {e}")
        return ckpt_path
    
    # Check if optimizer state exists and might cause issues
    if "optimizer" in ckpt:
        print("  - Removing optimizer state (causes DDP mismatch)")
        del ckpt["optimizer"]
    
    # Keep these for proper resume:
    # - 'model': model weights
    # - 'epoch': current epoch number  
    # - 'train_args': training arguments
    # - 'train_metrics': training metrics
    
    keys_to_keep = ['model', 'epoch', 'train_args', 'train_metrics', 'date', 'version']
    removed_keys = [k for k in list(ckpt.keys()) if k not in keys_to_keep and k != 'optimizer']
    
    print(f"  - Kept keys: {[k for k in ckpt.keys()]}")
    
    # Overwrite the original checkpoint (YOLO resume looks for this exact path)
    try:
        torch.save(ckpt, ckpt_path)
        print(f"  - Overwrote checkpoint (optimizer removed): {ckpt_path}")
        return ckpt_path
    except Exception as e:
        print(f"  - Warning: failed to save sanitized checkpoint: {e}")
        return ckpt_path

# ============================================================================
# Dataset Configuration
# ============================================================================
working_dir = '/kaggle/working/data'
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

print("Searching for datasets...")
custom_dataset_paths = [
    '/kaggle/input/custom-dataset/merged_dataset',
    '/kaggle/input/doclayout-dataset/data/merged_dataset',
    '/kaggle/input/doclayout-dataset/merged_dataset',
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

if not custom_root:
    dataset_zip = '/kaggle/input/doclayout-dataset/dataset.zip'
    if os.path.exists(dataset_zip):
        print(f"Extracting {dataset_zip}...")
        with zipfile.ZipFile(dataset_zip, 'r') as zf:
            zf.extractall(working_dir)
        for candidate in [
            os.path.join(working_dir, 'merged_dataset'),
            os.path.join(working_dir, 'data', 'merged_dataset'),
        ]:
            if os.path.exists(os.path.join(candidate, 'dataset.yaml')):
                custom_root = candidate
                break

print(f"Custom Dataset Root: {custom_root}")

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
        if os.path.isdir(os.path.join(candidate, "train", "images")):
            return candidate
    return None

doclaynet_root = resolve_doclaynet_root(public_doclaynet_paths)
if doclaynet_root:
    print(f"Found Public DocLayNet at: {doclaynet_root}")
else:
    print("WARNING: Public DocLayNet not found.")

# Copy Custom to Working (for writable)
if custom_root:
    dest_path = f"{working_dir}/merged_dataset"
    if not os.path.exists(dest_path) and custom_root != dest_path:
        print(f"Copying custom data to {dest_path}...")
        shutil.copytree(custom_root, dest_path)
    custom_root = dest_path

# ============================================================================
# Update YAML Path
# ============================================================================
import yaml

if not custom_root:
    raise FileNotFoundError("Custom dataset not found!")

yaml_path = os.path.join(custom_root, "dataset.yaml")
print(f"Updating dataset config: {yaml_path}")

with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

data_config['path'] = working_dir
data_config['train'] = [f"{working_dir}/merged_dataset/train/images"]
data_config['val'] = [f"{working_dir}/merged_dataset/val/images"]

if doclaynet_root:
    dl_train = os.path.join(doclaynet_root, "images", "train")
    dl_val = os.path.join(doclaynet_root, "images", "val")
    if not os.path.exists(dl_train):
        dl_train = os.path.join(doclaynet_root, "train", "images")
        dl_val = os.path.join(doclaynet_root, "valid", "images")
    if os.path.exists(dl_train):
        print(f"Adding DocLayNet Train: {dl_train}")
        data_config['train'].append(dl_train)
        if os.path.exists(dl_val):
            data_config['val'].append(dl_val)

# Ensure 11 classes for DocLayNet compatibility
if data_config.get('nc', 0) < 11:
    print("Upgrading class count to 11 for DocLayNet compatibility.")
    data_config['nc'] = 11
    names = data_config.get('names', [])
    if isinstance(names, dict):
        max_id = max(names.keys()) if names else -1
        for i in range(max_id + 1, 11):
            names[i] = f"class_{i}"
    elif isinstance(names, list):
        while len(names) < 11:
            names.append(f"class_{len(names)}")
    data_config['names'] = names

with open(yaml_path, 'w') as f:
    yaml.safe_dump(data_config, f)

print("Updated dataset.yaml:")
print(yaml.safe_dump(data_config))

# ============================================================================
# Find Resume Checkpoint / Run Directory
# ============================================================================
resume_run_dir = None
resume_ckpt = None

# First, try to find the full run directory (for true resume)
for path in RESUME_RUN_DIR_PATHS:
    if os.path.exists(path):
        # Check if this is a valid run directory (has weights/last.pt)
        weights_path = os.path.join(path, "weights", "last.pt")
        if os.path.exists(weights_path):
            resume_run_dir = path
            resume_ckpt = weights_path
            print(f"✅ Found resume run directory: {path}")
            break

# Fallback to weights-only if no run dir found
if not resume_ckpt:
    for path in RESUME_WEIGHTS_PATHS:
        if os.path.exists(path):
            resume_ckpt = path
            print(f"⚠️  Found weights only (no run dir): {path}")
            break

if not resume_ckpt:
    print("❌ ERROR: Resume checkpoint not found!")
    print("Available /kaggle/input directories:")
    if os.path.exists("/kaggle/input"):
        for d in os.listdir("/kaggle/input"):
            print(f"  - {d}")
            subdir = os.path.join("/kaggle/input", d)
            if os.path.isdir(subdir):
                for f in os.listdir(subdir)[:5]:
                    print(f"      {f}")
    raise FileNotFoundError("Resume checkpoint not found. Check dataset upload.")

# ============================================================================
# Training - Resume from V14
# ============================================================================
from ultralytics import YOLO

def log_section(title: str) -> None:
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")

def log_kv(pairs) -> None:
    if not pairs:
        return
    width = max(len(k) for k, _ in pairs)
    for key, value in pairs:
        print(f"{key.ljust(width)} : {value}")

def human_duration(seconds: float) -> str:
    total = int(seconds)
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

project_dir = "/kaggle/working/runs/detect"
device_ids = "0,1"

# Determine resume mode
can_true_resume = USE_TRUE_RESUME and resume_run_dir is not None

if can_true_resume:
    log_section(f"TRUE RESUME: Continue from Epoch {PREVIOUS_EPOCHS_COMPLETED} → {TARGET_TOTAL_EPOCHS}")
    
    # Copy entire run directory to working (required for DDP and writable access)
    target_run_dir = os.path.join(project_dir, "train_v5_stage_b_resume2")
    if not os.path.exists(target_run_dir):
        print(f"Copying run directory to {target_run_dir}...")
        shutil.copytree(resume_run_dir, target_run_dir)
    
    working_ckpt = os.path.join(target_run_dir, "weights", "last.pt")
    
    # CRITICAL: Sanitize checkpoint to remove optimizer state (causes DDP mismatch)
    working_ckpt = sanitize_checkpoint_for_ddp(working_ckpt)
    
    log_kv([
        ("start_time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("resume_mode", "TRUE RESUME (epochs continue, fresh optimizer)"),
        ("run_directory", target_run_dir),
        ("checkpoint", working_ckpt),
        ("resume_from_epoch", str(PREVIOUS_EPOCHS_COMPLETED)),
        ("target_epochs", str(TARGET_TOTAL_EPOCHS)),
        ("model", "YOLO11s (Small)"),
        ("device", device_ids),
    ])
    
    # Load model and train with resume=True
    model = YOLO(working_ckpt)
    stage_start = time.perf_counter()
    
    # True resume - YOLO will continue from where it left off (fresh optimizer due to sanitization)
    model.train(
        resume=True,  # TRUE RESUME - epochs continue as 19/37, 20/37, etc.
        device=device_ids,
    )
    
    run_name = "train_v5_stage_b_resume2"  # Same name since we're continuing
    
else:
    # Fallback: weights-only resume (epochs reset to 1, but model continues learning)
    log_section("WEIGHTS-ONLY RESUME (epoch counter resets)")
    
    remaining_epochs = TARGET_TOTAL_EPOCHS - PREVIOUS_EPOCHS_COMPLETED
    
    log_kv([
        ("start_time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("resume_mode", "WEIGHTS-ONLY (fresh epoch counter)"),
        ("checkpoint", resume_ckpt),
        ("new_epochs", str(remaining_epochs)),
        ("note", "Model weights from epoch 19, but showing 1/N"),
        ("model", "YOLO11s (Small)"),
        ("device", device_ids),
    ])
    
    # Copy checkpoint to a stable path for DDP access
    working_ckpt = Path("/kaggle/working") / "last_from_v14.pt"
    shutil.copy2(resume_ckpt, working_ckpt)
    if not working_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint copy failed: {working_ckpt}")
    print(f"Copied checkpoint to: {working_ckpt}")
    
    # Load model and train
    model = YOLO(str(working_ckpt))
    stage_start = time.perf_counter()
    
    run_name = "train_v5_stage_b_resume3"
    
    model.train(
        data=yaml_path,
        epochs=remaining_epochs,
        imgsz=1024,
        batch=8,
        project=project_dir,
        name=run_name,
        device=device_ids,
        resume=False,  # Fresh optimizer
        exist_ok=True,
    )

stage_elapsed = time.perf_counter() - stage_start
print(f"\nTraining runtime: {human_duration(stage_elapsed)}")

# ============================================================================
# Summarize Results
# ============================================================================
def summarize_results(run_dir: str) -> None:
    results_csv = os.path.join(run_dir, "results.csv")
    if not os.path.exists(results_csv):
        print(f"No results.csv found at {results_csv}")
        return

    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("results.csv is empty.")
        return

    latest = rows[-1]
    print("\n📊 Final Metrics:")
    log_kv([
        ("epoch", latest.get("epoch", "?")),
        ("mAP50", latest.get("metrics/mAP50(B)", "?")),
        ("mAP50-95", latest.get("metrics/mAP50-95(B)", "?")),
        ("precision", latest.get("metrics/precision(B)", "?")),
        ("recall", latest.get("metrics/recall(B)", "?")),
    ])

    # Find best mAP
    def best_metric(col: str):
        best_val, best_epoch = None, None
        for row in rows:
            raw = row.get(col, "")
            if raw:
                try:
                    val = float(raw)
                    if best_val is None or val > best_val:
                        best_val = val
                        best_epoch = row.get("epoch", "?")
                except ValueError:
                    pass
        return best_epoch, best_val

    for col in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        epoch, val = best_metric(col)
        if val:
            print(f"🏆 Best {col}: {val:.4f} (epoch {epoch})")

    weights_dir = os.path.join(run_dir, "weights")
    for fname in ("best.pt", "last.pt"):
        fpath = os.path.join(weights_dir, fname)
        if os.path.exists(fpath):
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"💾 {fname}: {size_mb:.1f} MB")

run_dir = os.path.join(project_dir, run_name)
summarize_results(run_dir)

# ============================================================================
# Zip Outputs
# ============================================================================
print("\n📦 Zipping outputs...")
os.system(
    'zip -r -1 -q results_v15.zip /kaggle/working/runs '
    '-x "/kaggle/working/data/*"'
)
print("✅ Download 'results_v15.zip' from Output.")
