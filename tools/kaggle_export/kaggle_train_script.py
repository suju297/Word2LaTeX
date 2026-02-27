# Kaggle Training Script for Document Layout Analysis (YOLOv8)
# ==========================================================
# Instructions:
# 1. Create a New Notebook on Kaggle.
# 2. Add Data: Upload 'dataset.zip' as a private Dataset.
# 3. Copy-paste this code into the notebook cells.
# 4. Turn on GPU (T4 x2 or P100) in Accelerator settings.

# Cell 1: Setup Environment
# -------------------------
import os
print("Installing dependencies...")

# CRITICAL FIX for Kaggle:
# The base environment has NumPy 2.0.2. 
# 1. 'pip install ultralytics' tries to upgrade to 2.2.6 (Breaks Matplotlib/Array API)
# 2. Downgrading to 1.x (Breaks Torch/Binary compatibility)
# SOLUTION: Force pip to keep the exact existing version (2.0.2).
os.system('pip install "numpy==2.0.2" "ultralytics>=8.3.0,<9" --no-warn-conflicts')

# Cell 2: Prepare Dataset
# -----------------------
import zipfile
import shutil
import os
from pathlib import Path

# Define destination (writable)
working_dir = '/kaggle/working/data'
if not os.path.exists(working_dir):
    os.makedirs(working_dir)

# 1. Search for datasets
print("Searching for datasets...")
custom_dataset_paths = [
    '/kaggle/input/custom-dataset/merged_dataset',
    '/kaggle/input/doclayout-dataset/data/merged_dataset',
    '/kaggle/input/doclayout-dataset/merged_dataset',
    '/kaggle/input/data/merged_dataset'
]

public_doclaynet_paths = [
    '/kaggle/input/doclaynet-v1-2-yolo/DocLayNet-v1.2-YOLODetection', # User Provided
    '/kaggle/input/doclaynet-v1-2-yolo',
    '/kaggle/input/doclaynet/DocLayNet_core',
    '/kaggle/input/doclaynet-dataset/DocLayNet_core',
    '/kaggle/input/doclaynet-v1-1/DocLayNet_core',
]

# Find Custom Dataset
custom_root = None
for p in custom_dataset_paths:
    if os.path.exists(p):
        custom_root = p
        break

if not custom_root:
    # Fallback to Zip
    dataset_zip = '/kaggle/input/doclayout-dataset/dataset.zip'
    if os.path.exists(dataset_zip):
        print(f"Extracting {dataset_zip}...")
        with zipfile.ZipFile(dataset_zip, 'r') as zf:
            zf.extractall(working_dir)

        candidate_roots = [
            os.path.join(working_dir, 'merged_dataset'),
            os.path.join(working_dir, 'data', 'merged_dataset'),
        ]
        for candidate in candidate_roots:
            if os.path.exists(os.path.join(candidate, 'dataset.yaml')):
                custom_root = candidate
                break
        if not custom_root:
            for root, _, files in os.walk(working_dir):
                if 'dataset.yaml' in files:
                    custom_root = root
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

# Find Public DocLayNet
doclaynet_root = resolve_doclaynet_root(public_doclaynet_paths)
if doclaynet_root:
    print(f"Found Public DocLayNet at: {doclaynet_root}")
else:
    print("WARNING: Public DocLayNet not found. Add it via 'Add Data' -> Search 'DocLayNet'.")

# Copy Custom to Working (for writable)
if custom_root:
    dest_path = f"{working_dir}/merged_dataset"
    if not os.path.exists(dest_path) and custom_root != dest_path:
        print(f"Copying custom data to {dest_path}...")
        shutil.copytree(custom_root, dest_path)
    custom_root = dest_path # Update to writable path

# Cell 3: Update YAML Path
# ------------------------
import yaml

if not custom_root:
    raise FileNotFoundError(
        "Custom dataset not found. Upload merged_dataset (folder or dataset.zip) and "
        "update custom_dataset_paths if your dataset slug differs from doclayout-dataset."
    )

yaml_path = os.path.join(custom_root, "dataset.yaml")
print(f"Updating dataset config: {yaml_path}")

# Load YAML
with open(yaml_path, 'r') as f:
    data_config = yaml.safe_load(f)

# FIX PATHS: Override absolute local paths with Kaggle paths
# HYBRID DATASET CONFIGURATION
# We list MULTIPLE root paths. YOLO will look for images in all of them.

# 1. Custom Dataset Paths
custom_train = f"{working_dir}/merged_dataset/train/images"
custom_val = f"{working_dir}/merged_dataset/val/images"

# 2. Public DocLayNet Paths (Read-Only Input)
# Structure on Kaggle usually: /kaggle/input/doclaynet/png/train, etc.
# But we need to verify. Assuming standard structure:
doclaynet_train = "/kaggle/input/doclaynet/png/train" 
doclaynet_val = "/kaggle/input/doclaynet/png/val"

# If exact path unknown, we just use Custom for now, user can uncomment
# data_config['train'] = [custom_train, doclaynet_train]
# data_config['val'] = [custom_val, doclaynet_val]

# For SAFETY: We start with just Custom, but provide the list for the user to enable
if doclaynet_root:
     # Heuristic to find image folders
     print(f"Inspecting {doclaynet_root}...")
     
     # Check for standard "images/train" structure
     dl_images = os.path.join(doclaynet_root, "images")
     if os.path.exists(dl_images):
         # It has an 'images' folder (Common in YOLO datasets)
         if os.path.exists(os.path.join(dl_images, "train")):
             print(f"Found train dir at {os.path.join(dl_images, 'train')}")
             # logic to append is handled below in the dataset.yaml section
         else:
             # Maybe flat structure?
             print(f"Warning: 'images' folder found but 'images/train' missing at {dl_images}")
     
     # Try to find 'train' folder directly in root
     for root, dirs, files in os.walk(doclaynet_root):
         if 'train' in dirs:
             print(f"Found train dir at {os.path.join(root, 'train')}")
             # logic to append...
             
# SIMPLIFICATION:
# We will write the config to use RELATIVE paths if possible, or list absolute
data_config['path'] = working_dir # Base
data_config['train'] = [
    f"{working_dir}/merged_dataset/train/images"
]
data_config['val'] = [
    f"{working_dir}/merged_dataset/val/images"
]

# If DocLayNet exists, append it
# NOTE: YOLO requires labels to be next to images or in parallel 'labels' folder.
# Public datasets often have JSON labels, not YOLO txt.
# CRITICAL: We cannot just "link" DocLayNet unless it is already in YOLO format.
# Most Kaggle DocLayNet datasets are JSON (COCO).
# We might need a converter step on Kaggle if we use raw DocLayNet.
# 
# PLAN B: Use a pre-processed "YOLO DocLayNet" dataset from Kaggle.
# Search for: "DocLayNet YOLO"
# Assuming user adds a YOLO-formatted DocLayNet.

if doclaynet_root:
    # 1. Check for "images/train" (Standard YOLO format)
    dl_train = os.path.join(doclaynet_root, "images", "train")
    dl_val = os.path.join(doclaynet_root, "images", "val")

    # 2. Check for "train/images" (Alternative)
    if not os.path.exists(dl_train):
        dl_train = os.path.join(doclaynet_root, "train", "images")
        dl_val = os.path.join(doclaynet_root, "valid", "images") # 'valid' or 'val'

    if os.path.exists(dl_train):
        print(f"Adding Public DocLayNet Train: {dl_train}")
        data_config['train'].append(dl_train)

        # Validation path fallback
        if not os.path.exists(dl_val) and "val" in dl_val:
            dl_val = dl_val.replace("val", "valid") # Try 'valid' instead of 'val'

        if os.path.exists(dl_val):
            print(f"Adding Public DocLayNet Val: {dl_val}")
            data_config['val'].append(dl_val)
        else:
            print(f"Warning: Validation path not found at {dl_val}. Using only Custom Val.")
    else:
        print(f"WARNING: Could not determine Train path structure for {doclaynet_root}")
        print("Expected 'images/train' or 'train/images'.")

# FIX: DocLayNet has 11 classes (IDs 0-10). Merged dataset has 9 (0-8).
# We must update 'nc' and 'names' to prevent "Label class 10 exceeds..." errors.
if data_config.get('nc', 0) < 11:
    print(f"Upgrading class count from {data_config.get('nc')} to 11 for DocLayNet compatibility.")
    data_config['nc'] = 11
    
    # Update Names
    names = data_config.get('names', [])
    # Handle dictionary format (common in COCO)
    if isinstance(names, dict):
        # Find max key
        max_id = max(names.keys()) if names else -1
        for i in range(max_id + 1, 11):
            names[i] = f"class_{i}"
    # Handle list format (Standard YOLO)
    elif isinstance(names, list):
        while len(names) < 11:
            names.append(f"class_{len(names)}")
    
    data_config['names'] = names

# Save back
with open(yaml_path, 'w') as f:
    yaml.safe_dump(data_config, f)

print("Updated dataset.yaml:")
print(yaml.safe_dump(data_config))

# Verify file exists
if not os.path.exists(yaml_path):
    print("ERROR: dataset.yaml not found!")
    # Debug listing
    print(os.listdir(working_dir))

# Cell 4: Train YOLOv11 (Curriculum Learning)
# ---------------------------------------------
import csv
import time
from datetime import datetime
from typing import Optional
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

def find_latest_run_dir(project_dir: str, name: str) -> Optional[str]:
    base = os.path.join(project_dir, name)
    if os.path.isdir(base):
        return base
    if not os.path.isdir(project_dir):
        return None
    candidates = [
        os.path.join(project_dir, d)
        for d in os.listdir(project_dir)
        if d.startswith(name)
    ]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

def summarize_results(run_dir: Optional[str], expect_val: bool = True) -> None:
    if not run_dir:
        print("Run directory not found; skipping summary.")
        return
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
    important = [
        "epoch",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    latest_pairs = [
        (k, latest.get(k, ""))
        for k in important
        if latest.get(k, "") != ""
    ]

    print("\nLatest metrics:")
    log_kv(latest_pairs)

    if not expect_val:
        print("Note: validation was disabled; mAP metrics may be missing.")

    def best_metric(col: str):
        best_val = None
        best_epoch = None
        for row in rows:
            raw = row.get(col, "")
            if raw in ("", None):
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            if best_val is None or val > best_val:
                best_val = val
                best_epoch = row.get("epoch", "?")
        return best_epoch, best_val

    for col in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if col in rows[-1]:
            epoch, val = best_metric(col)
            if val is not None:
                print(f"Best {col}: {val:.4f} (epoch {epoch})")

    weights_dir = os.path.join(run_dir, "weights")
    last_path = os.path.join(weights_dir, "last.pt")
    best_path = os.path.join(weights_dir, "best.pt")
    weights = []
    if os.path.exists(last_path):
        size_mb = os.path.getsize(last_path) / (1024 * 1024)
        weights.append(("last.pt", f"{last_path} ({size_mb:.1f} MB)"))
    if os.path.exists(best_path):
        size_mb = os.path.getsize(best_path) / (1024 * 1024)
        weights.append(("best.pt", f"{best_path} ({size_mb:.1f} MB)"))
    if weights:
        print("\nWeights:")
        log_kv(weights)

project_dir = "/kaggle/working/runs/detect"
run_name_a = "train_v5_stage_a"
run_name_b = "train_v5_stage_b"
device_ids = "0,1"  # Use "0" for single GPU sessions

# Resume controls
resume_stage_b = True  # Set to False for a fresh run
resume_ckpt_path = "/kaggle/input/resume-point/runs/detect/train_v5_stage_b/weights/last.pt"
resume_search_roots = ["/kaggle/input"]
resume_optimizer_state = False  # Use True only if optimizer state loads cleanly
total_epochs = 50
run_name_b_resume = f"{run_name_b}_resume"

def find_resume_checkpoint(search_roots, run_name):
    candidates = [
        os.path.join(run_name, "weights", "last.pt"),
        os.path.join("runs", "detect", run_name, "weights", "last.pt"),
        os.path.join("kaggle", "working", "runs", "detect", run_name, "weights", "last.pt"),
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dataset in os.listdir(root):
            base = os.path.join(root, dataset)
            for rel in candidates:
                path = os.path.join(base, rel)
                if os.path.exists(path):
                    return path
    return None

def prepare_resume_run(ckpt_path, project_dir, run_name):
    run_dir = Path(ckpt_path).parent.parent
    target_dir = Path(project_dir) / run_name
    if not target_dir.exists():
        print(f"Copying resume run dir to {target_dir}...")
        shutil.copytree(run_dir, target_dir)
    return str(target_dir / "weights" / "last.pt")

def copy_weights_only(ckpt_path, project_dir, run_name):
    target_dir = Path(project_dir) / run_name / "weights"
    target_dir.mkdir(parents=True, exist_ok=True)
    weights_only_path = target_dir / "last_weights_only.pt"
    shutil.copy2(ckpt_path, weights_only_path)
    return str(weights_only_path)

def sanitize_resume_checkpoint(ckpt_path: str) -> str:
    """Ensure optimizer state has 'step' to avoid resume crashes."""
    try:
        import torch
    except Exception as e:
        print(f"Warning: torch not available for checkpoint sanitization: {e}")
        return ckpt_path

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        print(f"Warning: failed to load checkpoint for sanitization: {e}")
        return ckpt_path

    optimizer = ckpt.get("optimizer")
    if not isinstance(optimizer, dict) or "state" not in optimizer:
        return ckpt_path

    changed = False
    for state in optimizer.get("state", {}).values():
        if isinstance(state, dict) and state and "step" not in state:
            state["step"] = 0
            changed = True

    if not changed:
        return ckpt_path

    sanitized_path = ckpt_path.replace(".pt", "_sanitized.pt")
    try:
        torch.save(ckpt, sanitized_path)
        print(f"Saved sanitized checkpoint: {sanitized_path}")
        return sanitized_path
    except Exception as e:
        print(f"Warning: failed to save sanitized checkpoint: {e}")
        return ckpt_path

resume_ckpt = None
resume_run_dir = None
if resume_stage_b:
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        resume_ckpt = resume_ckpt_path
    else:
        resume_ckpt = find_resume_checkpoint(resume_search_roots, run_name_b)
    if resume_ckpt:
        print(f"Found resume checkpoint: {resume_ckpt}")
        resume_ckpt = prepare_resume_run(resume_ckpt, project_dir, run_name_b)
        resume_run_dir = str(Path(resume_ckpt).parent.parent)
        if resume_optimizer_state:
            resume_ckpt = sanitize_resume_checkpoint(resume_ckpt)
        else:
            resume_ckpt = copy_weights_only(resume_ckpt, project_dir, run_name_b_resume)
    else:
        print("Resume checkpoint not found; falling back to Stage A + fresh Stage B.")
        resume_stage_b = False

# STAGE A: Coarse Training (Full Data, Low Res)
# ---------------------------------------------
weights_a = None
if not resume_stage_b:
    log_section("STAGE A: Coarse Training (640px)")
    log_kv([
        ("start_time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("goal", "fast convergence on general layout structure"),
        ("device", device_ids),
    ])

    model_a = YOLO('yolo11s.pt') 
    stage_a_start = time.perf_counter()
    model_a.train(
        data=yaml_path,
        epochs=8,       # Fast epochs for 2xT4 time budget
        imgsz=640,      # Low res for speed
        batch=24,       # 2xT4 -> 12 per GPU; drop to 16 if OOM
        project=project_dir,
        name=run_name_a,
        device=device_ids,
        save=True,
        val=False       # Skip validation to save time
    )
    stage_a_elapsed = time.perf_counter() - stage_a_start
    print(f"Stage A runtime: {human_duration(stage_a_elapsed)}")
    run_dir_a = find_latest_run_dir(project_dir, run_name_a)
    summarize_results(run_dir_a, expect_val=False)
    weights_a = f"{project_dir}/{run_name_a}/weights/last.pt"

# STAGE B: Refinement (High Res)
# ------------------------------
if resume_stage_b and resume_ckpt:
    log_section("STAGE B: Resume Training (1024px)")
    log_kv([
        ("start_time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("goal", "resume from previous checkpoint"),
        ("checkpoint", resume_ckpt),
        ("optimizer_state", "resume" if resume_optimizer_state else "weights-only"),
        ("device", device_ids),
    ])
    model_b = YOLO(resume_ckpt)
    stage_b_start = time.perf_counter()
    if resume_optimizer_state:
        model_b.train(
            resume=True,
            device=device_ids,
        )
    else:
        remaining_epochs = total_epochs
        if resume_run_dir:
            results_csv = os.path.join(resume_run_dir, "results.csv")
            if os.path.exists(results_csv):
                try:
                    with open(results_csv, newline="") as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        last_epoch = int(rows[-1].get("epoch", 0))
                        remaining_epochs = max(1, total_epochs - last_epoch)
                except Exception as e:
                    print(f"Warning: failed to read last epoch from {results_csv}: {e}")
        model_b.train(
            data=yaml_path,
            epochs=remaining_epochs,
            imgsz=1024,
            batch=8,
            project=project_dir,
            name=run_name_b_resume,
            device=device_ids,
            resume=False,
            exist_ok=False,
        )
else:
    log_section("STAGE B: Refinement (1024px)")
    log_kv([
        ("start_time", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")),
        ("goal", "high fidelity using pretrained weights from Stage A"),
        ("device", device_ids),
    ])

    # Resume from Stage A weights
    if not weights_a:
        weights_a = 'yolo11s.pt'
    model_b = YOLO(weights_a) 

    stage_b_start = time.perf_counter()
    model_b.train(
        data=yaml_path,  # Use full dataset for simplicity/robustness
        epochs=total_epochs,  # Use full budget; ensure checkpoints are saved for resume if needed
        imgsz=1024,      # High Res
        batch=8,         # Lower batch for 1024px to avoid OOM
        project=project_dir,
        name=run_name_b,
        device=device_ids,
    )
stage_b_elapsed = time.perf_counter() - stage_b_start
print(f"Stage B runtime: {human_duration(stage_b_elapsed)}")
run_dir_b = find_latest_run_dir(project_dir, run_name_b)
summarize_results(run_dir_b, expect_val=True)


# Cell 5: Zip Outputs (Exclude Data)
# ----------------------------------
print("Zipping outputs (excluding /kaggle/working/data)...")
# -r: recursive
# -1: fastest compression (saves CPU time)
# -q: quiet (reduces I/O overhead from printing thousands of lines)
os.system(
    'zip -r -1 -q results_v5_no_data.zip /kaggle/working '
    '-x "/kaggle/working/data/*"'
)
print("Download 'results_v5_no_data.zip' from Output.")
