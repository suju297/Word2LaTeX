# Model Optimization Research
**Date:** 2026-01-11
**Topic:** Improving Document Layout Analysis Accuracy

## 1. Model Architecture: YOLOv8 vs YOLOv11
- **Findings**: YOLOv11 generally outperforms YOLOv8 in accuracy (mAP) with fewer parameters.
- **DocLayNet Specifics**: YOLOv11n (Nano) showed 1.7% mAP increase over YOLOv8n on DocLayNet.
- **Recommendation**: Switch to **YOLOv11** for the next training run ("Model V4"). It offers better feature extraction for cluttered document scenes.

## 2. Image Resolution
- **Standard**: 640x640 is standard training size.
- **Optimized**: Recent papers use 1024x1024 or 1280x1280 for document analysis to capture small text/lines.
- **Trade-off**: Higher resolution linearly increases inference time.
- **Recommendation**: Train at **1024x1024**. This is the "sweet spot" for modern GPUs (T4/P100) on Kaggle.

## 3. SAHI (Slicing Aided Hyper Inference)
- **Concept**: Slice image into overlapping patches -> Infer -> Stitch results.
- **Benefit**: Massive gain for small objects (page numbers, footnotes, tiny floating figures).
- **Cost**: 4x-9x slower inference (runs model on each patch).
- **Decision**: Keep as an **inference-time option** (post-training). We don't need to train differently to use SAHI, but we need to integrate `sahi` library if we want to use it. For now, high-res input (1024) is a faster/simpler first step.

## 4. Data Augmentation (Albumentations)
- **Status**: YOLO already has Mosaic/MixUp.
- **Enhancement**: Document-specific augmentations are critical:
  - RandomRotate90 (rare for docs, but slight rotation +/- 5 deg is good)
  - GaussianBlur (simulate bad scans)
  - RandomBrightnessContrast
- **Action**: Add `albumentations` pipeline to the dataset loader if accuracy plateaus. For V4, standard YOLOv11 augmentation is likely sufficient as it's very robust.

## 5. Refined Layout Strategy (Toggles & Unification)
- **Problem**: `layout_oracle.py` is currently hard-wired to `layout_x101`, ignoring our new YOLO models (`layout_ml`).
- **Goal**: Create a unified `detect_layout` interface controlled by configuration.

### Implementation Plan
1.  **Configuration**:
    - Add `LAYOUT_MODEL` environment variable (default: `yolo`).
    - Options: `x101` (High Precision), `yolo` (High Speed, Model V4), `heuristic` (Fallback).

2.  **Unification in `layout_oracle.py`**:
    - Import both `layout_x101` and `layout_ml`.
    - Dispatch based on `os.getenv("LAYOUT_MODEL")`.
    - **Label Normalization**: Ensure X101 (PubLayNet) and YOLO (DocLayNet) use consistent labels (e.g., "Table", "Figure").

3.  **Verification**:
    - **Metrics**: Log inference time and region count.
    - **Test**: Run `scripts/vision_oracle.py` with `LAYOUT_MODEL=x101` and `LAYOUT_MODEL=yolo` to compare output.

## 6. Model V4 Upgrade (High Accuracy)
- **Configuration Changes**:
  - Model: switch to `yolo11s.pt` (Small) for better accuracy/efficiency than v8.
  - Resolution: set `imgsz=1024` to improve small text/line detection.
  - Batch Size: reduce to 8-10 if GPU memory is limited; keep 16 if V100/P100 available.
- **Execution Plan**:
  - Update `kaggle_export/kaggle_train_script.py` to use `yolo11s.pt`.
  - Run training on Kaggle and export `model_v4.pt`.
  - Update `layout_ml.py` to load the new model.

## 7. Model V5 (Data Scale-Up & Curriculum)
- **Problem**: 80k images @ 1024px is too slow for Kaggle (50h+).
- **Strategy: Progressive Resizing (Budget-Aware)**:
  - **Stage A (Coarse)**: Train on Full Dataset @ `640px` (Fast). 8 Epochs. Batch 32. Est: 2h.
  - **Stage B (Refine)**: Train on High-Value Subset @ `1024px` (Precise). Use "All Custom" + "20% DocLayNet". 8 Epochs. Batch 4-8. Est: 6h.
- **Benefits**:
  - fast convergence on structure (A).
  - high fidelity on small details (B).
  - fits within 9-12h timeout.
- **Execution Plan**:
  - Update `kaggle_train_script.py` to implement the 2-stage logic and subset generation.
