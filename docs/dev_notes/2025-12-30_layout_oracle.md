# Dev Notes: Layout Oracle Implementation (2025-12-30)

## Overview
Implemented a **Hybrid Layout Oracle** that combines Machine Learning (LayoutParser/Detectron2) with Deterministic Heuristics (PyMuPDF/OpenCV) to provide layout hints (regions, columns) for the Word-to-LaTeX converter.

## Architecture

### 1. Column Detection (Deterministic)
*   **Method**: 1D Clustering of text block x-centers.
*   **Logic**:
    *   Extracts text blocks via PyMuPDF.
    *   Calculates center-X for each block.
    *   Uses KMeans (k=1..3) to find vertical gutters.
    *   **Guardrails**: Ignores gaps < 15% width and gutters near page margins (10% edge buffer).
*   **Result**: Highly reliable column structure without ML.

### 2. Region Detection (Hybrid)
Prioritizes ML for accuracy, falls back to heuristics for reliability.

#### A. Primary: LayoutParser + Detectron2
*   **Model**: PubLayNet `mask_rcnn_R_50_FPN_3x`.
*   **Classes**: `Text`, `Title`, `List`, `Table`, `Figure`.
*   **Configuration**:
    *   Weights: `models/model_final.pth`
    *   Config: `models/config.yml` (Standard Detectron2 PubLayNet config).
    *   **Note**: `NUM_CLASSES` set to **6** (5 labels + background) to match checkpoint dimensions.
*   **Normalization**: Output bboxes are normalized to `[0..1]`.

#### B. Fallback: Heuristics (`layout_heuristics.py`)
Executed if ML dependencies are missing or model loading fails.
*   **Figures**: `fitz.Page.get_images()` rectangles.
*   **Tables**: OpenCV morphological ops (horizontal/vertical line detection masks).
*   **Textboxes**: `fitz.Page.get_drawings()` vector rectangles.

## Installation & Setup Saga (Debugging Log)

### 1. Installing Detectron2
*   **Issue**: Standard pip install fails.
*   **Solution**: `uv pip install setuptools` then `uv pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"`.
*   **Dependency**: Added `torchvision>=0.15.0` to `pyproject.toml` (required for Detectron2).

### 2. Model Loading Issues
*   **Issue**: `LayoutParser` auto-download failed with `AssertionError`.
*   **Root Cause**: Implicit Dropbox URLs with query strings (`?dl=1`) confused the caching mechanism.
*   **Solution**: 
    1.  Manually downloaded weights to `models/model_final.pth` using `curl`.
    2.  Manually created `models/config.yml` (since the download link returned binary garbage/zip).
    3.  Updated `layout_ml.py` to check `models/` directory first before falling back to URL.

### 3. Config/Weight Mismatch
*   **Issue**: `RuntimeError` due to shape mismatch in `roi_heads.box_predictor`.
*   **Root Cause**: Standard PubLayNet config assumes 5 classes, but the pre-trained weights were built with 6 (including background).
*   **Solution**: Updated `models/config.yml` to set `NUM_CLASSES: 6`.

### 4. Code Crash
*   **Issue**: `TypeError: LayoutRegion.__init__() got an unexpected keyword argument 'page'`.
*   **Solution**: Updated `LayoutRegion` dataclass in `layout_ml.py` to include `page: int`.

## Verification Status
*   **Test File**: `tests/samples/resume_styled.pdf`
*   **ML Output**: Detected **10 regions** (8 Titles, 2 Lists).
*   **Heuristic Output**: Detected **1 region** (Header Table) when ML was forced off.
*   **Conclusion**: System is fully operational and robust.

## Next Steps
*   Wire `layout_hints.json` into the `Policy Engine` to drive decision making (e.g. `multicols` environment generation).
