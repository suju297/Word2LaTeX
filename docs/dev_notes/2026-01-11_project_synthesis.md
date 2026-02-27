# Project Synthesis: Layout Fidelity & Model Optimization
**Date:** 2026-01-11
**Status:** Evidence-Based Review

## 1. Layout Fidelity: The "Rigid Container" Problem

### Observation
During the verification of `Microsoft_Resume_2`, we observed a persistent page overflow (6-8 pages generated vs 5 pages reference), despite the "Auto-Layout Optimizer" reporting aggressive line spacing compression (`squeeze_factor` down to 0.34).

### Hypothesis & Investigation
We hypothesized that the `squeeze_factor` was not effectively reducing the vertical footprint of the document. Inspection of the generated LaTeX revealed extensive use of `tabularx` environments for layout. Crucially, the row heights were defined using fixed LaTeX rules derived from Word (e.g., `\rule{0pt}{15.8pt}`).

### Evidence
*   **Before Fix**: `squeeze_factor` only applied to `\setstretch` (text leading). The text inside the cells shrank, but the cell boundaries (defined by the fixed `\rule`) remained static. The page count stalled at 6.
*   **Intervention**: We modified `generator.py` to propagate the `squeeze_factor` to structural dimensions:
    ```python
    row_height_pt = (row.height_twips / 20.0) * squeeze_factor
    pad_top = (cell.padding_twips['top'] / 20.0) * squeeze_factor
    ```
*   **Outcome**: The document successfully compressed to exactly 5 pages. This confirms that for high-fidelity Word-to-LaTeX conversion, **structural scaling** is as critical as text scaling for "squeeze-to-fit" functionality.

## 2. Model Performance Analysis

### Strategy
We employ a hybrid approach:
1.  **Rule-Based/Heuristic**: For standard text/headings (fast, determinstic).
2.  **ML-Based (Layout Oracle)**: For complex regions (Tables, Figures, Sidebars).

### Model V3 (YOLOv8 Nano)
*   **Dataset**: Combined DocLayNet + Resume + DocBank (k=2900 samples).
*   **Resolution**: 640x640.
*   **Performance**:
    *   **mAP50**: 0.77 (Good localization).
    *   **mAP50-95**: 0.60 (High precision).
*   **Limitations**: The "Nano" architecture and low resolution (640px) limit performance on dense text documents or small objects (page numbers, footnotes).

### Fallback: LayoutParser (X101)
*   **Architecture**: Mask R-CNN X101-FPN (ResNet-101 backbone).
*   **Pros**: State-of-the-art accuracy on PubLayNet accuracy; excellent segmentation masks.
*   **Cons**: extremely high inference latency (~2-5s CPU); heavy dependencies (`detectron2`).

## 3. Optimization Research (Model V4)

To bridge the gap between the speed of YOLO and the accuracy of X101, we conducted an evidence-based review of current state-of-the-art techniques.

### Findings
1.  **Architecture**: **YOLOv11** consistently outperforms YOLOv8 in document analysis benchmarks, offering better feature extraction for cluttered pages with fewer parameters [Ultralytics, 2024].
2.  **Resolution**: 640px is a bottleneck. Recent document analysis papers standardly use **1024px** or **1280px** to resolve fine-grained details like separators and small font text.
3.  **Inference**: **SAHI** (Slicing Aided Hyper Inference) improves small object detection by slicing images during inference. However, it increases latency by 4-9x.
4.  **Augmentation**: **Albumentations** (rotation, noise, blur) adds robustness against scan artifacts.

### Decision for V4
We will train **Model V4** using:
*   **Model**: `yolo11s.pt` (Small) — Step up from Nano for capacity.
*   **Resolution**: `1024x1024` — To resolve dense document structures.
*   **Augmentation**: Standard YOLO mosaic/mixup is sufficient for now; Albumentations reserved for V5 if needed.

This configuration targets the "sweet spot" of sub-second inference with accuracy approaching X101.
