# Dev Note: Vision-Guided Layout Detection
**Date**: 2026-01-10
**Focus**: Improving Layout Fidelity via PDF Analysis

## Context
We faced a persistent issue with complex layouts (specifically resumes like `Resume_1`) where standard block rendering caused content overflow (2 pages instead of 1) and columns were misaligned. OOXML `LayoutGrid` structures often provided insufficient or misleading width information for precise LaTeX `minipage` rendering.

## Solution: Vision-Guided Analysis
We shifted from a purely structure-based approach to a hybrid approach:
1. **Source of Truth**: Reference PDF (Visual Layout).
2. **Implementation**: `layout_oracle.detect_layout_pattern` uses `PyMuPDF` to analyze text block distribution.

### Algorithm Details
- **Coverage Histogram**: We use a 100-bin horizontal histogram (0-100% page width).
- **Gap Detection**: We identify "Zero Coverage" runs in the histogram.
  - **Wide Gap (>10%)**: Indicates a Sidebar or Split layout.
  - **Narrow Gap (<5%)**: Indicates multi-column text (or justified text noise).
- **Metrics**: We extract exact `sidebar_width_ratio`, `main_width_ratio`, and `gap_ratio` from the PDF.

## Key Findings

### 1. Robustness is Critical
- **Inverted Detection**: In `Resume_1`, the algorithm detected a "Right Sidebar" (`SIDEBAR_RIGHT`) with widths 60% (Left) / 30% (Right).
- **Reality**: The visual layout is Left Sidebar (35%) / Right Main (60%).
- **Heuristic Fix**: We implemented a robustness heuristic: **"Resumes typically have a Narrow Left Column"**.
  - Instead of trusting the detected pattern's orientation, we calculate:
    ```python
    sidebar_width = min(ratio1, ratio2)
    main_width = max(ratio1, ratio2)
    ```
  - This solved the inversion issue and produced a perfect 1-page render (`SSIM 0.55`).

### 2. Standard Fallbacks
- **IEEE Template**: Detected as `SINGLE_COLUMN` (Gap approx 0% due to tight text).
- This is a safe failure mode. It avoids forcing a sidebar layout on an academic paper.
- **Future Work**: Implement specific `TWO_COLUMN_EQUAL` logic using `multicol` package when the gap is distinct but balanced (45%-55% center).

### 3. Corpus Performance
- **Resume_1**: **Success**. 1 Page. Correct Columns.
- **Microsoft Resumes**: **Partial**. Content is high fidelity (`SSIM ~0.84`) but Vertical Spacing is too loose (11+ pages).
- **Academic**: **Needs Work**. `SSIM ~0.50` on IEEE. 

## Next Steps
1. **Vertical Compaction**: The standard block renderer needs a "Squeeze" factor or tighter defaults to match Word's vertical density.
2. **Multicol**: Implement `multicol` for academic papers.
3. **Header Alignment**: Improve `LayoutGrid` cell mapping to prevent header labels/values from incorrectly splitting across the sidebar/main boundary.
