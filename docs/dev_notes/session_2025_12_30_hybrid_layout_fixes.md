# Session Context: Hybrid Layout Engine Fixes
**Date**: 2025-12-30
**Topic**: Phase 4 - Hybrid Layout Engine Implementation & Refinement

## Context
We implemented a "Hybrid Layout Engine" to achieve high-fidelity "Word-like" output, focusing on three critical areas that were causing layout drift and low SSIM scores: Tab Stops, Absolute Textboxes, and Fixed-Width Tables.

## Accomplishments (The "3 Concrete Fixes")

### 1. Robust Tab Stops (Fix #1)
*   **Problem**: `\tabto{<pos>in}` was unreliable because Word tabs are relative to the paragraph origin (indent + margin), whereas `tabto` is absolute or flow-relative. Font metric differences caused right-aligned dates to shift.
*   **Solution**: "Overlay Method".
    *   For lines with **Right** or **Center** aligned tabs (e.g., Headers), we now use `\noindent` + `\makebox[0pt][alignment]{text}` positioned with calculated `\hspace`.
    *   This guarantees the text ends exactly at the tab stop location provided by Word, regardless of the content length or font.
    *   **Fallback**: Retained `\tabto` only for simple Left tabs or wrapping text bodies.
*   **Files**: `src/wordtolatex/generator.py` (`_format_paragraph`).

### 2. Absolute Textboxes (Fix #2)
*   **Problem**: `posOffset` in OOXML is in EMUs and is relative to an anchor (`relativeFrom="margin"` vs `"page"`). We were treating it as absolute page inches, causing massive positioning errors.
*   **Solution**:
    *   **Parser**: Extract `relativeFrom` attribute and `posOffset`. Convert EMUs to Inches (`value / 914400.0`).
    *   **IR**: Updated `TextboxBlock` to store `origin_x`/`origin_y` ("page", "margin", etc.).
    *   **Generator**: Introduced `_CURRENT_MARGINS` (populated from Section 1). If `origin="margin"`, add the margin offset to the `textpos` coordinates.
*   **Files**: `src/wordtolatex/ir.py`, `src/wordtolatex/docx_parser/document.py` (`_parse_drawing`), `src/wordtolatex/generator.py` (`_format_textbox`).

### 3. Fixed-Width Tables (Fix #3)
*   **Problem**: Using `tabularx` (auto-width columns) when `tblGrid` (explicit widths) existed caused columns to drift or resize unpredictably compared to Word.
*   **Solution**:
    *   **Parser**: Extract `w:tblGrid` -> `w:gridCol` widths (convert twips to inches). Store in `TableBlock.column_widths`.
    *   **Generator**: Check `column_widths`. If present, switch to `\begin{tabular}{|p{width}|...}`.
    *   **Spans**: For `\multicolumn`, calculate the sum of widths of the spanned columns (`sum(column_widths[start:end])`) to ensure alignment.
*   **Files**: `src/wordtolatex/ir.py`, `src/wordtolatex/docx_parser/document.py` (`_parse_table`), `src/wordtolatex/generator.py` (`_format_table`).

## Verification
We created a suite of targeted unit tests to verify these fixes:
*   `tests/test_fixes_e2e.py`: End-to-end verification of Tab Overlay logic and Textbox Margin Offsets.
*   `tests/test_tables_fixed.py`: Verification of Fixed-Width Table rendering and span width calculation.
*   `tests/test_textboxes.py` & `tests/test_parser_drawing.py`: Foundation tests for Textbox IR and Generator.

All tests are **PASSING**.

## Next Steps
1.  **Bundle Fonts**: Ensure Carlito and TeX Gyre Termes are used to match Calibri/Times metrics (Fix #1 from user list).
2.  **SSIM Verification**: Run `scripts/region_diff.py` to confirm Title SSIM > 0.6.
3.  **Refine Paragraph Spacing**: Address remaining "line height" drift using `w:spacing` extraction.
