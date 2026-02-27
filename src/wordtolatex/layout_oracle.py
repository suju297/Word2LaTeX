"""Layout Oracle - Extract layout information from rendered PDFs.

This module implements the "Layout Oracle" approach:
1. Extract deterministic text positions from PDF (PyMuPDF).
2. Detect structure (columns, reading order) via heuristics.
3. Detect layout patterns (sidebar, multi-column) for specialized rendering.
4. Integrate ML-detected regions (Tables, Figures) if available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import math

import fitz  # PyMuPDF

try:
    import numpy as np
except ImportError:
    np = None


class LayoutPattern(Enum):
    """Detected layout patterns for specialized rendering."""
    SINGLE_COLUMN = "single_column"      # Standard single-column document
    SIDEBAR_LEFT = "sidebar_left"        # Narrow left sidebar + wide main (resumes)
    SIDEBAR_RIGHT = "sidebar_right"      # Wide main + narrow right sidebar
    TWO_COLUMN_EQUAL = "two_col_equal"   # Academic papers - equal columns
    GRID_COMPLEX = "grid_complex"        # Complex table-based layout


@dataclass
class LayoutPatternInfo:
    """Details about detected layout pattern."""
    pattern: LayoutPattern
    confidence: float = 1.0
    sidebar_width_ratio: float = 0.0    # e.g., 0.28 for 28% sidebar
    main_width_ratio: float = 1.0       # e.g., 0.70 for 70% main
    column_gap_ratio: float = 0.0       # Gap between columns as ratio
    column_boundaries: List[float] = field(default_factory=list)  # Normalized x positions


@dataclass
class LayoutRegion:
    """A detected text region with semantic label."""
    bbox: Tuple[float, float, float, float]  # Normalized [0..1]: x0, y0, x1, y1
    label: str  # table, figure, text, header, footer, list
    confidence: float = 1.0
    text: Optional[str] = None
    page: int = 0
    source: str = "heuristic"  # "ml" or "heuristic"


@dataclass
class TextBlock:
    """A block of text with its position."""
    text: str
    x0: float  # left (points)
    y0: float  # top (points)
    x1: float  # right (points)
    y1: float  # bottom (points)
    page: int
    font_size: float = 0
    font: str = ""
    flags: int = 0  # bold, italic, etc.
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
        
    @property
    def is_bold(self) -> bool:
        return bool(self.flags & 2**4)
    
    @property
    def is_right_aligned(self) -> bool:
        return self.x1 > 500  # Rough heuristic for A4/Letter


@dataclass
class ColumnStructure:
    """Detected column structure for a page."""
    page_index: int
    count: int
    boundaries: List[float]  # X-coordinates of column boundaries
    
    def get_column_index(self, x_center: float) -> int:
        """Get column index for a given x position."""
        if self.count <= 1:
            return 0
        for i, boundary in enumerate(self.boundaries):
            if x_center < boundary:
                return i
        return self.count - 1


@dataclass
class LayoutAnalysis:
    """Analysis of a PDF's layout."""
    page_count: int = 0
    page_dims: List[Tuple[float, float]] = field(default_factory=list) # w, h per page
    text_blocks: List[TextBlock] = field(default_factory=list)
    regions: List[LayoutRegion] = field(default_factory=list)
    column_structure: List[ColumnStructure] = field(default_factory=list)
    
    def get_right_aligned_texts(self) -> List[TextBlock]:
        """Get texts that appear right-aligned."""
        return [b for b in self.text_blocks if b.is_right_aligned]
        
    def get_section_headers(self) -> List[TextBlock]:
        return [b for b in self.text_blocks if b.is_bold and b.font_size > 10]
    
    def get_regions_by_type(self, label: str) -> List[LayoutRegion]:
        return [r for r in self.regions if r.label == label]


def _detect_columns(blocks: List[TextBlock], page_width: float) -> ColumnStructure:
    """
    Detect columns using 1D clustering of text block x-centers.
    Returns detected boundaries.
    """
    if not blocks or np is None:
        return ColumnStructure(page_index=blocks[0].page if blocks else 0, count=1, boundaries=[])

    # 1. Get x-centers
    centers = np.array([b.center_x for b in blocks])
    
    # 2. Histogram analysis (finding gaps)
    # Binning: 10pt bins across page
    bins = np.arange(0, page_width, 10)
    hist, bin_edges = np.histogram(centers, bins=bins)
    
    # Find gaps (bins with 0 or very low count)
    # We look for "wide" gaps. A real gutter is usually > 20pt.
    # Simple heuristic: Look for empty regions in the central 80% of page.
    # (Side margins are also empty but not splitters).
    
    # Check for 2-column signature: Peak - Gap - Peak
    # We use a simple K-Means-like split check
    
    from sklearn.cluster import KMeans
    
    best_k = 1
    best_boundaries = []
    
    # Try k=2 (most common split)
    if len(centers) > 10:
        km = KMeans(n_clusters=2, n_init=3, random_state=42)
        km.fit(centers.reshape(-1, 1))
        
        # Check separation
        c1 = km.cluster_centers_[0][0]
        c2 = km.cluster_centers_[1][0]
        separation = abs(c1 - c2)
        gap_center = (c1 + c2) / 2
        
        # Guard 1: Separation must be significant (> 15% width)
        if separation > page_width * 0.15:
            # Guard 2: Gap must not be near edges (Margins are not gutters)
            if page_width * 0.15 < gap_center < page_width * 0.85:
                # Guard 3: Verify emptiness (Vertical Support)
                # Count blocks that bridge the gap
                overlap_count = sum(1 for b in blocks if b.x0 < gap_center < b.x1)
                
                # Allow very minimal noise (e.g. page numbers centered)
                if overlap_count < len(blocks) * 0.05:
                    best_k = 2
                    # Normalize boundary to [0..1]
                    best_boundaries = [float(gap_center / page_width)]
                
    return ColumnStructure(
        page_index=blocks[0].page,
        count=best_k,
        boundaries=best_boundaries
    )


def _sort_reading_order(blocks: List[TextBlock], columns: ColumnStructure) -> List[TextBlock]:
    """Sort blocks by reading order: Column First -> Top-Down."""
    if columns.count <= 1:
        # Simple top-down (y0), then left-right (x0)
        return sorted(blocks, key=lambda b: (round(b.y0, 1), b.x0))
    
    # Assign blocks to columns
    clustered = [[] for _ in range(columns.count)]
    for b in blocks:
        col_idx = columns.get_column_index(b.center_x)
        clustered[col_idx].append(b)
        
    # Sort each column top-down
    sorted_blocks = []
    for col_blocks in clustered:
        col_blocks.sort(key=lambda b: (round(b.y0, 1), b.x0))
        sorted_blocks.extend(col_blocks)
        
    return sorted_blocks


def detect_layout_pattern(pdf_path: Path, page_num: int = 0) -> LayoutPatternInfo:
    """
    Analyze PDF to detect column structure and layout pattern using coverage analysis.
    
    Algorithm:
    1. Extract text block extents (x0, x1).
    2. Build a high-resolution coverage mask (100 bins) of the page width.
    3. Find significant gaps (consecutive empty bins) in the central region.
    4. Classify patterns based on gap position, width, and content distribution.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page to analyze (default 0)
        
    Returns:
        LayoutPatternInfo with detected pattern and dimensions.
        Ratios are typically relative to the text width (content_max - content_min),
        but normalized to [0,1] relative to the detected content block.
    """
    if np is None:
        return LayoutPatternInfo(pattern=LayoutPattern.SINGLE_COLUMN)
    
    doc = fitz.open(pdf_path)
    if page_num >= len(doc):
        doc.close()
        return LayoutPatternInfo(pattern=LayoutPattern.SINGLE_COLUMN)
    
    page = doc[page_num]
    rect = get_canonical_rect(page)
    page_width = rect.width
    
    # Extract text blocks with positions
    blocks = []
    raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    
    for block in raw_blocks:
        if block["type"] != 0:  # Text blocks only
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                bbox = span.get("bbox", (0, 0, 0, 0))
                # Ignore very small snippets (e.g. noise) unless solitary
                if len(text) > 1 or (bbox[2] - bbox[0] > 5):
                    blocks.append({
                        "x0": bbox[0],
                        "x1": bbox[2],
                        "y0": bbox[1],
                        "y1": bbox[3],
                        "text": text
                    })
    
    doc.close()
    
    if len(blocks) < 5:
        return LayoutPatternInfo(pattern=LayoutPattern.SINGLE_COLUMN)
    
    # Determined content boundaries (margins)
    x0_values = np.array([b["x0"] for b in blocks])
    x1_values = np.array([b["x1"] for b in blocks])
    
    content_left = np.percentile(x0_values, 1)   # Ignore 1% outliers
    content_right = np.percentile(x1_values, 99) # Ignore 1% outliers
    content_width = content_right - content_left
    
    if content_width <= 0:
        return LayoutPatternInfo(pattern=LayoutPattern.SINGLE_COLUMN)

    # Create coverage profile (200 bins for high res to detect approx 0.5% gaps)
    n_bins = 200
    bin_width = page_width / n_bins
    coverage = np.zeros(n_bins)
    
    for b in blocks:
        # Map block extent to bins
        start_bin = int(max(0, min(n_bins-1, np.floor(b["x0"] / bin_width))))
        end_bin = int(max(0, min(n_bins-1, np.ceil(b["x1"] / bin_width))))
        
        # Mark bins as covered (overlap accumulation)
        for i in range(start_bin, end_bin):
            coverage[i] += 1
            
    # Find gaps in the central region (10% to 90% of page)
    # This avoids detecting margins as columns
    start_search = int(n_bins * 0.10)
    end_search = int(n_bins * 0.90)
    
    # Robust Gap Threshold: 25% of peak text density to filter out noise (captions, lines)
    peak_coverage = np.max(coverage)
    gap_threshold = max(0, int(0.25 * peak_coverage))
    
    gaps = []
    in_gap = False
    gap_start = 0
    
    for i in range(start_search, end_search):
        if coverage[i] <= gap_threshold: # Changed from == 0 to <= threshold
            if not in_gap:
                in_gap = True
                gap_start = i
        else:
            if in_gap:
                gap_end = i
                # Found a completed gap
                gap_center_idx = (gap_start + gap_end) / 2
                
                # Convert back to normalized coordinates relative to CONTENT
                # (since LaTeX minipages partition the \linewidth)
                
                # Page coords
                x_gap_start = gap_start * bin_width
                x_gap_end = gap_end * bin_width
                
                # Clip gap to content bounds for ratio calc
                c_gap_start = max(content_left, x_gap_start)
                c_gap_end = min(content_right, x_gap_end)
                
                if c_gap_end > c_gap_start:
                    gap_width_pt = c_gap_end - c_gap_start
                    gap_ratio = gap_width_pt / content_width
                    
                    # Sidebar ratio = fraction of content to the left of the gap
                    left_width_pt = c_gap_start - content_left
                    sidebar_ratio = left_width_pt / content_width
                    
                    gap_center_norm = (c_gap_start + c_gap_end) / 2 / page_width
                    
                    if gap_ratio > 0.007: # Ignore tiny gaps (<0.7%)
                        gaps.append({
                            "gap_ratio": gap_ratio,
                            "sidebar_ratio": sidebar_ratio, # Ratio of left column
                            "center_norm": gap_center_norm, # For left/right classification
                            "width_pt": gap_width_pt
                        })
                
                in_gap = False
                
    if not gaps:
        return LayoutPatternInfo(
            pattern=LayoutPattern.SINGLE_COLUMN,
            confidence=0.9,
            main_width_ratio=1.0
        )
        
    # Pick the most significant gap (widest)
    best_gap = max(gaps, key=lambda g: g["gap_ratio"])
    
    gap_ratio = best_gap["gap_ratio"]
    sidebar_ratio = best_gap["sidebar_ratio"]
    center = best_gap["center_norm"]
    
    # Classification Logic
    
    # 1. Wide gap (> 10%) usually implies Sidebar or Grid layout
    # 2. Narrow gap (< 6%) with central position usually implies Two-Column Text
    
    if gap_ratio > 0.10:
        # Layout Column Pattern
        if center < 0.45:
            # Left Sidebar
            return LayoutPatternInfo(
                pattern=LayoutPattern.SIDEBAR_LEFT,
                confidence=0.9,
                sidebar_width_ratio=sidebar_ratio,
                column_gap_ratio=gap_ratio,
                main_width_ratio=1.0 - sidebar_ratio - gap_ratio
            )
        elif center > 0.55:
            # Right Sidebar
             return LayoutPatternInfo(
                pattern=LayoutPattern.SIDEBAR_RIGHT,
                confidence=0.9,
                sidebar_width_ratio=1.0 - sidebar_ratio - gap_ratio, # sidebar is on right
                column_gap_ratio=gap_ratio,
                main_width_ratio=sidebar_ratio # main is the 'left' part
            )
        else:
            # Central wide gap - likely Grid or Split
             return LayoutPatternInfo(
                pattern=LayoutPattern.SIDEBAR_LEFT, # Default to left split
                confidence=0.7,
                sidebar_width_ratio=sidebar_ratio,
                column_gap_ratio=gap_ratio,
                main_width_ratio=1.0 - sidebar_ratio - gap_ratio
            )
            
    else:
        # Narrow gap - potentially Multi-column Text
        if 0.40 < center < 0.60:
             return LayoutPatternInfo(
                pattern=LayoutPattern.TWO_COLUMN_EQUAL,
                confidence=0.85,
                sidebar_width_ratio=0.48, # Approx half
                column_gap_ratio=gap_ratio,
                main_width_ratio=0.48
            )
    
    # Default fallback
    return LayoutPatternInfo(
        pattern=LayoutPattern.SINGLE_COLUMN,
        confidence=0.6,
        main_width_ratio=1.0
    )


def get_canonical_rect(page: fitz.Page) -> fitz.Rect:
    """Return the canonical page rect (cropbox if valid, else mediabox)."""
    cb = page.cropbox
    mb = page.mediabox
    # Use cropbox if it's not identical to mediabox and represents a proper box
    if cb and cb != mb and cb.width > 0 and cb.height > 0:
        return cb
    return mb

def extract_layout(pdf_path: Path) -> LayoutAnalysis:
    """Extract layout information (Text, Columns, Regions) from a PDF."""
    analysis = LayoutAnalysis()
    
    doc = fitz.open(pdf_path)
    analysis.page_count = len(doc)
    
    all_blocks = []
    
    for page_num, page in enumerate(doc):
        canonical = get_canonical_rect(page)
        w, h = canonical.width, canonical.height
        analysis.page_dims.append((w, h))
        
        # 1. Text Extraction
        raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        page_blocks = []
        
        for block in raw_blocks:
            if block["type"] != 0: continue
            
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text: continue
                    
                    bbox = span.get("bbox", (0,0,0,0))
                    tb = TextBlock(
                        text=text,
                        x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3],
                        page=page_num,
                        font_size=span.get("size", 0),
                        font=span.get("font", ""),
                        flags=span.get("flags", 0)
                    )
                    page_blocks.append(tb)
        
        # 2. Detect Columns & Reading Order
        # We need numpy/sklearn for clustering, otherwise fallback to 1 col
        try:
            if np:
                cols = _detect_columns(page_blocks, w)
            else:
                cols = ColumnStructure(page_num, 1, [])
        except Exception as e:
            print(f"Column detection failed (page {page_num}): {e}")
            cols = ColumnStructure(page_num, 1, [])
            
        analysis.column_structure.append(cols)
        
        # 3. Sort Blocks
        sorted_page_blocks = _sort_reading_order(page_blocks, cols)
        all_blocks.extend(sorted_page_blocks)
        
    analysis.text_blocks = all_blocks
    doc.close()

    # 4. Integrate ML Regions (PubLayNet X101) or Fallback
    ml_regions_found = False
    try:
        # Render page to image for X101
        # Use first page (index 0) from cached doc or re-open
        # To avoid re-opening, we could have kept page object, but doc closed.
        # Re-rendering simplistic for now.
        
        # NOTE: doc was closed above! 
        # Optimized: Only render if we have the module.
        from wordtolatex.layout_x101 import detect_layout
        
        doc_render = fitz.open(pdf_path)
        for i in range(len(doc_render)):
            page = doc_render[i]
            pix = page.get_pixmap(dpi=150) # Moderate DPI for speed/accuracy trade-off
            
            # Convert to numpy for LayoutParser (RGB)
            if np:
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4: # RGBA -> RGB
                     img_array = np.ascontiguousarray(img_array[..., :3])
                
                # Detect
                x101_regions = detect_layout(img_array)
                
                if x101_regions:
                    ml_regions_found = True
                    for r in x101_regions:
                        analysis.regions.append(LayoutRegion(
                            bbox=tuple(r.bbox), # Convert list to tuple
                            label=r.label,
                            confidence=r.confidence,
                            text=None,
                            page=i,
                            source="x101"
                        ))
        doc_render.close()
            
    except ImportError as e:
        print(f"Warning: Layout X101 import failed: {e}")
    except Exception as e:
        print(f"Warning: Layout X101 analysis failed: {e}")
        
    # 5. Fallback Heuristics
    if not ml_regions_found:
        print("Info: No ML regions found (or ML missing). Running heuristics...")
        try:
            from wordtolatex.layout_heuristics import detect_layout_heuristics
            h_regions = detect_layout_heuristics(str(pdf_path))
            for r in h_regions:
                # heuristic returns LayoutRegion-like object
                analysis.regions.append(LayoutRegion(
                    bbox=r.bbox,
                    label=r.label,
                    confidence=r.confidence,
                    page=r.page,
                    source="heuristic"  # FIX: Set source
                ))
        except Exception as e:
            print(f"Warning: Heuristic analysis failed: {e}")
    
    return analysis


def normalize_text(text: str) -> str:
    """Normalize text for matching.
    
    - NFKC normalization (converts ﬁ → fi, etc.)
    - Remove soft hyphens
    - Collapse whitespace
    - Replace common ligatures
    """
    import unicodedata
    
    # NFKC normalization (handles ligatures like ﬁ → fi)
    text = unicodedata.normalize('NFKC', text)
    
    # Remove soft hyphens and zero-width chars
    text = text.replace('\u00AD', '')  # soft hyphen
    text = text.replace('\u200B', '')  # zero-width space
    text = text.replace('\uFEFF', '')  # BOM
    
    # Collapse whitespace
    text = ' '.join(text.split())
    
    # Common cleanup
    text = text.strip()
    
    return text


def compare_layouts(original: LayoutAnalysis, generated: LayoutAnalysis) -> Dict:
    """Compare layouts using region awareness with improved matching.
    
    Improvements over exact matching:
    - Text normalization (NFKC, ligatures, whitespace)
    - Duplicate handling (text -> list[blocks], matched in reading order)
    - Reading order matching (sort by page, y, x)
    
    Returns dict with:
    - original_blocks: count of blocks in original
    - generated_blocks: count of blocks in generated
    - matched_pairs: count of text blocks that matched
    - match_method: 'normalized' (improved) or 'exact' (legacy)
    - region_shifts: drift by region type for matched blocks with y_diff > 20pt
    """
    from collections import defaultdict
    
    report = {
        "original_blocks": len(original.text_blocks),
        "generated_blocks": len(generated.text_blocks),
        "matched_pairs": 0,
        "match_method": "normalized",
        "differences": [],
        "region_shifts": {},
    }
    
    # Build normalized text -> list of blocks (handles duplicates)
    orig_by_text = defaultdict(list)
    gen_by_text = defaultdict(list)
    
    for b in original.text_blocks:
        norm = normalize_text(b.text)
        if norm:  # Skip empty after normalization
            orig_by_text[norm].append(b)
    
    for b in generated.text_blocks:
        norm = normalize_text(b.text)
        if norm:
            gen_by_text[norm].append(b)
    
    # Sort each list by reading order (page, y, x)
    def reading_order(block):
        return (block.page, round(block.y0, 1), block.x0)
    
    for blocks in orig_by_text.values():
        blocks.sort(key=reading_order)
    for blocks in gen_by_text.values():
        blocks.sort(key=reading_order)
    
    # Get page dimensions for normalizing
    orig_w, orig_h = original.page_dims[0] if original.page_dims else (612, 792)
    
    # Match blocks: pair in reading order
    for norm_text, orig_blocks in orig_by_text.items():
        if norm_text in gen_by_text:
            gen_blocks = gen_by_text[norm_text]
            
            # Match as many as we can (min of counts)
            for i in range(min(len(orig_blocks), len(gen_blocks))):
                orig_block = orig_blocks[i]
                gen_block = gen_blocks[i]
                
                report["matched_pairs"] += 1
                
                y_diff = abs(orig_block.y0 - gen_block.y0)
                
                if y_diff > 20: 
                    # Attribute to region
                    region_label = "unknown"
                    cx = (orig_block.center_x) / orig_w
                    cy = ((orig_block.y0 + orig_block.y1)/2) / orig_h
                    
                    for r in original.regions:
                        if r.bbox[0] <= cx <= r.bbox[2] and r.bbox[1] <= cy <= r.bbox[3]:
                            region_label = r.label
                            break
                    
                    if region_label not in report["region_shifts"]:
                        report["region_shifts"][region_label] = {"count": 0, "total_y_shift": 0}
                    report["region_shifts"][region_label]["count"] += 1
                    report["region_shifts"][region_label]["total_y_shift"] += y_diff

    return report


def create_layout_hints(analysis: LayoutAnalysis) -> Dict:
    """Create layout hints (Regions, Columns, Page Box)."""
    hints = {
        "regions": [],
        "columns": [],
        "page_box_type": "canonical",  # "cropbox" or "mediabox" based on get_canonical_rect logic
        "right_margin": 0,
        "line_spacing": 0,
        "section_positions": [],
        "body_font_family": "Calibri",
    }
    
    # 0. Dominant Font
    font_counts = {}
    for b in analysis.text_blocks:
        if b.font and len(b.text) > 3: # Ignore tiny fragments
            font_counts[b.font] = font_counts.get(b.font, 0) + len(b.text)
    
    if font_counts:
        hints["body_font_family"] = max(font_counts, key=font_counts.get)
        
    # 1. Regions (Normalized)
    for r in analysis.regions:
        hints["regions"].append({
            "label": r.label,
            "bbox": r.bbox,
            "confidence": r.confidence,
            "page": r.page,
            "source": r.source  # FIX: Include source
        })
        
    # 2. Columns
    for col in analysis.column_structure:
        hints["columns"].append({
            "page": col.page_index,
            "count": col.count,
            "boundaries": col.boundaries
        })
        
    # 3. Margins & Spacing (Legacy/Robust)
    right_blocks = analysis.get_right_aligned_texts()
    if right_blocks:
        right_x_positions = [b.x1 for b in right_blocks]
        hints["right_margin"] = max(right_x_positions)
        
    # Estimate line spacing (y-diffs)
    # Collect all unique Y positions
    y_pos = sorted(list(set(round(b.y0, 1) for b in analysis.text_blocks)))
    if len(y_pos) > 1:
        spacings = []
        for i in range(1, min(50, len(y_pos))):
            diff = y_pos[i] - y_pos[i-1]
            if 8 < diff < 25: # Typical single/double spacing
                spacings.append(diff)
        if spacings:
            hints["line_spacing"] = sum(spacings) / len(spacings)
            
    # Section Headers
    headers = analysis.get_section_headers()
    hints["section_positions"] = [(h.text, h.y0) for h in headers]
            
    return hints


if __name__ == "__main__":
    # Test with resume PDFs
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
        analysis = extract_layout(pdf_path)
        
        print(f"Pages: {analysis.page_count}")
        print(f"Text blocks: {len(analysis.text_blocks)}")
        # print(f"Line positions: {len(analysis.line_positions)}") # Removed attribute
        print(f"Right-aligned texts: {len(analysis.get_right_aligned_texts())}")
        
        print("\nFirst 10 text blocks:")
        for block in analysis.text_blocks[:10]:
            print(f"  [{block.x0:.0f},{block.y0:.0f}] {block.text[:50]}")
            
        print("\nDetected Regions (Source):")
        for r in analysis.regions:
            print(f"  {r.label} ({r.source}): {r.bbox} conf={r.confidence:.2f}")
