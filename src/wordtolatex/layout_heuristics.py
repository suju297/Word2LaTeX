"""Layout Heuristics - Non-ML fallbacks for region detection.

This module provides deterministic fallback methods to detect layout regions
(Figures, Textboxes, Tables) using PyMuPDF and basic image analysis,
ensuring the Layout Oracle is useful even without heavy ML dependencies.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import fitz  # PyMuPDF
from dataclasses import dataclass

@dataclass
class HeuristicRegion:
    bbox: Tuple[float, float, float, float]  # Normalized [0..1]
    label: str
    confidence: float
    page: int
    source: str = "heuristic"


def detect_figures(page: fitz.Page, page_num: int) -> List[HeuristicRegion]:
    """Detect images as semantic 'Figure' regions."""
    regions = []
    w, h = page.rect.width, page.rect.height
    
    # get_images returns (xref, smask, width, height, bpc, colorspace, alt.colorspace, name, filter)
    # create specific search for image rects
    image_list = page.get_images()
    
    for img in image_list:
        xref = img[0]
        # Find where this image is drawn
        rects = page.get_image_rects(xref)
        for r in rects:
            # Normalize
            regions.append(HeuristicRegion(
                bbox=(r.x0 / w, r.y0 / h, r.x1 / w, r.y1 / h),
                label="figure",
                confidence=1.0, # Deterministic presence
                page=page_num,
                source="heuristic"
            ))
            
    return regions

def detect_textboxes(page: fitz.Page, page_num: int) -> List[HeuristicRegion]:
    """Detect vector rectangles that likely enclose text (TextBoxes)."""
    regions = []
    w, h = page.rect.width, page.rect.height
    
    paths = page.get_drawings()
    for p in paths:
        # Heuristic: A textbox is often a closed vector path (rect) 
        # that is not filling the whole page background
        # Must be a significant size but not the whole page margins
        r = p["rect"]
        
        # 1. Size Gating: Ignore tiny noise or full page backgrounds
        if r.width < 50 or r.height < 20: continue # Too small
        if r.width > w * 0.95 and r.height > h * 0.95: continue # Background
        
        # 2. Type Check: Must have stroke or fill (visible box)
        if p["type"] not in ["s", "f", "fs"]:
             continue

        # 3. Position Gating: Textboxes are often "floating" or "inset"
        # If it's perfectly standard margin-to-margin, it might be just a separator line or background
        # But we'll leave that policy to the resolver. Here we just detect "Box".
        # We check if it's "filled" specifically for stronger confidence
        is_filled = p["type"] in ["f", "fs"]
        
        regions.append(HeuristicRegion(
            bbox=(r.x0 / w, r.y0 / h, r.x1 / w, r.y1 / h),
            label="textbox",
            confidence=0.8 if is_filled else 0.5, 
            page=page_num,
            source="heuristic"
        ))
             
    # TODO: Merge overlapping boxes (v2)
    return regions

def detect_tables(page: fitz.Page, page_num: int) -> List[HeuristicRegion]:
    """Detect tables using OpenCV with a Validation Gate."""
    regions = []
    
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
        
    w, h = page.rect.width, page.rect.height
    
    # Render page to image (grayscale)
    pix = page.get_pixmap(dpi=150, colorspace="GRAY")
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    
    # 1. Binarize (inverted: text/lines are white, background black)
    # Use adaptive thresholding to handle noise
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
                                   
    # 2. Detect Horizontal and Vertical Lines
    # Scale kernels relative to image size (150 dpi)
    # Length ~ 1/40 of dimension is decent for table rulings
    h_kernel_len = int(img.shape[1] / 40)
    v_kernel_len = int(img.shape[0] / 40)
    
    # Horizontal kernel
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
    
    # Vertical kernel
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
    
    # 3. Combine to find grid intersections
    # Table Gate Check 1: Must have significant intersections
    # Intersections are where H and V meet
    intersections = cv2.bitwise_and(h_lines, v_lines)
    
    # 4. Formulate Table Mask
    table_mask = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0.0)
    _, table_mask = cv2.threshold(table_mask, 0, 255, cv2.THRESH_BINARY)
    
    # Dialate to connect gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    table_mask = cv2.dilate(table_mask, kernel, iterations=3)
    
    # 5. Find Contours
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        
        # Gate 2: Size Filtering
        if cw < pix.width * 0.2 or ch < pix.height * 0.05: # Too small width or height
            continue
            
        # Gate 3: "Table Gate" - Validate Grid Structure
        # Check intersection density in this ROI
        roi_intersect = intersections[y:y+ch, x:x+cw]
        intersection_count = cv2.countNonZero(roi_intersect)
        
        # Normalize score independent of resolution/size
        # Density = intersections per 10000 pixels (arbitrary scale)
        area = cw * ch
        if area < 100: continue
        
        intersection_density = (intersection_count / area) * 10000
        
        # Heuristic: Valid table needs significant density
        # Threshold tuned: e.g., > 1.5 intersections per 10k pixels?
        # Absolute count filter (min 4) still good for tiny tables
        if intersection_count < 4 or intersection_density < 0.5:
            continue
            
        # Gate 4: Aspect Ratio Rejection
        # Reject extremely thin lines that might be just separators
        if cw / ch > 20 or ch / cw > 20:
             continue
        
        # Normalize
        regions.append(HeuristicRegion(
            bbox=(x / pix.width, y / pix.height, (x + cw) / pix.width, (y + ch) / pix.height),
            label="table",
            confidence=0.85, # Heuristic confidence
            page=page_num,
            source="heuristic"
        ))
        
    return regions

def detect_borderless_tables(page: fitz.Page, page_num: int) -> List[HeuristicRegion]:
    """Detect tables that have no borders but grid-like text alignment."""
    # Simplified approach:
    # 1. Get all text blocks
    # 2. Cluster their X-coordinates
    # 3. Check for repeating Y-intervals (rows)
    # This is a stub for v1 as the logic is complex to implement robustly in one go.
    # Future work: Implement full text clustering.
    return []


def detect_layout_heuristics(pdf_path: str) -> List[HeuristicRegion]:
    """Run all heuristics on the document."""
    all_regions = []
    
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        # 1. Figures
        all_regions.extend(detect_figures(page, page_num))
        
        # 2. Textboxes
        all_regions.extend(detect_textboxes(page, page_num))
        
        # 3. Tables (OpenCV)
        all_regions.extend(detect_tables(page, page_num))
        
    doc.close()
    return all_regions
