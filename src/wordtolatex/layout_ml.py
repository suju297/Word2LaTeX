"""
Layout Analysis using parallel scanners (Style Analysis + Structure Analysis).

This module analyzes document images to extract:
1. Style metrics (font size, margins, spacing) -> via PyMuPDF (StyleScanner)
2. Structure regions (tables, figures, lists) -> via LayoutLMv3 (StructureScanner)
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import json
from abc import ABC, abstractmethod

# Lazy imports for heavy ML dependencies
_processor = None
_model = None
_device = None


@dataclass
class LayoutRegion:
    """A detected region in the document."""
    text: str
    bbox: Tuple[int, int, int, int]  # x0, y0, x1, y1
    label: str  # heading, paragraph, list, table, figure
    confidence: float
    font_size_estimate: Optional[float] = None
    page: int = 0
    source: str = "ml"


@dataclass
class DocumentLayoutAnalysis:
    """Complete layout analysis of a document page."""
    page_num: int
    width: int
    height: int
    regions: List[LayoutRegion]
    
    # Inferred style parameters
    body_font_size: float = 11.0
    line_spacing: float = 1.0
    margin_top: float = 1.0
    margin_bottom: float = 1.0
    margin_left: float = 1.0
    margin_right: float = 1.0
    font_family: str = "sans"
    visual_elements: List[Dict[str, Any]] = None


class LayoutScanner(ABC):
    """Abstract base class for layout scanners."""
    
    @abstractmethod
    def analyze(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        """Analyze a page and return extracted features."""
        pass


class StyleScanner(LayoutScanner):
    """Extracts style metrics (font, spacing, margins) using PyMuPDF."""
    
    def analyze(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        import fitz  # PyMuPDF
        
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Extract text blocks with positions
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        
        font_sizes = []
        line_positions = []
        x_positions = []
        
        for block in blocks:
            if block["type"] != 0:  # Skip non-text
                continue
            
            for line in block.get("lines", []):
                line_positions.append(line["bbox"][1])
                for span in line.get("spans", []):
                    # Consider only meaningful text
                    if span.get("text", "").strip():
                        if span.get("size"):
                            font_sizes.append(span["size"])
                        # Track X start for margin detection
                        x_positions.append(span["bbox"][0])


        
        # 1. Determine Body Font Size (weighted mode)
        body_font_size = 11.0
        if font_sizes:
            from collections import Counter
            # Round to nearest 0.1 to group close sizes (e.g. 10.98 -> 11.0)
            rounded_sizes = [round(s, 1) for s in font_sizes]
            size_counts = Counter(rounded_sizes)
            if size_counts:
                # Get common sizes
                common = size_counts.most_common(3)
                # Heuristic: The body text is likely the smallest of the *common* sizes 
                # (to avoid picking huge headers even if frequent)
                # But typically body text is simply the MOST frequent.
                body_font_size = common[0][0]

        # 2. Determine Line Spacing (Mode of gaps)
        line_spacing = 1.0
        if len(line_positions) > 1:
            sorted_pos = sorted(set(line_positions))
            # Calculate all adjacent gaps
            gaps = [sorted_pos[i+1] - sorted_pos[i] for i in range(len(sorted_pos)-1)]
            
            # Filter gaps: Must be positive and reasonable (e.g., < 3x font size)
            # This discards paragraph breaks and section jumps
            valid_gaps = [g for g in gaps if 0.5 * body_font_size < g < 2.0 * body_font_size]
            
            if valid_gaps:
                # Round gaps to nearest 0.5 pt to handle scanning noise
                rounded_gaps = [round(g * 2) / 2 for g in valid_gaps]
                gap_counts = Counter(rounded_gaps)
                
                # The most frequent gap is the standard line spacing
                mode_gap = gap_counts.most_common(1)[0][0]
                
                # Calculate ratio
                raw_ratio = mode_gap / body_font_size
                
                # Map to standard LaTeX spacings or keep exact
                if raw_ratio < 1.05: line_spacing = 1.0
                elif raw_ratio < 1.25: line_spacing = 1.15
                elif raw_ratio < 1.4: line_spacing = 1.25
                elif raw_ratio < 1.6: line_spacing = 1.5
                elif raw_ratio < 2.2: line_spacing = 2.0
                else: line_spacing = raw_ratio
        
        # 3. Determine Margins (heuristic)
        margin_left = 1.0  # Default 1 inch
        if x_positions:
            min_x = min(x_positions)
            margin_left = min_x / 72.0  # Convert points to inches

        # 4. Determine Font Family
        font_family = "sans"
        try:
            fonts = page.get_fonts()
            serif_count = 0
            total_fonts = len(fonts)
            # print(f"[DEBUG] Found {total_fonts} fonts")
            for f in fonts:
                fname = f[3].lower()
                # print(f"[DEBUG] Font: {fname}")
                if "serif" in fname or "roman" in fname or "times" in fname:
                    serif_count += 1
            
            if total_fonts > 0:
                ratio = serif_count / total_fonts
                # print(f"[DEBUG] Serif Ratio: {ratio:.2f}")
                if ratio > 0.4:
                    font_family = "serif"
        except Exception:
            pass
            
        # 5. Extract Visual Elements (Horizontal Lines)
        visual_elements = []
        try:
            drawings = page.get_drawings()
            w, h = page.rect.width, page.rect.height
            for d in drawings:
                rect = d['rect']
                w_rect, h_rect = rect.width, rect.height
                
                # 1. Background Shape Heuristic (>90% of page)
                if (w_rect * h_rect) > (w * h * 0.9):
                     bbox = (rect.x0 / w, rect.y0 / h, rect.x1 / w, rect.y1 / h)
                     visual_elements.append({
                        "type": "background_shape",
                        "bbox": bbox,
                        "color": d.get('fill') # Tuple (r, g, b)
                    })
                
                # 2. General Shapes (Footer lines, Progress bars, Separators)
                # Filter out tiny noise (e.g. < 2pt x 2pt)
                elif w_rect > 2 and h_rect > 2:
                    bbox = (rect.x0 / w, rect.y0 / h, rect.x1 / w, rect.y1 / h)
                    visual_elements.append({
                        "type": "shape",
                        "bbox": bbox,
                        "color": d.get('fill') or d.get('color'),
                        "fill": d.get('fill'),
                        "stroke": d.get('color')
                    })
        except Exception:
            pass

        doc.close()
        return {
            "body_font_size": body_font_size,
            "line_spacing": line_spacing,
            "margin_left": margin_left,
            "margin_right": 1.0, 
            "margin_top": 1.0,
            "margin_bottom": 1.0,
            "font_family": font_family,
            "visual_elements": visual_elements
        }


def _load_layoutlm_model():
    """Lazy load LayoutLMv3."""
    global _processor, _model, _device
    if _model is not None:
        return _processor, _model, _device
    
    import torch
    from transformers import AutoProcessor, AutoModelForTokenClassification
    
    print("Loading LayoutLMv3 model...")
    if torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")
        
    # Use layoutlmv3-base-finetuned-publaynet for table/figure/list detection
    # NOTE: Using a public fine-tuned model for document layout analysis
    model_name = "microsoft/layoutlmv3-base" 
    # Ideally should be a PubLayNet finetune for Tables/Figures specifically,
    # but for this demo we stick to base or a known fine-tune if exists.
    # Since we can't easily download random HuggingFace models without verification,
    # we will use the base model and simulate the labels if standard weights don't have them,
    # OR better: use logic to assume specific output channels.
    
    _processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
    _model = AutoModelForTokenClassification.from_pretrained(model_name)
    _model.to(_device)
    _model.eval()
    return _processor, _model, _device


class StructureScanner(LayoutScanner):
    """Extracts structural regions (Tables, Figures) using LayoutParser (Detectron2)."""
    
    def analyze(self, pdf_path: Path, page_num: int) -> Dict[str, Any]:
        regions = []
        try:
            # Check for ML dependencies
            from ultralytics import YOLO
            import cv2
            import numpy as np
            import fitz
            from PIL import Image

            # Render PDF page to image for YOLO
            # We use higher DPI for detection
            doc = fitz.open(pdf_path)
            page = doc[page_num]
            w, h = page.rect.width, page.rect.height
            pix = page.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            
            doc.close()
            
            # Load Model (Lazy)
            global _model
            _model_version = None
            if _model is None:
                # Layout Oracle Models (priority order: V5 > V4 > V3 > V2)
                # Check for local model files
                model_dir = Path("models")
                model_candidates = [
                    ("model_v5.pt", "V5"),  # YOLOv11s, 1024px, mAP50=40.8%
                    ("model_v4.pt", "V4"),
                    ("model_v3.pt", "V3"),
                    ("model_v2.pt", "V2"),
                ]
                for model_file, version in model_candidates:
                    model_path = model_dir / model_file
                    if model_path.exists():
                        print(f"Loading Layout Oracle {version} from: {model_path.resolve()}")
                        _model = YOLO(str(model_path))
                        _model_version = version
                        print(f"  Classes: {list(_model.names.values())}")
                        break
                else:
                    # Fallback or initialization
                    print("Warning: Custom model not found. Using generic yolov8n.pt (will fail on layout classes).")
                    _model = YOLO("yolov8n.pt")
                    _model_version = "fallback" 

            # Inference
            # convert RGB to BGR for opencv/yolo compatibility if needed, 
            # though YOLO handles RGB numpy arrays from PIL/PyMuPDF fine mostly.
            # PyMuPDF samples are RGB usually.
            # Use imgsz=1024 to match V5 training resolution for best accuracy
            results = _model(img, verbose=False, imgsz=1024)
            
            # Process results
            # layout is a list of Results objects (one per image)
            for r in results:
                img_h, img_w = r.orig_shape # Should match img.shape
                
                for box in r.boxes:
                    # Bounding Box (xyxy)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Confidence & Class
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = _model.names[cls_id].lower() # e.g., 'table', 'title', 'footer'
                    
                    # Normalize to [0..1]
                    bbox = (
                        x1 / img_w,
                        y1 / img_h,
                        x2 / img_w,
                        y2 / img_h
                    )
                    
                    # *** BBOX PROVENANCE (for debugging drift) ***
                    bbox_px = (int(x1), int(y1), int(x2), int(y2))
                    
                    # Filter useful regions (Layout Oracle focus)
                    # V5 classes: Text, Title, List, Table, Figure, Caption, Footer, Header, Formula
                    if label in ["table", "figure", "list", "title", "header", "footer", "caption", "formula"]:
                        region = LayoutRegion(
                            text=None, # YOLO doesn't give text content directly
                            bbox=bbox,
                            label=label,
                            confidence=conf,
                            page=page_num,
                            source="ml_v5"
                        )
                        # Store provenance as extra attrs
                        region.render_dpi = 150
                        region.render_w_px = img_w
                        region.render_h_px = img_h
                        region.bbox_px = bbox_px
                        region.origin = "image_top_left"
                        
                        regions.append(region)
                    
        except ImportError as e:
            print(f"YOLO/Ultralytics ImportError: {e}")
            pass 
        except Exception as e:
            print(f"Layout Oracle inference failed: {e}")
            import traceback
            traceback.print_exc()
            
        return {"regions": regions}


def analyze_document(pdf_path: Path) -> DocumentLayoutAnalysis:
    """Run all scanners and merge results."""
    print(f"Scanning document: {pdf_path}")
    
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[0]
    width, height = page.rect.width, page.rect.height
    doc.close()
    
    # 1. Style Analysis (Robust, PyMuPDF only)
    style_scanner = StyleScanner()
    style_results = style_scanner.analyze(pdf_path, 0)
    
    # 2. Structure Analysis (Optional, ML-heavy)
    structure_results = {"regions": []}
    try:
        structure_scanner = StructureScanner()
        structure_results = structure_scanner.analyze(pdf_path, 0)
    except Exception as e:
        print(f"Skipping StructureScanner due to error: {e}")
    
    return DocumentLayoutAnalysis(
        page_num=0,
        width=width,
        height=height,
        regions=structure_results.get("regions", []),
        body_font_size=style_results["body_font_size"],
        line_spacing=style_results["line_spacing"],
        margin_left=style_results["margin_left"],
        margin_right=style_results["margin_right"],
        margin_top=style_results["margin_top"],
        margin_bottom=style_results["margin_bottom"],
        font_family=style_results.get("font_family", "sans"),
        visual_elements=style_results.get("visual_elements", [])
    )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        layout = analyze_document(Path(sys.argv[1]))
        print(f"Document Analysis for {sys.argv[1]}")
        print(f"  Body Font: {layout.body_font_size}pt ({layout.font_family})")
        print(f"  Line Spacing: {layout.line_spacing}x")
        print(f"  Visual Elements: {len(layout.visual_elements) if layout.visual_elements else 0}")
        print(f"  Regions Detected: {len(layout.regions)}")
        for r in layout.regions:
            print(f"    - {r.label} at {r.bbox}")
