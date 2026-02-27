"""Model Arbitrator - Compare and Verify Layout Models using Gemini Oracle.

This module implements the "Gemini as Judge" strategy:
1. Run Gemini Vision on a page to get Ground Truth (GT).
2. Run Offline Models (LayoutParser, Heuristics) to get Predictions.
3. Score Predictions against GT using IoU and F1-score.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz
import numpy as np

# Import our models
from wordtolatex.gemini_llm import analyze_layout_with_vision
# We will import StructureScanner inside the class to handle optional dependencies

@dataclass
class BBox:
    """Normalized Bounding Box [x0, y0, x1, y1] (0.0 - 1.0)."""
    coords: Tuple[float, float, float, float]
    label: str
    
    @property
    def area(self) -> float:
        w = max(0, self.coords[2] - self.coords[0])
        h = max(0, self.coords[3] - self.coords[1])
        return w * h

    def iou(self, other: 'BBox') -> float:
        """Calculate Intersection over Union."""
        xA = max(self.coords[0], other.coords[0])
        yA = max(self.coords[1], other.coords[1])
        xB = min(self.coords[2], other.coords[2])
        yB = min(self.coords[3], other.coords[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
            
        boxAArea = self.area
        boxBArea = other.area
        
        return interArea / float(boxAArea + boxBArea - interArea)


@dataclass
class ArbitrationResult:
    """Result of verifying a single model against Oracle."""
    model_name: str
    precision: float
    recall: float
    f1: float
    matches: List[Tuple[BBox, BBox]] = field(default_factory=list) # (GT, Pred)
    false_positives: List[BBox] = field(default_factory=list)
    false_negatives: List[BBox] = field(default_factory=list)


class ModelArbitrator:
    """Orchestrates model comparison."""
    
    def __init__(self):
        # Initialize scanners gracefully
        try:
            from wordtolatex.layout_ml import StructureScanner
            self.ml_scanner = StructureScanner()
            self.has_ml = True
        except ImportError:
            self.ml_scanner = None
            self.has_ml = False
            print("Warning: LayoutML not available for arbitration.")

    def verify_page(self, pdf_path: Path, page_num: int) -> Dict[str, ArbitrationResult]:
        """Run verification for a page."""
        
        # 1. Get Ground Truth (Gemini)
        # We need to render the page to an image first for Gemini
        img_path = self._render_page_to_image(pdf_path, page_num)
        print(f"Asking Gemini Oracle to verify: {img_path}")
        
        gemini_raw = analyze_layout_with_vision(img_path)
        gt_boxes = [
            BBox(tuple(r["bbox"]), r["label"]) 
            for r in gemini_raw
        ]
        
        results = {}
        
        # 2. Evaluate ML (LayoutParser)
        if self.has_ml:
            print("Running LayoutParser...")
            ml_raw = self.ml_scanner.analyze(pdf_path, page_num)
            pred_boxes = [
                BBox(tuple(r.bbox), r.label) 
                for r in ml_raw.get("regions", [])
            ]
            results["layout_parser"] = self._score_predictions(gt_boxes, pred_boxes, "layout_parser")
            
        # 3. Evaluate Heuristics (Simple/PyMuPDF)
        # TODO: Implement specific heuristic checks if needed
        # For now, we can check for "Figures" by extracting images via fitz
        print("Running Heuristics...")
        heuristic_boxes = self._run_heuristics(pdf_path, page_num)
        results["heuristics"] = self._score_predictions(gt_boxes, heuristic_boxes, "heuristics")
        
        return results

    def _render_page_to_image(self, pdf_path: Path, page_num: int) -> Path:
        """Render page to temporary image."""
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pix = page.get_pixmap(dpi=150)
        output = pdf_path.parent / f"temp_page_{page_num}.png"
        pix.save(output)
        doc.close()
        return output

    def _run_heuristics(self, pdf_path: Path, page_num: int) -> List[BBox]:
        """Extract bounding boxes using PyMuPDF heuristics."""
        boxes = []
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # 1. Images -> Figures
        images = page.get_images()
        for img in images:
            # get_image_rects is available in newer PyMuPDF
            try:
                rects = page.get_image_rects(img[0])
                for r in rects:
                    # Normalize
                    bbox = (r.x0 / page.rect.width, r.y0 / page.rect.height, 
                            r.x1 / page.rect.width, r.y1 / page.rect.height)
                    boxes.append(BBox(bbox, "figure"))
            except:
                pass
                
        # 2. Drawings -> Figures/Tables (hard to distinguish)
        # For now, let's just count images.
        
        doc.close()
        return boxes

    def _score_predictions(self, gt: List[BBox], preds: List[BBox], model_name: str, iou_thresh=0.5) -> ArbitrationResult:
        """Compute metrics."""
        matches = []
        fp = list(preds) # Start with all as FP
        fn = []
        
        matched_gt_indices = set()
        
        for p_idx, p in enumerate(preds):
            best_iou = 0
            best_gt_idx = -1
            
            for g_idx, g in enumerate(gt):
                if g.label != p.label and not (g.label in ["figure", "table"] and p.label in ["figure", "table"]):
                    # Strict label matching (allow confusion between fig/table?)
                    # Let's enforce strict for now
                    continue
                
                iou = p.iou(g)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx
            
            if best_iou >= iou_thresh:
                # Match found
                if best_gt_idx not in matched_gt_indices:
                    matches.append((gt[best_gt_idx], p))
                    matched_gt_indices.add(best_gt_idx)
                    if p in fp: fp.remove(p) # It's valid, remove from FP
                else:
                    # Duplicate detection for same GT? Count as FP
                    pass
            else:
                # Keep in FP
                pass
                
        # Calculate FN (GT that wasn't matched)
        for g_idx, g in enumerate(gt):
            if g_idx not in matched_gt_indices:
                fn.append(g)
                
        # Metrics
        tp_count = len(matches)
        fp_count = len(fp)
        fn_count = len(fn)
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return ArbitrationResult(
            model_name=model_name,
            precision=precision,
            recall=recall,
            f1=f1,
            matches=matches,
            false_positives=fp,
            false_negatives=fn
        )
