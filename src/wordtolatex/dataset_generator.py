"""Dataset Generator - Create synthetic training data using Gemini Vision."""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

import fitz  # PyMuPDF
import PIL.Image
from tqdm import tqdm

try:
    from wordtolatex.gemini_llm import _init_gemini
except ImportError:
    # Fallback or local definition if module structure varies
    import google.generativeai as genai
    import os
    def _init_gemini():
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
             raise ValueError("GEMINI_API_KEY not found in environment")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash-exp') # Use latest Flash

logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Generates synthetic dataset by labeling PDF pages with Gemini."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.images_dir = output_dir / "images"
        self.labels_dir = output_dir / "labels"  # YOLO format
        
        # Ensure directories exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO Accumulator
        self.coco_data = {
            "info": {"description": "WordToLaTeX Synthetic Dataset", "year": 2026},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 0, "name": "Text"}, # Implicit background/text
                {"id": 1, "name": "Title"},
                {"id": 2, "name": "List"},
                {"id": 3, "name": "Table"},
                {"id": 4, "name": "Figure"},
                {"id": 5, "name": "Header"},
                {"id": 6, "name": "Footer"}
            ]
        }
        
        self.category_map = {c["name"].lower(): c["id"] for c in self.coco_data["categories"]}
        
        self.image_id_counter = 0
        self.annotation_id_counter = 0
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _init_gemini()
        return self._model

    def process_corpus(self, corpus_dir: Path, limit: int = 0) -> None:
        """Process PDFs in the corpus directory.
        
        Args:
            corpus_dir: Directory containing PDFs
            limit: Maximum number of PDFs to process (0 = no limit)
        """
        pdf_files = list(corpus_dir.glob("**/ref.pdf"))
        print(f"Found {len(pdf_files)} PDFs in {corpus_dir}")
        
        if limit > 0:
            pdf_files = pdf_files[:limit]
            print(f"Limiting to first {limit} PDFs")
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            self._process_pdf(pdf_path)
            
        # Save COCO JSON
        coco_path = self.output_dir / "annotations.json"
        with open(coco_path, "w") as f:
            json.dump(self.coco_data, f, indent=2)
        print(f"Saved COCO annotations to {coco_path}")

    def _process_pdf(self, pdf_path: Path) -> None:
        """Process a single PDF."""
        try:
            doc = fitz.open(pdf_path)
            doc_name = pdf_path.parent.name
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Render to high-res image
                pix = page.get_pixmap(dpi=150) # 150 DPI is usually sufficient for layout
                img_filename = f"{doc_name}_p{page_num}.jpg"
                img_path = self.images_dir / img_filename
                pix.save(str(img_path))
                
                # Label with Gemini
                labels = self._label_with_gemini(img_path)
                
                # Update COCO and Save YOLO
                image_id = self.image_id_counter
                self.image_id_counter += 1
                
                # COCO Image Entry
                self.coco_data["images"].append({
                    "id": image_id,
                    "file_name": img_filename,
                    "width": pix.width,
                    "height": pix.height
                })
                
                # Save YOLO Labels
                yolo_lines = []
                for label in labels:
                    cat_name = label['label'].capitalize()
                    cat_id = self.category_map.get(cat_name.lower())
                    
                    if cat_id is None:
                        continue # Skip unknown categories
                        
                    bbox = label['bbox'] # [xmin, ymin, xmax, ymax] normalized
                    
                    # Store COCO Annotation
                    # COCO bbox format: [x, y, width, height] (absolute pixels)
                    x_px = bbox[0] * pix.width
                    y_px = bbox[1] * pix.height
                    w_px = (bbox[2] - bbox[0]) * pix.width
                    h_px = (bbox[3] - bbox[1]) * pix.height
                    
                    self.coco_data["annotations"].append({
                        "id": self.annotation_id_counter,
                        "image_id": image_id,
                        "category_id": cat_id,
                        "bbox": [x_px, y_px, w_px, h_px],
                        "area": w_px * h_px,
                        "iscrowd": 0,
                        "segmentation": [] # Bbox only
                    })
                    self.annotation_id_counter += 1
                    
                    # YOLO Format: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
                    # Input bbox is [xmin, ymin, xmax, ymax] normalized
                    w_norm = bbox[2] - bbox[0]
                    h_norm = bbox[3] - bbox[1]
                    xc_norm = bbox[0] + w_norm / 2
                    yc_norm = bbox[1] + h_norm / 2
                    
                    yolo_lines.append(f"{cat_id} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
                
                # Write YOLO file
                yolo_path = self.labels_dir / f"{doc_name}_p{page_num}.txt"
                with open(yolo_path, "w") as f:
                    f.write("\n".join(yolo_lines))
                    
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")

    def _label_with_gemini(self, img_path: Path) -> List[Dict]:
        """Send image to Gemini and get bounding boxes."""
        model = self._get_model()
        img = PIL.Image.open(img_path)
        
        prompt = """Analyze this document image for layout object detection.
Task: Detect bounding boxes for the following classes:
- Table
- Figure (charts, images, diagrams)
- List (detect the entire list block, not items)
- Title (main document title)
- Header (running header)
- Footer (page number, running footer)

Format: Return a JSON object with a "regions" key containing a list of objects.
Each object must have:
- "label": One of ["Table", "Figure", "List", "Title", "Header", "Footer"]
- "bbox": [ymin, xmin, ymax, xmax] coordinates normalized to 0-1000 scale (integers).

Example:
{
  "regions": [
    {"label": "Table", "bbox": [150, 100, 450, 900]},
    {"label": "Footer", "bbox": [950, 100, 980, 900]}
  ]
}
Return valid JSON only.
"""
        try:
            response = model.generate_content([prompt, img])
            text = response.text.strip()
            
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            data = json.loads(text)
            regions = data.get("regions", [])
            
            normalized = []
            for r in regions:
                label = r.get("label")
                bbox = r.get("bbox")
                if not label or not bbox or len(bbox) != 4:
                    continue
                    
                # Normalize 0-1000 -> 0.0-1.0
                # Gemini bbox: [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = bbox
                
                # Convert to [xmin, ymin, xmax, ymax]
                n_bbox = [
                    xmin / 1000.0,
                    ymin / 1000.0,
                    xmax / 1000.0,
                    ymax / 1000.0
                ]
                
                # Clamp
                n_bbox = [max(0.0, min(1.0, x)) for x in n_bbox]
                
                normalized.append({
                    "label": label,
                    "bbox": n_bbox
                })
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Gemini labeling failed for {img_path.name}: {e}")
            return []
