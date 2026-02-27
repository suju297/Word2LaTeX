"""PDF Cropper - Extract regions from PDF as images.

This module provides functionality to crop specific regions from a PDF page
based on normalized bounding boxes, used for image fallback strategies.
"""

from pathlib import Path
import fitz  # PyMuPDF
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

def crop_pdf_region(
    pdf_path: Path,
    page_index: int,
    bbox: List[float],
    output_path: Path,
    dpi: int = 300
) -> bool:
    """Crop a region from a PDF page and save as image.
    
    Args:
        pdf_path: Path to source PDF
        page_index: Page number (0-indexed)
        bbox: Normalized bounding box [x0, y0, x1, y1] (0.0 - 1.0)
        output_path: Path to save the cropped image (e.g. .png)
        dpi: Resolution for the output image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        doc = fitz.open(pdf_path)
        if page_index >= len(doc):
            logger.error(f"Page index {page_index} out of range for {pdf_path}")
            return False
            
        page = doc[page_index]
        
        # Get page dimensions
        rect = page.rect
        width = rect.width
        height = rect.height
        
        # Convert normalized bbox to PDF coordinates
        x0, y0, x1, y1 = bbox
        
        # Add slight padding (1%) to ensure we capture borders
        pad_x = 0.01
        pad_y = 0.01
        
        clip_rect = fitz.Rect(
            max(0, x0 - pad_x) * width,
            max(0, y0 - pad_y) * height,
            min(1, x1 + pad_x) * width,
            min(1, y1 + pad_y) * height
        )
        
        # Render the clipped region
        # matrix = fitz.Matrix(zoom, zoom) where zoom = dpi / 72
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        
        pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(output_path))
        
        doc.close()
        return True
        
    except Exception as e:
        logger.error(f"Failed to crop PDF region: {e}")
        return False
