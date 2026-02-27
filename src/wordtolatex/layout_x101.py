"""LayoutParser X101 Model Wrapper.

Integrates the "De-facto standard" PubLayNet-trained Mask R-CNN X101 model
for pixel-accurate document layout detection.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import cv2
except ImportError as e:
    cv2 = None
    print(f"DEBUG: cv2 import failed: {e}")

try:
    import numpy as np
except ImportError as e:
    np = None
    print(f"DEBUG: numpy import failed: {e}")

try:
    import layoutparser as lp
except ImportError as e:
    lp = None
    print(f"DEBUG: layoutparser import failed: {e}")

from dataclasses import dataclass

@dataclass
class X101Region:
    """Standardized region from X101 model."""
    bbox: List[float]  # [x0, y0, x1, y1] normalized 0-1
    label: str         # Text, Title, List, Table, Figure
    confidence: float
    text: Optional[str] = None

# Model Configuration
# PubLayNet-trained Mask R-CNN X101-FPN
# Using local copy to avoid iopath naming issues (dl=1)
MODEL_DIR = Path("models/publaynet-x101")
CONFIG_PATH = str(MODEL_DIR / "config.yaml")
WEIGHTS_PATH = str(MODEL_DIR / "model_final.pth")

LABEL_MAP = {
    0: "Text",
    1: "Title",
    2: "List",
    3: "Table",
    4: "Figure"
}

_model_instance = None

def _get_model():
    """Lazy load the heavyweight model."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    
    if lp is None:
        raise ImportError("layoutparser not installed. Install with 'uv sync --extra ml'")
        
    print(f"Loading LayoutParser model from {MODEL_DIR}...")
    
    if not Path(CONFIG_PATH).exists() or not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"Model files not found in {MODEL_DIR}. Please run download setup.")

    # Suppress warnings from detectron2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _model_instance = lp.Detectron2LayoutModel(
            CONFIG_PATH,
            extra_config=[
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5,
                "MODEL.WEIGHTS", WEIGHTS_PATH
            ],
            label_map=LABEL_MAP
        )
    return _model_instance

def detect_layout(image_input: Any) -> List[X101Region]:
    """Detect layout regions in an image (path or numpy array).
    
    Args:
        image_input: standard image path or numpy array (cv2 image)
    """
    model = _get_model()
    
    # Load image if path
    if isinstance(image_input, (str, Path)):
        if cv2 is None:
             raise ImportError("opencv-python not installed")
        image = cv2.imread(str(image_input))
        if image is None:
            raise ValueError(f"Could not read image: {image_input}")
    else:
        image = image_input  # Assume numpy array
        
    # Detect
    layout = model.detect(image)
    
    height, width = image.shape[:2]
    
    regions = []
    for block in layout:
        # LayoutParser returns coordinates in [x1, y1, x2, y2] absolute pixels
        # We need normalized 0-1
        b = block.block
        x1, y1, x2, y2 = b.x_1, b.y_1, b.x_2, b.y_2
        
        # Normalize
        norm_bbox = [
            max(0.0, float(x1) / width),
            max(0.0, float(y1) / height),
            min(1.0, float(x2) / width),
            min(1.0, float(y2) / height)
        ]
        
        regions.append(X101Region(
            bbox=norm_bbox,
            label=block.type,
            confidence=block.score
        ))
        
    return regions
