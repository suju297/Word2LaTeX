
import layoutparser as lp
import fitz
import sys
import time
from pathlib import Path

def test_layoutparser(pdf_path):
    print(f"Testing LayoutParser on {pdf_path}...")
    
    # Initialize CPU model with local files
    try:
        model = lp.Detectron2LayoutModel(
            config_path='config.yml',
            model_path='model_final.pth',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
            label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
            device='cpu' # Force CPU
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    doc = fitz.open(pdf_path)
    if len(doc) > 1:
        page = doc[1] # Test second page (usually 2-col)
        print("Testing Page 2...")
    else:
        page = doc[0]
        print("Testing Page 1...")
    
    # LayoutParser needs an image
    pix = page.get_pixmap(dpi=300) # Higher DPI for better separation
    img_data = pix.tobytes("png")
    
    # Convert to CV2 image/Array
    import numpy as np
    import cv2
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    start = time.time()
    layout = model.detect(image)
    end = time.time()
    
    print(f"\nInference Time (CPU): {end - start:.2f}s")
    print(f"Detected {len(layout)} blocks:")
    
    page_width = image.shape[1]
    
    col_1_blocks = 0
    col_2_blocks = 0
    full_width_blocks = 0
    
    for block in layout:
        x1, y1, x2, y2 = block.coordinates
        width = x2 - x1
        center_x = (x1 + x2) / 2
        
        rel_width = width / page_width
        rel_center = center_x / page_width
        
        type_name = block.type
        print(f" - {type_name}: Width={rel_width:.2f}, CenterX={rel_center:.2f}")
        
        if rel_width > 0.7:
            full_width_blocks += 1
        elif rel_center < 0.5:
            col_1_blocks += 1
        else:
            col_2_blocks += 1
            
    print("\nSummary:")
    print(f" - Full Width Blocks: {full_width_blocks}")
    print(f" - Left Column Blocks: {col_1_blocks}")
    print(f" - Right Column Blocks: {col_2_blocks}")
    
    if col_1_blocks > 0 and col_2_blocks > 0:
        print("\nSUCCESS: Two-Column Layout Detected!")
    else:
        print("\nFAILURE: Columns NOT detected.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_layout_models.py <pdf_path>")
        sys.exit(1)
        
    test_layoutparser(sys.argv[1])
