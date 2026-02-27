import json
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np

OUTPUT_ROOT = Path("tests/corpus").resolve()

def is_page_empty(doc, page_idx):
    page = doc[page_idx]
    
    # 1. Check visual intensity (72 DPI sufficient for this)
    pix = page.get_pixmap(dpi=72)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        gray = np.mean(img[..., :3], axis=2)
    else:
        gray = img[..., 0]
    mean_intensity = gray.mean()
    
    # 2. Check text content
    page_text = page.get_text().strip()
    has_text = len(page_text) > 0

    # Logic matched to scripts/image_diff.py
    if mean_intensity > 254.5:
         return True
    elif mean_intensity > 250.0 and not has_text:
         return True
         
    return False

for case_dir in OUTPUT_ROOT.iterdir():
    if not case_dir.is_dir():
        continue
    
    gen_pdf = case_dir / "gen.pdf"
    report_path = case_dir / "report.json"
    
    if gen_pdf.exists() and report_path.exists():
        try:
            doc = fitz.open(gen_pdf)
            empty_pages = [i for i in range(len(doc)) if is_page_empty(doc, i)]
            doc.close()
            
            with open(report_path) as f:
                report = json.load(f)
            
            if "image_diff" not in report:
                report["image_diff"] = {}
                
            report["image_diff"]["gen_empty_pages"] = empty_pages
            report["image_diff"]["has_empty_gen_pages"] = len(empty_pages) > 0
            
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
                
            status = "EMPTY PAGES FOUND" if empty_pages else "OK"
            print(f"Patched {case_dir.name}: {status}")
            
        except Exception as e:
            print(f"Failed to patch {case_dir.name}: {e}")
