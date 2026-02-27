import fitz
import numpy as np
from pathlib import Path

pdfs = [
    "tests/corpus/Microsoft_Resume1/gen.pdf",
    "tests/corpus/Microsoft_Resume_2/gen.pdf"
]

for pdf_path in pdfs:
    path = Path(pdf_path)
    if not path.exists():
        print(f"{path} does not exist")
        continue

    print(f"\nAnalyzing {path.name}...")
    try:
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            # Render at 150 DPI
            pix = page.get_pixmap(dpi=150)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n >= 3:
                gray = np.mean(img[..., :3], axis=2)
            else:
                gray = img[..., 0]
            
            mean_val = gray.mean()
            has_text = bool(page.get_text().strip())
            text_preview = page.get_text().strip()[:50].replace("\n", " ")
            
            print(f"  Page {i+1}: Mean Intensity={mean_val:.4f}, Has Text={has_text}")
            print(f"    Preview: '{text_preview}...'")
            
        doc.close()
    except Exception as e:
        print(f"  Error: {e}")
