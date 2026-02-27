"""Vision Oracle - Compare visual layout using Gemini 1.5 Pro.

This script renders the Reference PDF and Generated PDF to images,
then sends both to Gemini to ask for a visual comparison of specific elements
(Progress Bars) to diagnose positioning issues.
"""

import sys
import os
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.wordtolatex.gemini_llm import _init_gemini

def render_page_to_image(pdf_path: str, page_num: int = 0, dpi: int = 150) -> Image.Image:
    """Render a PDF page to a PIL Image."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def compare_layouts(ref_path: str, gen_path: str):
    """Ask Gemini to compare the progress bars in both images."""
    print(f"Comparing:\nRef: {ref_path}\nGen: {gen_path}")
    
    # Render Page 1 (Resume usually page 0)
    # Ensure they exist
    if not os.path.exists(ref_path):
        print(f"Error: Ref not found {ref_path}")
        return
    if not os.path.exists(gen_path):
        print(f"Error: Gen not found {gen_path}")
        return

    ref_img = render_page_to_image(ref_path)
    gen_img = render_page_to_image(gen_path)
    
    model = _init_gemini()
    
    prompt = """Compare these two resume images (Image 1 is Reference, Image 2 is Generated).
Focus specifically on the 'Black Progress Bars' next to the Skills (Creativity, Leadership, Problem Solving).

1. accurately describe the POSITION of the bars in the Reference. Are they aligned with the text? Below it? Centered?
2. accurately describe the POSITION of the bars in the Generated version.
3. What is the detailed visual difference? Is the generated bar too high? Too low? Too short?
4. Look at the Text 'Creativity', 'Leadership'. Is the vertical gap between them the same in both?

Be precise. Use terms like 'aligned with baseline', 'centered vertically', 'overlapping'.
"""
    
    print("Sending to Gemini Vision...")
    try:
        response = model.generate_content([prompt, ref_img, gen_img])
        print("\n--- Gemini Analysis ---")
        print(response.text)
        print("-----------------------")
    except Exception as e:
        print(f"Gemini Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/vision_oracle.py <ref_pdf> <gen_pdf>")
        print("Example: python scripts/vision_oracle.py tests/corpus/Microsoft_Resume1/ref.pdf tests/corpus/Microsoft_Resume1/gen_v2_styled.pdf")
        sys.exit(1)
        
    compare_layouts(sys.argv[1], sys.argv[2])
