import fitz
import sys
from collections import Counter

def analyze_style(pdf_path):
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    print(f"--- Analyzing {pdf_path} ---")
    
    # 1. Fonts
    print("\n[Fonts]")
    fonts = page.get_fonts()
    for f in fonts:
        # (xref, ext, type, basefont, name, coding)
        print(f" - {f[3]} ({f[4]})")
        
    # 2. Colors / Shapes
    print("\n[Background/Shapes]")
    # Get all drawings
    drawings = page.get_drawings()
    bg_candidates = []
    
    w, h = page.rect.width, page.rect.height
    print(f"Page Size: {w}x{h}")
    
    for d in drawings:
        rect = d['rect']
        fill = d.get('fill')
        color = d.get('color')
        
        print(f" - Shape: {rect} | Fill: {fill} | Stroke: {color}")
        
        # Check for horizontal lines / rules
        # Logic: Width > 50pt, Height < 5pt
        if rect.width > 50 and rect.height < 5:
             print(f" - Horizontal Line: {rect} | Color: {d.get('color') or d.get('fill')}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_style(sys.argv[1])
    else:
        print("Usage: python inspect_style.py <pdf_path>")
