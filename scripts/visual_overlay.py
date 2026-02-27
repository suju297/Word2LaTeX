#!/usr/bin/env python3
"""
Diagnostic script to generate a visual overlay of Reference PDF vs Generated PDF.
Ref is rendered in RED.
Gen is rendered in CYAN.
Result:
- Black/Gray = Match
- Red = Missing in Gen
- Cyan = Extra in Gen / Moved
"""
import sys
import argparse
from pathlib import Path
import fitz  # pymupdf
from PIL import Image, ImageChops
import numpy as np

def render_page(doc, page_num, zoom=2.0):
    if page_num >= doc.page_count:
        return None
    page = doc[page_num]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def create_overlay(ref_img, gen_img):
    # Resize to match easiest common size (usually ref size)
    if ref_img.size != gen_img.size:
        print(f"  Resizing Gen {gen_img.size} to Ref {ref_img.size}")
        gen_img = gen_img.resize(ref_img.size, Image.LANCZOS)
    
    # Convert to grayscale
    ref_gray = ref_img.convert("L")
    gen_gray = gen_img.convert("L")
    
    # Invert grayscale (0=white, 255=black) -> (255=ink, 0=paper)
    ref_ink = ImageChops.invert(ref_gray)
    gen_ink = ImageChops.invert(gen_gray)
    
    # Create RGB channels
    # Red channel = Ref Ink
    # Green channel = Gen Ink
    # Blue channel = Gen Ink
    # Result: Ref=Red, Gen=Cyan (G+B)
    
    # We want white background.
    # Logic: Start with White. 
    # Subtract Ref from G+B (leaving Red).
    # Subtract Gen from R (leaving Cyan).
    
    # Simpler approach:
    # R = (255 - Ref_Ink)   (So ink stays low/black, but we want it RED)
    # Wait.
    # Red Overlay means: Ref stuff is Red. Gen stuff is Cyan.
    # R channel: 1.0 where Gen is white, 0.0 where Gen is ink.
    # G channel: 1.0 where Ref is white, 0.0 where Ref is ink.
    # B channel: 1.0 where Ref is white, 0.0 where Ref is ink.
    
    # Where both are ink: R=0, G=0, B=0 -> Black.
    # Where only Ref is ink: R=0, G=1, B=1 -> Cyan? No.
    # Ref Ink -> We want RED (255, 0, 0).
    # Gen Ink -> We want CYAN (0, 255, 255).
    # Background -> White (255, 255, 255).
    
    # Ref Ink (0=paper, 255=ink)
    ref_arr = np.array(ref_ink)
    gen_arr = np.array(gen_ink)
    
    # Initialize canvas white
    h, w = ref_arr.shape
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    
    # Subtract Ref ink from G and B (making it Red)
    # If Ref has ink (255), we want G=0, B=0.
    # So G = G - Ref_Ink
    out[:, :, 1] = np.clip(out[:, :, 1] - ref_arr, 0, 255)
    out[:, :, 2] = np.clip(out[:, :, 2] - ref_arr, 0, 255)
    
    # Subtract Gen ink from R (making it Cyan)
    # If Gen has ink (255), we want R=0.
    out[:, :, 0] = np.clip(out[:, :, 0] - gen_arr, 0, 255)
    
    return Image.fromarray(out)

def main():
    parser = argparse.ArgumentParser(description="Generate visual overlay diff")
    parser.add_argument("doc_id", help="Document ID (e.g. Resume_1)")
    parser.add_argument("--corpus-root", default="tests/corpus", help="Path to corpus root")
    parser.add_argument("--page", type=int, default=0, help="Page number to analyze (0-indexed)")
    args = parser.parse_args()
    
    corpus_root = Path(args.corpus_root)
    doc_dir = corpus_root / args.doc_id
    
    if not doc_dir.exists():
        print(f"Error: {doc_dir} not found")
        sys.exit(1)
        
    ref_path = doc_dir / "ref.pdf"
    gen_path = doc_dir / "gen.pdf"
    
    if not ref_path.exists() or not gen_path.exists():
        print(f"Error: Missing ref.pdf ({ref_path.exists()}) or gen.pdf ({gen_path.exists()})")
        sys.exit(1)
        
    print(f"Processing {args.doc_id} Page {args.page}...")
    doc_ref = fitz.open(ref_path)
    doc_gen = fitz.open(gen_path)
    
    ref_img = render_page(doc_ref, args.page)
    gen_img = render_page(doc_gen, args.page)
    
    if not ref_img or not gen_img:
        print(f"Error: Could not render page {args.page}. Ref pages: {doc_ref.page_count}, Gen pages: {doc_gen.page_count}")
        sys.exit(1)
        
    overlay = create_overlay(ref_img, gen_img)
    out_path = doc_dir / "diff_overlay.png"
    overlay.save(out_path)
    print(f"Saved overlay to {out_path}")
    print("Red = Reference (Missing in Gen)")
    print("Cyan = Generated (Extra/Moved in Gen)")
    print("Black = Perfect Match")

if __name__ == "__main__":
    main()
