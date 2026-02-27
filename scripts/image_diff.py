#!/usr/bin/env python
"""Image-based visual diff between ref.pdf and gen.pdf.

This module provides visual similarity metrics (SSIM or MAD fallback)
to complement text-based drift analysis.

Usage:
    python scripts/image_diff.py --ref ref.pdf --gen gen.pdf --out image_diff.json
"""

from pathlib import Path
import json
import argparse

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Optional: scikit-image for SSIM
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("Note: scikit-image not available. Using MAD fallback for similarity.")


def compute_mad_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute 1 - normalized Mean Absolute Difference (MAD).
    
    Returns value in [0, 1] where 1 = identical.
    """
    diff = np.abs(img1.astype(float) - img2.astype(float))
    mad = diff.mean() / 255.0
    return 1.0 - mad


def compute_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute similarity score between two grayscale images.
    
    Uses SSIM if available, else falls back to MAD.
    Returns value in [0, 1] where 1 = identical.
    """
    if HAS_SSIM:
        # SSIM returns value in [-1, 1], typically [0, 1] for natural images
        score = ssim(img1, img2, data_range=255)
        return float(score)
    else:
        return compute_mad_similarity(img1, img2)


def pdf_page_to_grayscale(doc: fitz.Document, page_idx: int, dpi: int = 150) -> np.ndarray:
    """Render a PDF page to grayscale numpy array."""
    page = doc[page_idx]
    pix = page.get_pixmap(dpi=dpi)
    
    # Convert to numpy array
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    
    # Convert to grayscale (average of RGB channels, ignore alpha if present)
    if pix.n >= 3:
        gray = np.mean(img[..., :3], axis=2).astype(np.uint8)
    else:
        gray = img[..., 0]
    
    return gray


def resize_to_common(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize both images to the minimum common dimensions."""
    min_h = min(img1.shape[0], img2.shape[0])
    min_w = min(img1.shape[1], img2.shape[1])
    
    # Using PIL for resizing
    pil1 = Image.fromarray(img1)
    pil2 = Image.fromarray(img2)
    
    resized1 = np.array(pil1.resize((min_w, min_h), Image.Resampling.LANCZOS))
    resized2 = np.array(pil2.resize((min_w, min_h), Image.Resampling.LANCZOS))
    
    return resized1, resized2


def compute_page_similarity(
    ref_pdf: Path, 
    gen_pdf: Path, 
    dpi: int = 150
) -> dict:
    """Compute per-page visual similarity between two PDFs.
    
    Returns dict with:
    - page_scores: list of {page, similarity}
    - avg_similarity: average across common pages
    - page_count_match: whether page counts match
    - ref_pages, gen_pages: page counts
    - method: 'ssim' or 'mad'
    """
    ref_doc = fitz.open(ref_pdf)
    gen_doc = fitz.open(gen_pdf)
    
    results = {
        "page_scores": [],
        "avg_similarity": 0.0,
        "page_count_match": len(ref_doc) == len(gen_doc),
        "ref_pages": len(ref_doc),
        "gen_pages": len(gen_doc),
        "method": "ssim" if HAS_SSIM else "mad",
        "dpi": dpi,
    }
    
    common_pages = min(len(ref_doc), len(gen_doc))
    
    if common_pages == 0:
        ref_doc.close()
        gen_doc.close()
        return results
    
    for page_idx in range(common_pages):
        ref_gray = pdf_page_to_grayscale(ref_doc, page_idx, dpi)
        gen_gray = pdf_page_to_grayscale(gen_doc, page_idx, dpi)
        
        # Resize to common dimensions
        ref_resized, gen_resized = resize_to_common(ref_gray, gen_gray)
        
        # Compute similarity
        score = compute_similarity(ref_resized, gen_resized)
        
        results["page_scores"].append({
            "page": page_idx,
            "similarity": round(score, 4),
            "ref_dims": f"{ref_gray.shape[1]}x{ref_gray.shape[0]}",
            "gen_dims": f"{gen_gray.shape[1]}x{gen_gray.shape[0]}",
        })
    
    if results["page_scores"]:
        results["avg_similarity"] = round(
            sum(p["similarity"] for p in results["page_scores"]) / len(results["page_scores"]),
            4
        )
    
    # Check for empty pages in generated PDF
    # We check all pages in gen_doc, not just common ones
    empty_pages = []
    for page_idx in range(len(gen_doc)):
        # Render at low DPI for speed
        gen_gray = pdf_page_to_grayscale(gen_doc, page_idx, dpi=72)
        mean_intensity = gen_gray.mean()
        
        # Check text content (to differentiate sparse text from empty)
        page_text = gen_doc[page_idx].get_text().strip()
        has_text = len(page_text) > 0
        
        # Is empty if:
        # 1. Very close to pure white (>254.5) regardless of text (invisible text?)
        # 2. Pretty white (>250) AND no text content
        if mean_intensity > 254.5:
             empty_pages.append(page_idx)
        elif mean_intensity > 250.0 and not has_text:
             empty_pages.append(page_idx)
            
    results["gen_empty_pages"] = empty_pages
    results["has_empty_gen_pages"] = len(empty_pages) > 0
    
    ref_doc.close()
    gen_doc.close()
    
    return results


def generate_diff_images(
    ref_pdf: Path,
    gen_pdf: Path,
    output_dir: Path,
    dpi: int = 150
) -> list[Path]:
    """Generate visual diff images showing differences.
    
    Returns list of output image paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ref_doc = fitz.open(ref_pdf)
    gen_doc = fitz.open(gen_pdf)
    
    output_paths = []
    common_pages = min(len(ref_doc), len(gen_doc))
    
    for page_idx in range(common_pages):
        ref_gray = pdf_page_to_grayscale(ref_doc, page_idx, dpi)
        gen_gray = pdf_page_to_grayscale(gen_doc, page_idx, dpi)
        
        # Resize to common dimensions
        ref_resized, gen_resized = resize_to_common(ref_gray, gen_gray)
        
        # Compute absolute difference
        diff = np.abs(ref_resized.astype(np.int16) - gen_resized.astype(np.int16))
        
        # Enhance visibility: scale diff to [0, 255]
        if diff.max() > 0:
            diff_enhanced = ((diff / diff.max()) * 255).astype(np.uint8)
        else:
            diff_enhanced = diff.astype(np.uint8)
        
        # Save diff image
        diff_path = output_dir / f"diff_p{page_idx + 1}.png"
        Image.fromarray(diff_enhanced).save(diff_path)
        output_paths.append(diff_path)
        
    ref_doc.close()
    gen_doc.close()
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Compute visual diff between PDFs")
    parser.add_argument("--ref", type=Path, required=True, help="Reference PDF")
    parser.add_argument("--gen", type=Path, required=True, help="Generated PDF")
    parser.add_argument("--out", type=Path, default=Path("image_diff.json"), help="Output JSON")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for rendering")
    parser.add_argument("--diff-images", action="store_true", help="Generate diff images")
    parser.add_argument("--diff-dir", type=Path, default=Path("diff_images"), help="Dir for diff images")
    
    args = parser.parse_args()
    
    if not args.ref.exists():
        print(f"Error: Reference PDF not found: {args.ref}")
        return 1
        
    if not args.gen.exists():
        print(f"Error: Generated PDF not found: {args.gen}")
        return 1
    
    print(f"Computing visual similarity...")
    print(f"  Reference: {args.ref}")
    print(f"  Generated: {args.gen}")
    print(f"  Method: {'SSIM' if HAS_SSIM else 'MAD (fallback)'}")
    print(f"  DPI: {args.dpi}")
    
    results = compute_page_similarity(args.ref, args.gen, args.dpi)
    
    print(f"\n--- RESULTS ---")
    print(f"Page count: ref={results['ref_pages']}, gen={results['gen_pages']} ({'MATCH' if results['page_count_match'] else 'MISMATCH'})")
    print(f"Average similarity: {results['avg_similarity']:.4f}")
    
    for page in results["page_scores"]:
        status = "✅" if page["similarity"] >= 0.85 else "⚠️" if page["similarity"] >= 0.70 else "❌"
        print(f"  Page {page['page'] + 1}: {page['similarity']:.4f} {status}")
    
    # Save results
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.out}")
    
    # Generate diff images if requested
    if args.diff_images:
        print(f"\nGenerating diff images...")
        paths = generate_diff_images(args.ref, args.gen, args.diff_dir, args.dpi)
        print(f"  Saved {len(paths)} diff images to {args.diff_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())
