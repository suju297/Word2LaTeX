#!/usr/bin/env python3
"""Page-by-Page Visual Overlay Comparison.

Generates a visual comparison report showing ref vs gen PDFs side by side,
with overlays highlighting differences and detailed metrics per page.

Usage:
    uv run python scripts/page_overlay.py <doc_id>
    uv run python scripts/page_overlay.py ACM_SUBMISSION_TEMPLATE --save-overlays
    uv run python scripts/page_overlay.py Resume_1 --pages 1,2
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Optional SSIM support
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False


@dataclass
class PageComparison:
    """Comparison results for a single page."""
    page_num: int  # 1-indexed
    similarity: float
    method: str  # 'ssim' or 'mad'
    ref_present: bool = True
    gen_present: bool = True
    
    # Drift metrics
    content_top_ref: float = 0.0  # First non-white row (normalized)
    content_top_gen: float = 0.0
    vertical_drift_px: int = 0
    
    # Density metrics
    ref_chars_approx: int = 0
    gen_chars_approx: int = 0
    density_ratio: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "page": self.page_num,
            "similarity": round(self.similarity, 4),
            "method": self.method,
            "ref_present": self.ref_present,
            "gen_present": self.gen_present,
            "content_top_ref": round(self.content_top_ref, 3),
            "content_top_gen": round(self.content_top_gen, 3),
            "vertical_drift_px": self.vertical_drift_px,
            "density_ratio": round(self.density_ratio, 3),
        }


@dataclass
class ComparisonReport:
    """Full comparison report for two PDFs."""
    doc_id: str
    ref_pages: int
    gen_pages: int
    page_match: bool
    avg_similarity: float
    median_vertical_drift: float
    pages: List[PageComparison] = field(default_factory=list)
    
    # Summary verdict
    verdict: str = "UNKNOWN"
    
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "ref_pages": self.ref_pages,
            "gen_pages": self.gen_pages,
            "page_match": self.page_match,
            "avg_similarity": round(self.avg_similarity, 4),
            "median_vertical_drift_px": round(self.median_vertical_drift, 1),
            "verdict": self.verdict,
            "pages": [p.to_dict() for p in self.pages],
        }


def render_page_grayscale(doc: fitz.Document, page_idx: int, dpi: int = 150) -> np.ndarray:
    """Render a PDF page to grayscale numpy array."""
    page = doc[page_idx]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def find_content_top(gray: np.ndarray, threshold: float = 250.0) -> float:
    """Find the normalized position of first non-white row."""
    row_means = np.mean(gray, axis=1)
    content_rows = np.where(row_means < threshold)[0]
    if len(content_rows) == 0:
        return 1.0  # All white
    return content_rows[0] / gray.shape[0]


def extract_text_count(doc: fitz.Document, page_idx: int) -> int:
    """Extract approximate character count from a page."""
    page = doc[page_idx]
    text = page.get_text()
    return len(text.replace(" ", "").replace("\n", ""))


def compute_similarity(img1: np.ndarray, img2: np.ndarray) -> Tuple[float, str]:
    """Compute similarity between two grayscale images."""
    # Resize to same dimensions
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    
    if HAS_SSIM:
        win_size = min(7, min(h, w) // 2 * 2 + 1)  # Ensure odd and fits
        if win_size >= 3:
            score = ssim(img1, img2, win_size=win_size)
            return score, "ssim"
    
    # MAD fallback
    diff = np.abs(img1.astype(float) - img2.astype(float))
    mad = np.mean(diff)
    score = 1.0 - (mad / 255.0)
    return score, "mad"


def create_overlay_image(
    ref_gray: np.ndarray, 
    gen_gray: np.ndarray, 
    page_num: int
) -> Image.Image:
    """Create a side-by-side overlay comparison image."""
    # Resize to same height
    h = max(ref_gray.shape[0], gen_gray.shape[0])
    w_ref = ref_gray.shape[1]
    w_gen = gen_gray.shape[1]
    
    # Pad shorter image
    if ref_gray.shape[0] < h:
        pad = np.full((h - ref_gray.shape[0], w_ref), 255, dtype=np.uint8)
        ref_gray = np.vstack([ref_gray, pad])
    if gen_gray.shape[0] < h:
        pad = np.full((h - gen_gray.shape[0], w_gen), 255, dtype=np.uint8)
        gen_gray = np.vstack([gen_gray, pad])
    
    # Compute difference
    min_w = min(w_ref, w_gen)
    diff = np.abs(ref_gray[:, :min_w].astype(int) - gen_gray[:, :min_w].astype(int))
    diff_normalized = np.clip(diff * 3, 0, 255).astype(np.uint8)  # Enhance visibility
    
    # Create RGB images for each panel
    ref_rgb = np.stack([ref_gray, ref_gray, ref_gray], axis=-1)
    gen_rgb = np.stack([gen_gray, gen_gray, gen_gray], axis=-1)
    
    # Diff as red channel overlay
    diff_rgb = np.zeros((h, min_w, 3), dtype=np.uint8)
    diff_rgb[:, :, 0] = diff_normalized  # Red channel for differences
    diff_rgb[:, :, 1] = 255 - diff_normalized  # Green = similar
    diff_rgb[:, :, 2] = 255 - diff_normalized  # Blue = similar
    
    # Concatenate: Ref | Diff | Gen
    spacer = np.full((h, 10, 3), 200, dtype=np.uint8)  # Gray spacer
    combined = np.hstack([
        ref_rgb[:, :min_w, :],
        spacer,
        diff_rgb,
        spacer,
        gen_rgb[:, :min_w, :],
    ])
    
    # Add header
    header_height = 40
    header = np.full((header_height, combined.shape[1], 3), 255, dtype=np.uint8)
    combined = np.vstack([header, combined])
    
    # Convert to PIL for text
    img = Image.fromarray(combined)
    draw = ImageDraw.Draw(img)
    
    # Add labels
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    section_width = min_w
    draw.text((section_width // 2 - 50, 10), "REFERENCE", fill=(0, 100, 0), font=font)
    draw.text((section_width + 10 + section_width // 2 - 30, 10), "DIFF", fill=(180, 0, 0), font=font)
    draw.text((2 * section_width + 20 + section_width // 2 - 50, 10), "GENERATED", fill=(0, 0, 180), font=font)
    draw.text((10, 10), f"Page {page_num}", fill=(50, 50, 50), font=font)
    
    return img


def compare_documents(
    ref_pdf: Path,
    gen_pdf: Path,
    doc_id: str,
    dpi: int = 150,
    pages: Optional[List[int]] = None,
) -> ComparisonReport:
    """Compare two PDFs page by page."""
    ref_doc = fitz.open(ref_pdf)
    gen_doc = fitz.open(gen_pdf)
    
    ref_pages = len(ref_doc)
    gen_pages = len(gen_doc)
    max_pages = max(ref_pages, gen_pages)
    
    comparisons = []
    
    for page_idx in range(max_pages):
        page_num = page_idx + 1  # 1-indexed
        
        # Skip if not in requested pages
        if pages and page_num not in pages:
            continue
        
        ref_present = page_idx < ref_pages
        gen_present = page_idx < gen_pages
        
        if ref_present and gen_present:
            ref_gray = render_page_grayscale(ref_doc, page_idx, dpi)
            gen_gray = render_page_grayscale(gen_doc, page_idx, dpi)
            
            similarity, method = compute_similarity(ref_gray, gen_gray)
            
            # Content position
            content_top_ref = find_content_top(ref_gray)
            content_top_gen = find_content_top(gen_gray)
            vertical_drift_px = int((content_top_gen - content_top_ref) * ref_gray.shape[0])
            
            # Text density
            ref_chars = extract_text_count(ref_doc, page_idx)
            gen_chars = extract_text_count(gen_doc, page_idx)
            density_ratio = gen_chars / ref_chars if ref_chars > 0 else 1.0
            
            comp = PageComparison(
                page_num=page_num,
                similarity=similarity,
                method=method,
                ref_present=True,
                gen_present=True,
                content_top_ref=content_top_ref,
                content_top_gen=content_top_gen,
                vertical_drift_px=vertical_drift_px,
                ref_chars_approx=ref_chars,
                gen_chars_approx=gen_chars,
                density_ratio=density_ratio,
            )
        elif ref_present:
            comp = PageComparison(
                page_num=page_num,
                similarity=0.0,
                method="missing",
                ref_present=True,
                gen_present=False,
            )
        else:
            comp = PageComparison(
                page_num=page_num,
                similarity=0.0,
                method="extra",
                ref_present=False,
                gen_present=True,
            )
        
        comparisons.append(comp)
    
    ref_doc.close()
    gen_doc.close()
    
    # Compute summary stats
    valid_comparisons = [c for c in comparisons if c.ref_present and c.gen_present]
    avg_similarity = np.mean([c.similarity for c in valid_comparisons]) if valid_comparisons else 0.0
    drifts = [abs(c.vertical_drift_px) for c in valid_comparisons]
    median_drift = np.median(drifts) if drifts else 0.0
    
    # Verdict
    if ref_pages != gen_pages:
        verdict = "FAIL_PAGE_COUNT"
    elif avg_similarity >= 0.90:
        verdict = "EXCELLENT" if median_drift < 10 else "GOOD_SSIM_DRIFTED"
    elif avg_similarity >= 0.80:
        verdict = "ACCEPTABLE"
    elif avg_similarity >= 0.70:
        verdict = "MARGINAL"
    else:
        verdict = "POOR"
    
    return ComparisonReport(
        doc_id=doc_id,
        ref_pages=ref_pages,
        gen_pages=gen_pages,
        page_match=(ref_pages == gen_pages),
        avg_similarity=avg_similarity,
        median_vertical_drift=median_drift,
        pages=comparisons,
        verdict=verdict,
    )


def generate_overlays(
    ref_pdf: Path,
    gen_pdf: Path,
    output_dir: Path,
    dpi: int = 150,
    pages: Optional[List[int]] = None,
) -> List[Path]:
    """Generate overlay images for each page."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ref_doc = fitz.open(ref_pdf)
    gen_doc = fitz.open(gen_pdf)
    
    common_pages = min(len(ref_doc), len(gen_doc))
    output_paths = []
    
    for page_idx in range(common_pages):
        page_num = page_idx + 1
        if pages and page_num not in pages:
            continue
        
        ref_gray = render_page_grayscale(ref_doc, page_idx, dpi)
        gen_gray = render_page_grayscale(gen_doc, page_idx, dpi)
        
        overlay = create_overlay_image(ref_gray, gen_gray, page_num)
        out_path = output_dir / f"overlay_p{page_num}.png"
        overlay.save(out_path)
        output_paths.append(out_path)
    
    ref_doc.close()
    gen_doc.close()
    
    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Page-by-page visual comparison")
    parser.add_argument("doc_id", help="Document ID from corpus (e.g. ACM_SUBMISSION_TEMPLATE)")
    parser.add_argument("--corpus-root", default="tests/corpus", help="Corpus root directory")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI")
    parser.add_argument("--pages", help="Comma-separated page numbers to compare (e.g. 1,2,3)")
    parser.add_argument("--save-overlays", action="store_true", help="Save overlay images")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON path")
    
    args = parser.parse_args()
    
    corpus_root = Path(args.corpus_root)
    doc_dir = corpus_root / args.doc_id
    
    if not doc_dir.exists():
        print(f"Error: {doc_dir} not found")
        sys.exit(1)
    
    ref_pdf = doc_dir / "ref.pdf"
    gen_pdf = doc_dir / "gen.pdf"
    
    if not ref_pdf.exists():
        print(f"Error: {ref_pdf} not found")
        sys.exit(1)
    
    if not gen_pdf.exists():
        print(f"Error: {gen_pdf} not found")
        sys.exit(1)
    
    # Parse page list
    pages = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]
    
    print(f"Comparing: {args.doc_id}")
    print(f"  Reference: {ref_pdf}")
    print(f"  Generated: {gen_pdf}")
    print(f"  Method: {'SSIM' if HAS_SSIM else 'MAD'}")
    
    report = compare_documents(ref_pdf, gen_pdf, args.doc_id, args.dpi, pages)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"COMPARISON REPORT: {args.doc_id}")
    print(f"{'=' * 60}")
    print(f"Pages: ref={report.ref_pages}, gen={report.gen_pages} {'✅' if report.page_match else '❌'}")
    print(f"Average Similarity: {report.avg_similarity:.4f}")
    print(f"Median Vertical Drift: {report.median_vertical_drift:.1f}px")
    print(f"Verdict: {report.verdict}")
    print()
    
    # Per-page details
    print("Page Details:")
    for p in report.pages:
        if p.ref_present and p.gen_present:
            status = "✅" if p.similarity >= 0.85 else "⚠️" if p.similarity >= 0.70 else "❌"
            drift_note = f" (drift: {p.vertical_drift_px:+d}px)" if abs(p.vertical_drift_px) > 5 else ""
            density_note = f" [density: {p.density_ratio:.2f}]" if abs(p.density_ratio - 1.0) > 0.1 else ""
            print(f"  Page {p.page_num}: {p.similarity:.4f} {status}{drift_note}{density_note}")
        elif not p.gen_present:
            print(f"  Page {p.page_num}: MISSING in generated ❌")
        else:
            print(f"  Page {p.page_num}: EXTRA in generated ⚠️")
    
    # Save JSON
    output_path = args.output or (doc_dir / "page_comparison.json")
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nReport saved: {output_path}")
    
    # Generate overlays
    if args.save_overlays:
        overlay_dir = doc_dir / "page_overlays"
        print(f"\nGenerating overlay images...")
        paths = generate_overlays(ref_pdf, gen_pdf, overlay_dir, args.dpi, pages)
        print(f"  Saved {len(paths)} overlays to {overlay_dir}")
    
    # Return exit code based on verdict
    if report.verdict in ["EXCELLENT", "GOOD_SSIM_DRIFTED"]:
        return 0
    elif report.verdict == "ACCEPTABLE":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
