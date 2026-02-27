#!/usr/bin/env python3
"""Compare embedded images between ref.pdf and gen.pdf for a corpus document.

Usage: python scripts/compare_images.py <doc_id>
Example: python scripts/compare_images.py IEEE_TEMPLATE
"""

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF


def get_image_info(pdf_path: Path) -> dict:
    """Extract image count per page and total images from a PDF."""
    doc = fitz.open(pdf_path)
    result = {
        "page_count": doc.page_count,
        "images_per_page": [],
        "total_images": 0,
    }
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        images = page.get_images(full=True)
        result["images_per_page"].append(len(images))
        result["total_images"] += len(images)
    
    doc.close()
    return result


def compare_pdfs(ref_path: Path, gen_path: Path) -> dict:
    """Compare two PDFs and return mismatch report."""
    ref_info = get_image_info(ref_path)
    gen_info = get_image_info(gen_path)
    
    mismatches = []
    
    # Compare page by page
    max_pages = max(ref_info["page_count"], gen_info["page_count"])
    
    for page in range(max_pages):
        ref_count = ref_info["images_per_page"][page] if page < ref_info["page_count"] else 0
        gen_count = gen_info["images_per_page"][page] if page < gen_info["page_count"] else 0
        
        if ref_count != gen_count:
            mismatches.append({
                "page": page + 1,  # 1-indexed for display
                "ref_images": ref_count,
                "gen_images": gen_count,
                "diff": ref_count - gen_count,
            })
    
    return {
        "ref": ref_info,
        "gen": gen_info,
        "mismatches": mismatches,
        "total_diff": ref_info["total_images"] - gen_info["total_images"],
    }


def print_report(doc_id: str, report: dict):
    """Print a human-readable comparison report."""
    print(f"\n{'='*60}")
    print(f"Image Parity Report: {doc_id}")
    print(f"{'='*60}")
    
    print(f"\nReference PDF:")
    print(f"  Pages: {report['ref']['page_count']}")
    print(f"  Total Images: {report['ref']['total_images']}")
    print(f"  Per Page: {report['ref']['images_per_page']}")
    
    print(f"\nGenerated PDF:")
    print(f"  Pages: {report['gen']['page_count']}")
    print(f"  Total Images: {report['gen']['total_images']}")
    print(f"  Per Page: {report['gen']['images_per_page']}")
    
    if report["mismatches"]:
        print(f"\n⚠️  MISMATCHES FOUND ({len(report['mismatches'])} pages)")
        for m in report["mismatches"]:
            diff_str = f"+{-m['diff']}" if m['diff'] < 0 else f"-{m['diff']}"
            status = "MISSING" if m['diff'] > 0 else "EXTRA"
            print(f"  Page {m['page']}: Ref={m['ref_images']}, Gen={m['gen_images']} ({status} {abs(m['diff'])})")
        
        print(f"\n  Total Difference: {report['total_diff']} images")
        if report['total_diff'] > 0:
            print(f"  ⚠️  {report['total_diff']} image(s) MISSING in generated PDF")
        elif report['total_diff'] < 0:
            print(f"  ℹ️  {-report['total_diff']} extra image(s) in generated PDF")
    else:
        print(f"\n✅ PARITY: All images accounted for!")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare images in ref.pdf vs gen.pdf")
    parser.add_argument("doc_id", help="Document ID (e.g. IEEE_TEMPLATE)")
    parser.add_argument("--corpus-root", default="tests/corpus", help="Corpus root directory")
    args = parser.parse_args()
    
    corpus_root = Path(args.corpus_root)
    doc_dir = corpus_root / args.doc_id
    
    if not doc_dir.exists():
        print(f"Error: {doc_dir} not found")
        sys.exit(1)
    
    ref_path = doc_dir / "ref.pdf"
    gen_path = doc_dir / "gen.pdf"
    
    if not ref_path.exists():
        print(f"Error: {ref_path} not found")
        sys.exit(1)
    
    if not gen_path.exists():
        print(f"Error: {gen_path} not found")
        sys.exit(1)
    
    report = compare_pdfs(ref_path, gen_path)
    print_report(args.doc_id, report)
    
    # Exit with error code if mismatches found
    if report["mismatches"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
