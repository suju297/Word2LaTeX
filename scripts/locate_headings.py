#!/usr/bin/env python3
"""Locate heading text boxes in gen.pdf vs Oracle title regions.

This debug script proves WHERE headings ended up in gen.pdf compared
to where the Oracle (Word PDF) expected them.
"""
import json
import sys
from pathlib import Path

import fitz  # PyMuPDF


def locate_headings(doc_dir: Path) -> dict:
    """Compare title text locations between ref.pdf and gen.pdf."""
    ref_pdf_path = doc_dir / "ref.pdf"
    gen_pdf_path = doc_dir / "gen.pdf"
    region_stats_path = doc_dir / "region_stats.json"
    
    if not ref_pdf_path.exists():
        return {"error": f"ref.pdf not found: {ref_pdf_path}"}
    if not gen_pdf_path.exists():
        return {"error": f"gen.pdf not found: {gen_pdf_path}"}
    if not region_stats_path.exists():
        return {"error": f"region_stats.json not found: {region_stats_path}"}
    
    # Load region stats
    with open(region_stats_path) as f:
        region_stats = json.load(f)
    
    # Open PDFs
    ref_doc = fitz.open(ref_pdf_path)
    gen_doc = fitz.open(gen_pdf_path)
    
    results = []
    
    # Process each title region
    for region in region_stats.get("regions", []):
        if region.get("label") != "title":
            continue
        
        region_id = region.get("region_id", "unknown")
        page_idx = region.get("page", 0)
        bbox_norm = region.get("bbox_norm", [0, 0, 1, 1])
        
        # Get page dimensions
        if page_idx >= len(ref_doc):
            continue
        ref_page = ref_doc[page_idx]
        page_rect = ref_page.rect  # (x0, y0, x1, y1)
        page_w, page_h = page_rect.width, page_rect.height
        
        # Convert normalized bbox to absolute
        ref_bbox = fitz.Rect(
            bbox_norm[0] * page_w,
            bbox_norm[1] * page_h,
            bbox_norm[2] * page_w,
            bbox_norm[3] * page_h
        )
        
        # Extract text from ref PDF in this region
        ref_text = ref_page.get_text("text", clip=ref_bbox).strip()
        # Get first line (usually the title)
        title_text = ref_text.split("\n")[0].strip() if ref_text else ""
        
        # Search for this text in gen.pdf
        gen_matches = []
        for gen_page_idx, gen_page in enumerate(gen_doc):
            matches = gen_page.search_for(title_text)
            for match in matches:
                gen_matches.append({
                    "page": gen_page_idx,
                    "bbox": [match.x0, match.y0, match.x1, match.y1],
                })
        
        # Find closest match by Y distance (on same page first)
        same_page_matches = [m for m in gen_matches if m["page"] == page_idx]
        other_page_matches = [m for m in gen_matches if m["page"] != page_idx]
        
        best_match = None
        dx, dy = None, None
        
        if same_page_matches:
            # Pick closest by Y
            ref_y_center = (ref_bbox.y0 + ref_bbox.y1) / 2
            same_page_matches.sort(key=lambda m: abs((m["bbox"][1] + m["bbox"][3]) / 2 - ref_y_center))
            best_match = same_page_matches[0]
            dx = best_match["bbox"][0] - ref_bbox.x0
            dy = best_match["bbox"][1] - ref_bbox.y0
        elif other_page_matches:
            best_match = other_page_matches[0]
            dx = best_match["bbox"][0] - ref_bbox.x0
            dy = "PAGE_MOVED"
        
        results.append({
            "region_id": region_id,
            "title_text": title_text[:50] + ("..." if len(title_text) > 50 else ""),
            "ref_bbox": [round(x, 1) for x in [ref_bbox.x0, ref_bbox.y0, ref_bbox.x1, ref_bbox.y1]],
            "gen_bbox": best_match["bbox"] if best_match else None,
            "found": best_match is not None,
            "same_page": best_match["page"] == page_idx if best_match else False,
            "dx": round(dx, 1) if isinstance(dx, float) else dx,
            "dy": round(dy, 1) if isinstance(dy, float) else dy,
            "all_matches": len(gen_matches),
            "ssim_from_stats": region.get("score"),
            "blank_gen": region.get("blank_gen"),
        })
    
    ref_doc.close()
    gen_doc.close()
    
    return {
        "doc_id": doc_dir.name,
        "title_regions": len(results),
        "found_count": sum(1 for r in results if r["found"]),
        "same_page_count": sum(1 for r in results if r.get("same_page")),
        "results": results,
    }


def main():
    if len(sys.argv) < 2:
        # Default to Resume_1
        doc_dir = Path("tests/corpus/Resume_1")
    else:
        doc_dir = Path(sys.argv[1])
    
    results = locate_headings(doc_dir)
    
    print(json.dumps(results, indent=2))
    
    # Summary
    print("\n=== Summary ===")
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Title regions: {results['title_regions']}")
    print(f"Found in gen.pdf: {results['found_count']}")
    print(f"On same page: {results['same_page_count']}")
    
    print("\n=== Per-Title Details ===")
    for r in results.get("results", []):
        status = "✓ FOUND" if r["found"] else "✗ MISSING"
        if r["found"] and not r["same_page"]:
            status = "⚠ WRONG_PAGE"
        
        print(f"{r['region_id']}: {status}")
        print(f"  Text: '{r['title_text']}'")
        print(f"  Ref bbox: {r['ref_bbox']}")
        print(f"  Gen bbox: {r['gen_bbox']}")
        if r["found"]:
            print(f"  Delta: dx={r['dx']}, dy={r['dy']}")
        print(f"  SSIM: {r['ssim_from_stats']:.3f}, blank_gen: {r['blank_gen']}")
        print()


if __name__ == "__main__":
    main()
