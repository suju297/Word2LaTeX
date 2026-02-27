#!/usr/bin/env python3
"""Structural comparison of PDFs using DocTR layout detection.

Compares reference PDF (from Word) with generated PDF (from LaTeX)
at the text block level, computing match rates and position drift.

Usage:
    python scripts/structural_compare.py tests/corpus/IEEE_TEMPLATE
    python scripts/structural_compare.py --all
    python scripts/structural_compare.py tests/corpus/Resume_1 --visuals
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF for rendering overlays


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """A text block with bounding box and content."""
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1) normalized 0-1
    text: str
    page: int
    confidence: float = 1.0
    
    @property
    def area(self) -> float:
        """Area of the bounding box (normalized)."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])


@dataclass
class MatchedPair:
    """A matched pair of blocks from ref and gen."""
    ref_block: Block
    gen_block: Block
    iou: float
    text_sim: float
    dx: float  # Horizontal drift (gen - ref)
    dy: float  # Vertical drift (gen - ref)


@dataclass
class MatchResult:
    """Result of block matching."""
    matched_pairs: List[MatchedPair] = field(default_factory=list)
    unmatched_ref: List[Block] = field(default_factory=list)
    unmatched_gen: List[Block] = field(default_factory=list)
    
    @property
    def match_rate(self) -> float:
        """Percentage of ref blocks matched."""
        total_ref = len(self.matched_pairs) + len(self.unmatched_ref)
        if total_ref == 0:
            return 0.0
        return len(self.matched_pairs) / total_ref


@dataclass
class DriftStats:
    """Statistics about position drift."""
    median_dx: float = 0.0
    median_dy: float = 0.0
    max_dx: float = 0.0
    max_dy: float = 0.0
    iqr_dx: float = 0.0
    iqr_dy: float = 0.0
    outlier_count: int = 0


@dataclass
class ComparisonResult:
    """Full comparison result for a document."""
    doc_id: str
    ref_block_count: int
    gen_block_count: int
    match_rate: float
    drift: DriftStats
    match_result: Optional[MatchResult] = None


# ---------------------------------------------------------------------------
# DocTR extraction
# ---------------------------------------------------------------------------

def extract_blocks_doctr(pdf_path: Path, granularity: str = "line") -> List[Block]:
    """Extract text elements from PDF using DocTR.
    
    Args:
        pdf_path: Path to PDF file
        granularity: "block" for paragraph-level, "line" for line-level (more granular)
    """
    try:
        from doctr.io import DocumentFile
        from doctr.models import ocr_predictor
    except ImportError:
        print("ERROR: DocTR not installed. Run: pip install python-doctr[torch]")
        sys.exit(1)
    
    # Load PDF
    doc = DocumentFile.from_pdf(str(pdf_path))
    
    # Load model (cached after first call)
    model = ocr_predictor(pretrained=True)
    
    # Run OCR
    result = model(doc)
    
    blocks = []
    for page_idx, page in enumerate(result.pages):
        for block in page.blocks:
            if granularity == "line":
                # Extract each line as a separate block for more granular comparison
                for line in block.lines:
                    text = " ".join(w.value for w in line.words)
                    confidences = [w.confidence for w in line.words]
                    avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
                    
                    # line.geometry is ((x0, y0), (x1, y1)) normalized
                    bbox = (
                        line.geometry[0][0],
                        line.geometry[0][1],
                        line.geometry[1][0],
                        line.geometry[1][1],
                    )
                    
                    blocks.append(Block(
                        bbox=bbox,
                        text=text,
                        page=page_idx,
                        confidence=avg_conf
                    ))
            else:
                # Original block-level extraction
                text_parts = []
                confidences = []
                for line in block.lines:
                    line_text = " ".join(w.value for w in line.words)
                    text_parts.append(line_text)
                    confidences.extend(w.confidence for w in line.words)
                
                text = " ".join(text_parts)
                avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
                
                # block.geometry is ((x0, y0), (x1, y1)) normalized
                bbox = (
                    block.geometry[0][0],
                    block.geometry[0][1],
                    block.geometry[1][0],
                    block.geometry[1][1],
                )
                
                blocks.append(Block(
                    bbox=bbox,
                    text=text,
                    page=page_idx,
                    confidence=avg_conf
                ))
    
    return blocks


# ---------------------------------------------------------------------------
# Block matching
# ---------------------------------------------------------------------------

def iou(bbox1: Tuple[float, ...], bbox2: Tuple[float, ...]) -> float:
    """Compute Intersection over Union for two bboxes."""
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    
    if x1 <= x0 or y1 <= y0:
        return 0.0
    
    inter = (x1 - x0) * (y1 - y0)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def text_similarity(t1: str, t2: str) -> float:
    """Compute normalized text similarity (Jaccard on words)."""
    words1 = set(t1.lower().split())
    words2 = set(t2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def match_blocks(
    ref_blocks: List[Block],
    gen_blocks: List[Block],
    iou_threshold: float = 0.2,
    text_threshold: float = 0.3,
    cross_page: bool = False,
) -> MatchResult:
    """Match blocks between ref and gen based on position and text overlap.
    
    Args:
        cross_page: If True, allow matching blocks across different pages
                    (useful when pagination differs between ref and gen)
    """
    result = MatchResult()
    gen_used = set()
    
    for ref_block in ref_blocks:
        best_match = None
        best_score = 0.0
        best_gen_idx = -1
        
        for gen_idx, gen_block in enumerate(gen_blocks):
            if gen_idx in gen_used:
                continue
            
            # Page matching logic
            if not cross_page and gen_block.page != ref_block.page:
                continue
            
            # If cross_page, prefer same page but allow ±1 page
            page_penalty = 0.0
            if cross_page and gen_block.page != ref_block.page:
                page_diff = abs(gen_block.page - ref_block.page)
                if page_diff > 2:  # Don't match if more than 2 pages apart
                    continue
                page_penalty = 0.1 * page_diff  # Small penalty for different pages
            
            # Compute match score
            iou_score = iou(ref_block.bbox, gen_block.bbox)
            text_score = text_similarity(ref_block.text, gen_block.text)
            
            # Combined score (weighted average) with page penalty
            combined = 0.4 * iou_score + 0.6 * text_score - page_penalty
            
            if combined > best_score and (iou_score >= iou_threshold or text_score >= text_threshold):
                best_score = combined
                best_match = gen_block
                best_gen_idx = gen_idx
        
        if best_match:
            gen_used.add(best_gen_idx)
            
            # Compute drift (center-to-center)
            ref_cx = (ref_block.bbox[0] + ref_block.bbox[2]) / 2
            ref_cy = (ref_block.bbox[1] + ref_block.bbox[3]) / 2
            gen_cx = (best_match.bbox[0] + best_match.bbox[2]) / 2
            gen_cy = (best_match.bbox[1] + best_match.bbox[3]) / 2
            
            result.matched_pairs.append(MatchedPair(
                ref_block=ref_block,
                gen_block=best_match,
                iou=iou(ref_block.bbox, best_match.bbox),
                text_sim=text_similarity(ref_block.text, best_match.text),
                dx=gen_cx - ref_cx,
                dy=gen_cy - ref_cy,
            ))
        else:
            result.unmatched_ref.append(ref_block)
    
    # Find unmatched gen blocks
    for gen_idx, gen_block in enumerate(gen_blocks):
        if gen_idx not in gen_used:
            result.unmatched_gen.append(gen_block)
    
    return result


# ---------------------------------------------------------------------------
# Drift statistics
# ---------------------------------------------------------------------------

def compute_drift_stats(matched_pairs: List[MatchedPair]) -> DriftStats:
    """Compute statistics about position drift."""
    if not matched_pairs:
        return DriftStats()
    
    import numpy as np
    
    dxs = np.array([p.dx for p in matched_pairs])
    dys = np.array([p.dy for p in matched_pairs])
    
    # Convert to points (assuming 612pt width, 792pt height for letter)
    # Normalized coords * page_size -> points
    # We'll use normalized values directly for now (multiply by ~600 for approx points)
    scale = 600  # Approximate conversion to points
    
    stats = DriftStats(
        median_dx=float(np.median(dxs)) * scale,
        median_dy=float(np.median(dys)) * scale,
        max_dx=float(np.max(np.abs(dxs))) * scale,
        max_dy=float(np.max(np.abs(dys))) * scale,
        iqr_dx=float(np.percentile(np.abs(dxs), 75) - np.percentile(np.abs(dxs), 25)) * scale,
        iqr_dy=float(np.percentile(np.abs(dys), 75) - np.percentile(np.abs(dys), 25)) * scale,
    )
    
    # Count outliers (drift > 2x IQR from median)
    outlier_threshold_x = stats.iqr_dx * 2
    outlier_threshold_y = stats.iqr_dy * 2
    
    outliers = 0
    for p in matched_pairs:
        if abs(p.dx * scale - stats.median_dx) > outlier_threshold_x:
            outliers += 1
        elif abs(p.dy * scale - stats.median_dy) > outlier_threshold_y:
            outliers += 1
    
    stats.outlier_count = outliers
    return stats


# ---------------------------------------------------------------------------
# Visual overlay generation
# ---------------------------------------------------------------------------

def generate_overlay(
    ref_path: Path,
    gen_path: Path,
    match_result: MatchResult,
    output_dir: Path
):
    """Generate visual overlay images comparing ref and gen."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ref_doc = fitz.open(str(ref_path))
    gen_doc = fitz.open(str(gen_path))
    
    # Colors
    COLOR_MATCHED = (0, 0.8, 0)    # Green
    COLOR_UNMATCHED = (1, 0, 0)    # Red
    COLOR_EXTRA = (1, 0.5, 0)      # Orange
    
    max_pages = max(len(ref_doc), len(gen_doc))
    
    for page_idx in range(max_pages):
        # Create side-by-side image
        ref_pix = ref_doc[page_idx].get_pixmap(dpi=100) if page_idx < len(ref_doc) else None
        gen_pix = gen_doc[page_idx].get_pixmap(dpi=100) if page_idx < len(gen_doc) else None
        
        if ref_pix is None or gen_pix is None:
            continue
        
        # Get page dimensions for coordinate conversion
        ref_page = ref_doc[page_idx]
        gen_page = gen_doc[page_idx]
        ref_w, ref_h = ref_page.rect.width, ref_page.rect.height
        gen_w, gen_h = gen_page.rect.width, gen_page.rect.height
        
        # Draw matched blocks on ref (green)
        for pair in match_result.matched_pairs:
            if pair.ref_block.page != page_idx:
                continue
            bbox = pair.ref_block.bbox
            rect = fitz.Rect(
                bbox[0] * ref_w, bbox[1] * ref_h,
                bbox[2] * ref_w, bbox[3] * ref_h
            )
            ref_page.draw_rect(rect, color=COLOR_MATCHED, width=2)
        
        # Draw unmatched ref blocks (red)
        for block in match_result.unmatched_ref:
            if block.page != page_idx:
                continue
            bbox = block.bbox
            rect = fitz.Rect(
                bbox[0] * ref_w, bbox[1] * ref_h,
                bbox[2] * ref_w, bbox[3] * ref_h
            )
            ref_page.draw_rect(rect, color=COLOR_UNMATCHED, width=2)
        
        # Draw matched blocks on gen (green)
        for pair in match_result.matched_pairs:
            if pair.gen_block.page != page_idx:
                continue
            bbox = pair.gen_block.bbox
            rect = fitz.Rect(
                bbox[0] * gen_w, bbox[1] * gen_h,
                bbox[2] * gen_w, bbox[3] * gen_h
            )
            gen_page.draw_rect(rect, color=COLOR_MATCHED, width=2)
        
        # Draw unmatched gen blocks (orange)
        for block in match_result.unmatched_gen:
            if block.page != page_idx:
                continue
            bbox = block.bbox
            rect = fitz.Rect(
                bbox[0] * gen_w, bbox[1] * gen_h,
                bbox[2] * gen_w, bbox[3] * gen_h
            )
            gen_page.draw_rect(rect, color=COLOR_EXTRA, width=2)
        
        # Re-render with annotations
        ref_pix = ref_page.get_pixmap(dpi=100)
        gen_pix = gen_page.get_pixmap(dpi=100)
        
        # Save individual pages
        ref_pix.save(str(output_dir / f"ref_p{page_idx+1}.png"))
        gen_pix.save(str(output_dir / f"gen_p{page_idx+1}.png"))
        
        print(f"  Saved overlay for page {page_idx + 1}")
    
    ref_doc.close()
    gen_doc.close()


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare_document(doc_dir: Path, generate_visuals: bool = False, cross_page: bool = False) -> ComparisonResult:
    """Compare ref.pdf and gen.pdf in a document directory."""
    doc_id = doc_dir.name
    ref_path = doc_dir / "ref.pdf"
    gen_path = doc_dir / "gen.pdf"
    
    if not ref_path.exists():
        print(f"ERROR: {ref_path} not found")
        return ComparisonResult(doc_id=doc_id, ref_block_count=0, gen_block_count=0, 
                                match_rate=0.0, drift=DriftStats())
    
    if not gen_path.exists() or gen_path.stat().st_size < 1000:
        print(f"WARNING: {gen_path} missing or invalid")
        return ComparisonResult(doc_id=doc_id, ref_block_count=0, gen_block_count=0,
                                match_rate=0.0, drift=DriftStats())
    
    print(f"\n{'='*60}")
    print(f"Comparing: {doc_id}")
    print(f"{'='*60}")
    
    # Extract blocks
    print("Extracting blocks from ref.pdf...")
    start = time.time()
    ref_blocks = extract_blocks_doctr(ref_path)
    print(f"  Found {len(ref_blocks)} blocks in {time.time() - start:.1f}s")
    
    print("Extracting blocks from gen.pdf...")
    start = time.time()
    gen_blocks = extract_blocks_doctr(gen_path)
    print(f"  Found {len(gen_blocks)} blocks in {time.time() - start:.1f}s")
    
    # Match blocks
    mode = "cross-page" if cross_page else "same-page"
    print(f"Matching blocks ({mode})...")
    match_result = match_blocks(ref_blocks, gen_blocks, cross_page=cross_page)
    
    print(f"  Matched: {len(match_result.matched_pairs)}")
    print(f"  Unmatched ref: {len(match_result.unmatched_ref)}")
    print(f"  Unmatched gen: {len(match_result.unmatched_gen)}")
    print(f"  Match rate: {match_result.match_rate:.1%}")
    
    # Compute drift
    drift = compute_drift_stats(match_result.matched_pairs)
    print(f"\nDrift stats (in points):")
    print(f"  Median: dx={drift.median_dx:.1f}pt, dy={drift.median_dy:.1f}pt")
    print(f"  Max: dx={drift.max_dx:.1f}pt, dy={drift.max_dy:.1f}pt")
    print(f"  Outliers: {drift.outlier_count}")
    
    # Generate visuals if requested
    if generate_visuals:
        vis_dir = doc_dir / "structural_overlay"
        print(f"\nGenerating visual overlays in {vis_dir}...")
        generate_overlay(ref_path, gen_path, match_result, vis_dir)
    
    # Save results
    result = ComparisonResult(
        doc_id=doc_id,
        ref_block_count=len(ref_blocks),
        gen_block_count=len(gen_blocks),
        match_rate=match_result.match_rate,
        drift=drift,
        match_result=match_result,
    )
    
    # Save JSON report
    report = {
        "doc_id": doc_id,
        "ref_blocks": len(ref_blocks),
        "gen_blocks": len(gen_blocks),
        "matched": len(match_result.matched_pairs),
        "match_rate": match_result.match_rate,
        "drift": {
            "median_dx": drift.median_dx,
            "median_dy": drift.median_dy,
            "max_dx": drift.max_dx,
            "max_dy": drift.max_dy,
            "outliers": drift.outlier_count,
        },
        "unmatched_ref_sample": [
            {"text": b.text[:50], "page": b.page}
            for b in match_result.unmatched_ref[:5]
        ],
    }
    
    report_path = doc_dir / "structural_compare.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    
    return result


def print_summary(results: List[ComparisonResult]):
    """Print summary table of all comparisons."""
    print("\n" + "="*70)
    print("STRUCTURAL COMPARISON SUMMARY")
    print("="*70)
    
    print("\n{:<25} {:>8} {:>8} {:>8} {:>10} {:>10}".format(
        "Document", "Ref", "Gen", "Match%", "Drift-Y", "Status"))
    print("-"*70)
    
    for r in results:
        status = "✅" if r.match_rate >= 0.8 else "⚠️" if r.match_rate >= 0.5 else "❌"
        drift_y = f"{r.drift.median_dy:.1f}pt" if r.match_rate > 0 else "N/A"
        
        print("{:<25} {:>8} {:>8} {:>7.0%} {:>10} {:>10}".format(
            r.doc_id[:25],
            r.ref_block_count,
            r.gen_block_count,
            r.match_rate,
            drift_y,
            status
        ))
    
    # Summary
    good = sum(1 for r in results if r.match_rate >= 0.8)
    warn = sum(1 for r in results if 0.5 <= r.match_rate < 0.8)
    bad = sum(1 for r in results if r.match_rate < 0.5)
    
    print("-"*70)
    print(f"Summary: {good} good (≥80%), {warn} warning (50-80%), {bad} poor (<50%)")


def main():
    parser = argparse.ArgumentParser(
        description="Structural PDF comparison using DocTR"
    )
    parser.add_argument(
        "doc_dir", 
        type=Path, 
        nargs="?",
        help="Document directory with ref.pdf and gen.pdf"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Compare all corpus documents"
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("tests/corpus"),
        help="Corpus root directory"
    )
    parser.add_argument(
        "--visuals",
        action="store_true",
        help="Generate visual overlay images"
    )
    parser.add_argument(
        "--cross-page",
        action="store_true",
        help="Allow matching blocks across pages (for pagination drift)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        doc_dirs = sorted([
            d for d in args.corpus_root.iterdir()
            if d.is_dir() and (d / "ref.pdf").exists()
        ])
    elif args.doc_dir:
        doc_dirs = [args.doc_dir]
    else:
        print("Usage: structural_compare.py <doc_dir> or --all")
        sys.exit(1)
    
    results = []
    for doc_dir in doc_dirs:
        result = compare_document(doc_dir, generate_visuals=args.visuals, cross_page=args.cross_page)
        results.append(result)
    
    print_summary(results)


if __name__ == "__main__":
    main()
