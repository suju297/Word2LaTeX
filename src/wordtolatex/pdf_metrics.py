"""PDF Metrics Extractor - Get font sizes, line heights, and spacing from PDFs."""

import fitz
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import json
import sys


@dataclass
class PDFMetrics:
    """Extracted metrics from a PDF."""
    avg_font_size: float
    avg_line_height: float
    avg_block_gap: float
    page_count: int
    
    def to_dict(self):
        return {
            "avg_font_size": self.avg_font_size,
            "avg_line_height": self.avg_line_height,
            "avg_block_gap": self.avg_block_gap,
            "page_count": self.page_count,
        }


def extract_metrics(pdf_path: Path) -> PDFMetrics:
    """Extract spacing metrics from a PDF."""
    doc = fitz.open(pdf_path)
    
    all_sizes = []
    line_heights = []
    block_gaps = []
    
    # Sample first 3 pages or all if fewer
    for page_num in range(min(3, len(doc))):
        page = doc[page_num]
        blocks = page.get_text('dict')['blocks']
        
        prev_block_bottom = None
        
        for block in blocks:
            if 'lines' not in block:
                continue
            
            # Block gap
            if prev_block_bottom is not None:
                gap = block['bbox'][1] - prev_block_bottom
                if 0 < gap < 100:
                    block_gaps.append(gap)
            
            prev_line_bottom = None
            for line in block['lines']:
                # Line height
                if prev_line_bottom is not None:
                    lh = line['bbox'][1] - prev_line_bottom
                    if 5 < lh < 30:
                        line_heights.append(lh)
                
                for span in line['spans']:
                    all_sizes.append(span['size'])
                
                prev_line_bottom = line['bbox'][3]
            
            prev_block_bottom = block['bbox'][3]
    
    page_count = len(doc)
    doc.close()
    
    return PDFMetrics(
        avg_font_size=sum(all_sizes)/len(all_sizes) if all_sizes else 0,
        avg_line_height=sum(line_heights)/len(line_heights) if line_heights else 0,
        avg_block_gap=sum(block_gaps)/len(block_gaps) if block_gaps else 0,
        page_count=page_count,
    )


def compare_pdfs(ref_path: Path, gen_path: Path) -> dict:
    """Compare metrics between reference and generated PDFs."""
    ref = extract_metrics(ref_path)
    gen = extract_metrics(gen_path)
    
    return {
        "reference": ref.to_dict(),
        "generated": gen.to_dict(),
        "ratios": {
            "font_size": gen.avg_font_size / ref.avg_font_size if ref.avg_font_size else 1,
            "line_height": gen.avg_line_height / ref.avg_line_height if ref.avg_line_height else 1,
            "block_gap": gen.avg_block_gap / ref.avg_block_gap if ref.avg_block_gap else 1,
            "page_count": gen.page_count / ref.page_count if ref.page_count else 1,
        },
        "corrections_needed": {
            "reduce_line_spacing_by": f"{(1 - ref.avg_line_height/gen.avg_line_height)*100:.0f}%" if gen.avg_line_height else "0%",
            "reduce_block_gap_by": f"{(1 - ref.avg_block_gap/gen.avg_block_gap)*100:.0f}%" if gen.avg_block_gap else "0%",
        }
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python pdf_metrics.py ref.pdf gen.pdf")
        sys.exit(1)
    
    ref_path = Path(sys.argv[1])
    gen_path = Path(sys.argv[2])
    
    result = compare_pdfs(ref_path, gen_path)
    print(json.dumps(result, indent=2))
