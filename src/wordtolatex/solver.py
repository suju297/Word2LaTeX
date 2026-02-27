import argparse
import sys
import subprocess
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple, Dict, Any

from wordtolatex.ir import TableBlock
from wordtolatex.docx_parser.document import parse_docx
from wordtolatex.layout_ml import analyze_document
from wordtolatex.generator import generate_latex
import tempfile
import shutil

def get_page_count(pdf_path: str) -> int:
    """Get the number of pages in a PDF."""
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return 0

def compile_latex(tex_source: str, output_dir: Path, job_name: str) -> Path:
    """
    Compile LaTeX source to PDF using pdflatex.
    Returns the path to the generated PDF.
    """
    tex_file = output_dir / f"{job_name}.tex"
    pdf_file = output_dir / f"{job_name}.pdf"
    
    with open(tex_file, "w") as f:
        f.write(tex_source)
    
    # Check for xelatex availability
    if not shutil.which("xelatex"):
        print("Warning: xelatex not found. Skipping compilation.", file=sys.stderr)
        return None

    try:
        subprocess.run(
            ["xelatex", "-interaction=nonstopmode", f"{job_name}.tex"],
            cwd=output_dir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return pdf_file
    except subprocess.CalledProcessError as e:
        print(f"Compilation finished with errors (check log): {e}")
        # Even with error, PDF might exist
        return pdf_file if pdf_file.exists() else None

def find_optimal_layout(doc_ir, target_pages: int, output_dir: Path) -> str:
    """
    Core loop to find layout settings that result in the target page count.
    Returns the best LaTeX source string found.
    """
    current_squeeze = 0.9  # Initial guess
    best_tex = generate_latex(doc_ir, squeeze_factor=current_squeeze)
    min_page_diff = float('inf')
    
    # Check if we can even compile
    if not shutil.which("xelatex"):
        print("Warning: 'xelatex' not found. Cannot optimize layout dynamically.")
        return best_tex

    print(f"Optimizing layout to match {target_pages} pages...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        for attempt in range(1, 9): # Try 8 times
            print(f"   [Iter {attempt}] Squeeze: {current_squeeze:.3f}...")
            
            # Generate
            tex_source = generate_latex(doc_ir, squeeze_factor=current_squeeze)
            
            # Compile
            pdf_out = compile_latex(tex_source, temp_path, f"attempt_{attempt}")
            
            if not pdf_out:
                print("   -> Compilation failed. Keeping previous best.")
                break 
            
            generated_pages = get_page_count(pdf_out)
            diff = generated_pages - target_pages
            print(f"   -> Result: {generated_pages} pages (Diff: {diff})")
            
            if diff == 0:
                print("   -> Match found!")
                return tex_source
            
            # Record best
            if abs(diff) < min_page_diff:
                min_page_diff = abs(diff)
                best_tex = tex_source

            # Feedback
            if diff > 0:
                print("   -> Too long. Compressing.")
                current_squeeze -= 0.08
            else:
                print("   -> Too short. Expanding.")
                current_squeeze += 0.02
                
    print("   -> Optimization finished (limit reached). Using best result.")
    return best_tex

def solve_layout(docx_path: str, pdf_ref_path: str, output_tex_path: str):
    """CLI Wrapper for solver."""
    # 1. Parse
    print(f"Parsing DOCX: {docx_path}")
    doc_ir = parse_docx(docx_path)
    
    # 2. Analyze
    print(f"Analyzing Reference PDF: {pdf_ref_path}")
    layout_analysis = analyze_document(pdf_ref_path)
    from wordtolatex.ir import LayoutHints
    doc_ir.layout_hints = LayoutHints(
        body_font_size=layout_analysis.body_font_size,
        line_spacing=layout_analysis.line_spacing,
        margin_top=layout_analysis.margin_top,
        margin_bottom=layout_analysis.margin_bottom,
        margin_left=layout_analysis.margin_left,
        margin_right=layout_analysis.margin_right,
        page_width=layout_analysis.width / 72.0,
        page_height=layout_analysis.height / 72.0
    )
    
    target_pages = get_page_count(pdf_ref_path)
    if target_pages == 0:
        print("Error: Could not read page count from reference PDF.")
        return

    # 3. Optimize
    final_latex = find_optimal_layout(doc_ir, target_pages, Path("."))
    
    # 4. Save
    with open(output_tex_path, "w") as f:
        f.write(final_latex)
    print(f"Saved optimized LaTeX to: {output_tex_path}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python -m wordtolatex.solver <input.docx> <ref.pdf> <output.tex>")
        sys.exit(1)
    
    solve_layout(sys.argv[1], sys.argv[2], sys.argv[3])
