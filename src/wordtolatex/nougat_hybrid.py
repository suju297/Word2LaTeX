"""Nougat Hybrid Pipeline - Merge OOXML content with Nougat structure.

This module provides the hybrid approach:
1. Parse OOXML (our standard path) → IR
2. Run Nougat on ref.pdf → structural template
3. Compare structures and apply corrections to IR
4. Generate LaTeX with Nougat-informed formatting
"""

from pathlib import Path
from typing import Optional, Dict, List
import logging
import json
import subprocess

from wordtolatex.ir import Document, HeadingBlock
from wordtolatex.docx_parser.document import parse_docx
from wordtolatex.generator import generate_latex
from wordtolatex.utils import compile_latex

logger = logging.getLogger(__name__)


def run_nougat(pdf_path: Path, output_dir: Path) -> Optional[Path]:
    """Run Nougat on a PDF and return path to .mmd output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = subprocess.run(
            ["uv", "run", "nougat", str(pdf_path), "-o", str(output_dir), "--no-skipping"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Find the output file
        mmd_files = list(output_dir.glob("*.mmd"))
        if mmd_files:
            return mmd_files[0]
        
        logger.warning(f"Nougat completed but no .mmd file found")
        return None
        
    except subprocess.TimeoutExpired:
        logger.error("Nougat timed out")
        return None
    except Exception as e:
        logger.error(f"Nougat failed: {e}")
        return None


def load_nougat_structure(mmd_path: Path) -> Dict:
    """Load structural analysis from Nougat output."""
    # Import locally to avoid circular deps
    import sys
    scripts_dir = Path(__file__).parent.parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from nougat_compare import extract_sections_from_markdown
    from nougat_params import extract_parameters
    
    text = mmd_path.read_text(encoding='utf-8')
    sections = extract_sections_from_markdown(text)
    params = extract_parameters(mmd_path)
    
    return {
        "sections": sections,
        "params": params,
        "heading_pattern": params.heading_pattern,
        "table_count": len(params.tables),
        "equation_count": len(params.equations)
    }


def apply_nougat_corrections(doc: Document, nougat_structure: Dict) -> List[str]:
    """Apply corrections to IR based on Nougat's detected structure."""
    corrections = []
    
    # 1. Heading level correction
    # Nougat uses # for H1, ## for H2, etc.
    # Map our heading blocks to match Nougat's levels
    nougat_sections = nougat_structure.get("sections", [])
    
    if nougat_sections:
        # Build title → level mapping from Nougat
        nougat_levels = {}
        for sec in nougat_sections:
            normalized = sec.title.lower().strip()
            # Remove leading numbers
            import re
            normalized = re.sub(r'^[\d.]+\s*', '', normalized)
            nougat_levels[normalized] = sec.level
        
        # Correct heading levels in our IR
        for block in doc.blocks:
            if isinstance(block, HeadingBlock):
                # Extract text from content spans
                block_text = ''.join(span.text for span in block.content if hasattr(span, 'text'))
                title_norm = block_text.lower().strip()
                title_norm = re.sub(r'^[\d.]+\s*', '', title_norm)
                
                if title_norm in nougat_levels:
                    expected_level = nougat_levels[title_norm]
                    current_level = block.level
                    
                    if current_level != expected_level:
                        block.level = expected_level
                        corrections.append(
                            f"Heading '{block_text[:30]}...' level: {current_level}→{expected_level}"
                        )
    
    # 2. Heading pattern (numbered sections)
    heading_pattern = nougat_structure.get("heading_pattern", "arabic")
    if heading_pattern == "arabic":
        doc.use_numbered_sections = True
        corrections.append("Set numbered sections: Arabic (1, 1.1, 1.1.1)")
    elif heading_pattern == "roman":
        doc.use_numbered_sections = True
        # Set specific format (would need template support)
        corrections.append("Set numbered sections: Roman (I, II, III)")
    
    # 3. Document-level hints from Nougat params
    params = nougat_structure.get("params")
    if params:
        if params.table_count > 0:
            corrections.append(f"Detected {params.table_count} tables in reference")
        if params.equation_count > 0:
            corrections.append(f"Detected {params.equation_count} equations in reference")
    
    return corrections


def hybrid_convert(
    docx_path: Path,
    ref_pdf_path: Path,
    output_dir: Path,
    use_nougat: bool = True
) -> Path:
    """Hybrid conversion: OOXML content + Nougat structure.
    
    Args:
        docx_path: Input DOCX file
        ref_pdf_path: Reference PDF (for Nougat analysis)
        output_dir: Output directory for results
        use_nougat: Whether to run Nougat analysis
        
    Returns:
        Path to generated .tex file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting hybrid conversion for {docx_path.name}")
    
    # Step 1: Parse OOXML
    logger.info("Step 1: Parsing OOXML...")
    doc = parse_docx(docx_path, output_dir)
    
    # Step 2: Run Nougat (if enabled)
    nougat_structure = None
    if use_nougat and ref_pdf_path.exists():
        logger.info("Step 2: Running Nougat on reference PDF...")
        nougat_dir = output_dir / "nougat_output"
        mmd_path = run_nougat(ref_pdf_path, nougat_dir)
        
        if mmd_path:
            logger.info(f"Nougat output: {mmd_path}")
            nougat_structure = load_nougat_structure(mmd_path)
    
    # Step 3: Apply Nougat corrections
    corrections = []
    if nougat_structure:
        logger.info("Step 3: Applying Nougat-guided corrections...")
        corrections = apply_nougat_corrections(doc, nougat_structure)
        for c in corrections:
            logger.info(f"  • {c}")
    
    # Step 4: Generate LaTeX
    logger.info("Step 4: Generating LaTeX...")
    tex_path = output_dir / "gen.tex"
    latex_content = generate_latex(doc, output_dir)
    tex_path.write_text(latex_content, encoding='utf-8')
    
    # Step 5: Compile PDF
    logger.info("Step 5: Compiling PDF...")
    pdf_path = output_dir / "gen.pdf"
    compile_latex(tex_path, pdf_path)
    
    # Save conversion report
    report = {
        "docx": str(docx_path),
        "ref_pdf": str(ref_pdf_path),
        "nougat_used": nougat_structure is not None,
        "corrections_applied": corrections,
        "output_tex": str(tex_path),
        "output_pdf": str(pdf_path) if pdf_path.exists() else None
    }
    (output_dir / "hybrid_report.json").write_text(json.dumps(report, indent=2))
    
    logger.info(f"Hybrid conversion complete: {tex_path}")
    return tex_path


if __name__ == "__main__":
    import argparse
    import sys
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Hybrid OOXML+Nougat conversion")
    parser.add_argument("docx", type=Path, help="Input DOCX file")
    parser.add_argument("ref_pdf", type=Path, help="Reference PDF")
    parser.add_argument("-o", "--output", type=Path, default=Path("hybrid_output"), help="Output directory")
    parser.add_argument("--no-nougat", action="store_true", help="Skip Nougat analysis")
    
    args = parser.parse_args()
    
    tex_path = hybrid_convert(
        args.docx,
        args.ref_pdf,
        args.output,
        use_nougat=not args.no_nougat
    )
    
    print(f"\nOutput: {tex_path}")
