"""Feedback Loop Controller - Orchestrate vision-guided correction.

This module manages the iterative process of:
1. Generating PDF
2. Comparing with reference
3. Applying corrections
4. Re-generating until quality target met
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import time
import shutil

from wordtolatex.ir import Document
from wordtolatex.docx_parser.document import parse_docx
from wordtolatex.generator import generate_latex
from wordtolatex.utils import compile_latex
from wordtolatex.vision_diff import extract_vision_diffs, VisionDiffReport
from wordtolatex.corrections import CorrectionEngine

logger = logging.getLogger(__name__)


class FeedbackController:
    """Orchestrates the conversion-feedback loop."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.correction_engine = CorrectionEngine()
        self.history = []

    def run_loop(
        self,
        docx_path: Path,
        ref_pdf: Path,
        max_iterations: int = 3,
        target_match_rate: float = 0.95,
        squeeze_factor: float = 0.9
    ) -> bool:
        """Run feedback loop until target match rate or max iterations reached.
        
        Returns:
            bool: True if target match rate reached.
        """
        logger.info(f"Starting feedback loop for {docx_path.name}")
        
        # 1. Parse Initial OOXML
        doc = parse_docx(docx_path, self.output_dir)
        
        # Initial parameters
        current_squeeze = squeeze_factor
        
        for iteration in range(1, max_iterations + 1):
            logger.info(f"\n--- Iteration {iteration} ---")
            
            # Update dynamic parameters from IR hints (if modified by corrections)
            if doc.layout_hints.suggested_squeeze != 1.0:
                current_squeeze = squeeze_factor * doc.layout_hints.suggested_squeeze
            
            # 2. Generate LaTeX & Compile PDF
            tex_path = self.output_dir / "gen.tex"
            pdf_path = self.output_dir / "gen.pdf"
            
            # Clean up previous PDF to force new generation
            if pdf_path.exists():
                pdf_path.unlink()
                
            latex_content = generate_latex(doc, self.output_dir, squeeze_factor=current_squeeze)
            tex_path.write_text(latex_content, encoding="utf-8")
            
            # Compile (silence output unless error)
            compile_latex(tex_path, pdf_path)
            
            if not pdf_path.exists():
                logger.error("PDF generation failed.")
                return False
                
            # 3. Vision Comparison
            # Force no-cache for iterations > 1 to see effect of fixes
            use_cache = (iteration == 1) 
            report = extract_vision_diffs(ref_pdf, pdf_path, use_cache=use_cache)
            report.save(self.output_dir / f"diff_iter_{iteration}.json")
            
            logger.info(f"Match Rate: {report.match_rate:.1%}")
            self.history.append({
                "iteration": iteration,
                "match_rate": report.match_rate,
                "fixes": []
            })
            
            # Check success condition
            if report.match_rate >= target_match_rate:
                logger.info(f"Target match rate reached! ({report.match_rate:.1%} >= {target_match_rate:.1%})")
                return True
                
            if iteration >= max_iterations:
                logger.info("Max iterations reached.")
                break
                
            # 4. Apply Corrections
            fixes = self.correction_engine.apply_corrections(doc, report)
            
            if not fixes:
                logger.info("No further corrections identified.")
                break
                
            self.history[-1]["fixes"] = fixes
            for fix in fixes:
                logger.info(f"Applied fix: {fix}")
                
        return False

# CLI Integration
if __name__ == "__main__":
    import argparse
    import sys
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    parser = argparse.ArgumentParser(description="Run feedback loop conversion")
    parser.add_argument("docx_path", type=Path, help="Input DOCX")
    parser.add_argument("ref_pdf", type=Path, help="Reference PDF")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--iterations", type=int, default=3, help="Max iterations")
    
    args = parser.parse_args()
    
    controller = FeedbackController(args.output_dir)
    success = controller.run_loop(
        args.docx_path,
        args.ref_pdf,
        max_iterations=args.iterations
    )
    
    sys.exit(0 if success else 1)
