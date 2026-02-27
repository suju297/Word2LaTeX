"""Correction Engine - Apply fixes to IR based on vision diffs.

This module applies rule-based corrections to the document IR to address
discrepancies identified by the vision diff extractor.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
import logging

from wordtolatex.ir import Document, HeadingBlock, ParagraphBlock, Block, FigureBlock
from wordtolatex.vision_diff import VisionDiffReport, RegionDiff, DiffType
from wordtolatex.block_mapper import map_diffs_to_blocks
from wordtolatex.pdf_cropper import crop_pdf_region
from wordtolatex.pdf_metrics import compare_pdfs
from pathlib import Path
import uuid

logger = logging.getLogger(__name__)


class CorrectionEngine:
    """Applies rule-based corrections to document IR."""
    
    def __init__(self):
        self.applied_fixes: List[str] = []

    def apply_corrections(self, doc: Document, report: VisionDiffReport) -> List[str]:
        """Apply all applicable corrections to the document.
        
        Args:
            doc: The document IR to modify.
            report: The vision diff report.
            
        Returns:
            List of descriptions of applied fixes.
        """
        self.applied_fixes = []
        
        # 1. Global Checks
        self._check_section_numbering(doc, report)
        self._check_pagination(doc, report)
        self._check_two_column(doc, report)
        self._check_missing_semantic_elements(doc, report)
        self._check_pdf_metrics(doc, report)  # NEW: Direct PDF metrics comparison
        
        # 2. Local Checks (Region-specific)
        self._apply_local_fixes(doc, report)
        
        return self.applied_fixes

    def _check_section_numbering(self, doc: Document, report: VisionDiffReport) -> None:
        """Rule: Enable numbered sections if missing from headings."""
        if doc.use_numbered_sections:
            return  # Already enabled
            
        # Find headings that are missing in Gen but present in Ref with numbering
        missing_numbered_headings = [
            d for d in report.diffs
            if d.diff_type == DiffType.MISSING
            and d.has_numbering
            and d.is_heading_like
            and d.confidence == 1.0
        ]
        
        if len(missing_numbered_headings) >= 2:
            doc.use_numbered_sections = True
            msg = f"Enabled numbered sections (found {len(missing_numbered_headings)} missing numbered headings)"
            self.applied_fixes.append(msg)
            logger.info(msg)

    def _check_pagination(self, doc: Document, report: VisionDiffReport) -> None:
        """Rule: Adjust spacing proportionally to match page count."""
        ref_pages = report.ref_page_count
        gen_pages = report.gen_page_count
        
        if ref_pages == gen_pages or ref_pages == 0:
            return
        
        # Calculate inflation/deflation ratio
        inflation = gen_pages / ref_pages
        
        # Initialize squeeze factor if needed
        current_squeeze = getattr(doc.layout_hints, "suggested_squeeze", 1.0)
        
        if inflation > 1.05:  # Gen has >5% more pages
            # Calculate correction factor (dampened to avoid overcorrection)
            # If inflation=1.46 (46% more pages), we need ~30% reduction
            correction_factor = 1 / (inflation ** 0.7)  # Dampened inverse
            new_squeeze = current_squeeze * correction_factor
            
            # Clamp to reasonable bounds
            new_squeeze = max(0.6, min(1.0, new_squeeze))
            
            doc.layout_hints.suggested_squeeze = new_squeeze
            
            # Also reduce line spacing directly
            current_line = getattr(doc.layout_hints, "line_spacing", 1.15)
            new_line = max(0.7, current_line * correction_factor)  # Allow tighter spacing
            doc.layout_hints.line_spacing = new_line
            
            msg = f"Page inflation ({gen_pages}/{ref_pages}={inflation:.2f}x): squeeze={new_squeeze:.2f}, line_spacing={new_line:.2f}"
            self.applied_fixes.append(msg)
            logger.info(msg)
            
        elif inflation < 0.95:  # Gen has <95% of ref pages
            # Expand spacing slightly
            correction_factor = 1 / (inflation ** 0.5)
            new_squeeze = min(1.2, current_squeeze * correction_factor)
            
            doc.layout_hints.suggested_squeeze = new_squeeze
            
            msg = f"Page deflation ({gen_pages}/{ref_pages}={inflation:.2f}x): squeeze={new_squeeze:.2f}"
            self.applied_fixes.append(msg)

    def _check_two_column(self, doc: Document, report: VisionDiffReport) -> None:
         """Rule: Enable two-column if vision detects 2 columns in Ref."""
         # Use detected metrics from report
         if getattr(report, "detected_ref_columns", 1) == 2:
             # Check if we are already in 2-column mode
             current_cols = 1
             if doc.sections:
                 current_cols = doc.sections[0].col_count
            
             # If IR thinks 1-column, but Vision sees 2-column -> Force Correction
             if current_cols == 1:
                 # Update Layout Hint
                 if hasattr(doc.layout_hints, "suggested_columns"):
                     doc.layout_hints.suggested_columns = 2
                
                 # Update Section 0 (Primary)
                 if doc.sections:
                     doc.sections[0].col_count = 2
                     # Set default spacing if missing/zero
                     if doc.sections[0].col_spacing < 1:
                         doc.sections[0].col_spacing = 24.0 # 0.33 inches (standard academic)
                 
                 msg = "Layout Mismatch: Forced 2-column mode based on Reference PDF vision analysis"
                 self.applied_fixes.append(msg)
                 logger.info(msg)

    def _check_pdf_metrics(self, doc: Document, report: VisionDiffReport) -> None:
        """Rule: Use PDF metrics to calculate precise font/spacing corrections."""
        try:
            # Get paths from report
            ref_path = Path(report.ref_path)
            gen_path = Path(report.gen_path)
            
            if not ref_path.exists() or not gen_path.exists():
                return
                
            # Compare metrics
            metrics = compare_pdfs(ref_path, gen_path)
            
            font_ratio = metrics["ratios"]["font_size"]
            gap_ratio = metrics["ratios"]["block_gap"]
            
            # If font is too large (>5% larger), suggest smaller base size
            if font_ratio > 1.05:
                # Calculate target font size
                current_font = getattr(doc.layout_hints, "body_font_size", 10.0)
                target_font = current_font / font_ratio
                target_font = max(8.0, min(12.0, target_font))  # Clamp 8-12pt
                
                doc.layout_hints.suggested_font_size = target_font
                
                msg = f"PDF Metrics: Font {font_ratio:.2f}x too large, suggest {target_font:.1f}pt"
                self.applied_fixes.append(msg)
                logger.info(msg)
            
            # If block gap is too large (>50% larger), reduce section spacing
            if gap_ratio > 1.5:
                ref_gap = metrics["reference"]["avg_block_gap"]
                gen_gap = metrics["generated"]["avg_block_gap"]
                
                # Reduction needed (as percentage to apply to current spacing)
                reduction = ref_gap / gen_gap  # e.g., 9.6 / 23.6 = 0.41
                
                # Set suggested section spacing multiplier  
                doc.layout_hints.suggested_section_spacing = reduction
                
                msg = f"PDF Metrics: Block gap {gap_ratio:.2f}x too large, reduce spacing by {(1-reduction)*100:.0f}%"
                self.applied_fixes.append(msg)
                logger.info(msg)
                
        except Exception as e:
            logger.warning(f"PDF metrics check failed: {e}")

    def _check_missing_semantic_elements(self, doc: Document, report: VisionDiffReport) -> None:
        """Rule: Identify missing semantic elements (Tables, Figures) and apply fallback."""
        count = 0
        for d in report.diffs:
            if d.diff_type == DiffType.MISSING and getattr(d, "ref_block_type", "Text") in ["Table", "Figure"]:
                 # Apply Image Fallback
                 self._apply_image_fallback(doc, report, d)
                 count += 1
                 
        if count > 0:
            logger.info(f"applied image fallback for {count} missing elements")

    def _apply_image_fallback(self, doc: Document, report: VisionDiffReport, diff: RegionDiff) -> None:
        """Extract missing region as image and insert into document."""
        try:
             # Determine output path
             filename = f"fallback_{uuid.uuid4().hex[:8]}.png"
             # Output dir relative to gen.tex (report.gen_path)
             output_dir = Path(report.gen_path).parent / "media"
             output_path = output_dir / filename
             
             # Helper to get list from BBox
             bbox = [diff.ref_bbox.x0, diff.ref_bbox.y0, diff.ref_bbox.x1, diff.ref_bbox.y1]
             
             # Extract
             success = crop_pdf_region(
                 Path(report.ref_path), 
                 diff.ref_page, 
                 bbox, 
                 output_path
             )
             
             if success:
                 # Create Block
                 fallback_block = FigureBlock(
                     id=f"fallback-{uuid.uuid4().hex[:8]}",
                     block_type="figure",
                     caption=f"Recovered {diff.ref_block_type} (Page {diff.ref_page + 1})",
                     image_path=f"media/{filename}",
                     width=6.0, # Default width
                     render_as_image=True,
                     policy_reason=f"Missing {getattr(diff, 'ref_block_type', 'Element')} recovered from Ref Page {diff.ref_page+1}"
                 )
                 
                 # Insert (Append)
                 doc.blocks.append(fallback_block)
                 
                 msg = f"Recovered missing {getattr(diff, 'ref_block_type', 'Element')} as image: {filename}"
                 self.applied_fixes.append(msg)
                 logger.info(msg)
                 
        except Exception as e:
             logger.error(f"Image fallback failed: {e}")

    def _apply_local_fixes(self, doc: Document, report: VisionDiffReport) -> None:
        """Apply fixes specific to mapped blocks."""
        
        # Filter for relevant diffs
        # 1. Missing headings (that aren't covered by global numbering rule)
        # 2. Shifted blocks (Vertical drift)
        
        # Identify shifted blocks
        shifted_diffs = [
            d for d in report.diffs 
            if d.diff_type == DiffType.SHIFTED 
            and d.position_drift is not None
            and abs(d.position_drift[1]) > 0.05 # >5% vertical drift
        ]
        
        if not shifted_diffs:
            return
            
        # Map to blocks
        mapped = map_diffs_to_blocks(doc, shifted_diffs)
        
        for diff, block in mapped:
            if not block:
                continue
                
            dy = diff.position_drift[1] # Positive = Gen is lower than Ref
            
            # Correction: 
            # If Gen is lower (dy > 0), we need to reduce spacing before
            # If Gen is higher (dy < 0), we need to increase spacing before
            
            # Convert normalized dy to points (approx page height 792pt)
            dy_pt = dy * 792
            
            if isinstance(block, (ParagraphBlock, HeadingBlock)):
                # Adjust spacing_before
                current = block.spacing_before or 0
                
                # Apply correction (dampened)
                # If Gen is lower (dy>0), we want to Move Up -> Decrease spacing
                # Correction = -dy
                correction = -dy_pt * 0.5 # Dampening factor
                
                # Boost correction for headings/titles as they are visually critical
                if getattr(diff, "ref_block_type", "Text") in ("Title", "Heading"):
                    correction *= 1.5
                
                new_spacing = max(0, current + correction)
                
                if abs(new_spacing - current) > 1: # Only if meaningful change
                    block.spacing_before = int(new_spacing)
                    self.applied_fixes.append(f"Adjusted spacing_before for {block.id}: {current} -> {int(new_spacing)} (dy={dy_pt:.1f}pt)")

