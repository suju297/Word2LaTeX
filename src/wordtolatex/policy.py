
"""Policy Engine - Decide semantic LaTeX vs fallback for each block."""

from typing import Literal, List, Dict, Optional, Tuple
from dataclasses import dataclass

from wordtolatex.ir import Block, TableBlock, FigureBlock
from wordtolatex.layout_oracle import LayoutRegion


@dataclass
class LayoutContext:
    """Contextual layout information for a specific block decision.
    
    Attributes:
        page_index: The 0-based page index where the block starts.
        column_count: Detected column count for this section/page.
        overlapping_regions: List of oracle regions that overlap with this block.
                             Each region carries its own label, confidence, and source.
        block_bbox_norm: The normalized bounding box (x0, y0, x1, y1) [0..1] of the block.
    """
    page_index: int
    column_count: int
    overlapping_regions: List[LayoutRegion]
    block_bbox_norm: Tuple[float, float, float, float]


@dataclass
class PolicyResult:
    """The decision made by the Policy Engine."""
    decision: Literal["semantic", "fallback"]
    reason: str  # e.g., "OOXML_TRUTHS_TABLE", "ORACLE_TEXTBOX_FLOATING"
    confidence: float = 1.0


def decide_policy(block: Block, ctx: Optional[LayoutContext] = None) -> PolicyResult:
    """Decide whether a block should be rendered semantically or as fallback.

    Args:
        block: The block to evaluate.
        ctx: Optional layout context (overlapping regions, etc.)

    Returns:
        PolicyResult with decision and reason.
    """
    # 1. OOXML Truth Resolution (Content Truth)
    # If OOXML clearly defines it as a table, we default to Table logic
    if isinstance(block, TableBlock):
        # Check OOXML complexity first
        ooxml_complexity = score_table_complexity(block)
        
        # High complexity -> Fallback immediately (unless user overrides?)
        if ooxml_complexity > 18:
            return PolicyResult("fallback", f"OOXML_COMPLEXITY_HIGH_{ooxml_complexity}")
            
        # If OOXML says simple/medium, it's semantic.
        return PolicyResult("semantic", "OOXML_TRUTHS_TABLE")

    # 2. Oracle Hints Resolution (Layout Truth vs Content Truth)
    # For non-Table blocks (Paragraphs, etc.) that Oracle thinks might be something else.
    if ctx and ctx.overlapping_regions:
        for region in ctx.overlapping_regions:
            # Handle both dict and object regions
            region_label = region.get('label', '') if isinstance(region, dict) else getattr(region, 'label', '')
            region_source = region.get('source', 'heuristic') if isinstance(region, dict) else getattr(region, 'source', 'heuristic')
            region_confidence = region.get('confidence', 0) if isinstance(region, dict) else getattr(region, 'confidence', 0)
            
            # Case A: Oracle says "Table" but OOXML doesn't (it's a Paragraph/List)
            if region_label == "table":
                is_heuristic = region_source == 'heuristic'
                
                if is_heuristic:
                    # Heuristics (lines) often trigger on headers/underlines.
                    # Unless confidence is extremely high, ignore.
                    continue
                else:
                    # Source=ML. 
                    if region_confidence > 0.85:
                        # Strong ML signal for Table, but OOXML has text.
                        return PolicyResult("fallback", "ML_TABLE_HIGH_CONF_NO_OOXML")

            # Case B: Oracle says "Textbox"
            if region_label == "textbox":
                return PolicyResult("semantic", "ORACLE_TEXTBOX_FLOATING")

    # Default to semantic
    return PolicyResult("semantic", "DEFAULT_SEMANTIC")


def score_table_complexity(block: TableBlock) -> int:
    """Calculate table complexity score.

    Scoring (from README):
    - merges: +2 per merged region
    - nested table: +10 each
    - columns > 8: +3
    - rows > 30: +3
    - mixed border styles: +5

    Buckets:
    - 0-8: simple → semantic
    - 9-18: medium → semantic
    - 19+: complex → fallback
    """
    score = 0

    # Count merges
    for row in block.rows:
        for cell in row.cells:
            if cell.col_span > 1 or cell.row_span > 1:
                score += 2

    # Column count
    if block.rows and len(block.rows[0].cells) > 8:
        score += 3

    # Row count
    if len(block.rows) > 30:
        score += 3

    return score
