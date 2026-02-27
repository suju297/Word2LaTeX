"""Report Generator - Create conversion quality report."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from wordtolatex.ir import Document, Block, HeadingBlock, ParagraphBlock, ListBlock, TableBlock, FigureBlock


@dataclass
class ConversionReport:
    """Report on the conversion quality and decisions."""
    
    # Metadata
    input_file: str = ""
    output_file: str = ""
    timestamp: str = ""
    
    # Stats
    total_blocks: int = 0
    semantic_blocks: int = 0
    fallback_blocks: int = 0
    
    # Block type counts
    headings: int = 0
    paragraphs: int = 0
    lists: int = 0
    tables: int = 0
    figures: int = 0
    
    # Lists
    list_items: int = 0
    
    # Tables
    simple_tables: int = 0
    medium_tables: int = 0
    complex_tables: int = 0
    
    # Formatting
    bold_spans: int = 0
    italic_spans: int = 0
    underline_spans: int = 0
    
    # Assets
    images_extracted: int = 0
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    # Fallback decisions
    fallbacks: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def generate_report(
    doc: Document,
    input_path: Path,
    output_path: Path,
) -> ConversionReport:
    """Generate a conversion report for a document.
    
    Args:
        doc: The parsed document IR.
        input_path: Path to the input DOCX file.
        output_path: Path to the output TEX file.
    
    Returns:
        ConversionReport with stats and decisions.
    """
    report = ConversionReport(
        input_file=str(input_path),
        output_file=str(output_path),
        timestamp=datetime.now().isoformat(),
        total_blocks=len(doc.blocks),
        images_extracted=len(doc.assets),
    )
    
    for block in doc.blocks:
        # Count by type
        if isinstance(block, HeadingBlock):
            report.headings += 1
        elif isinstance(block, ParagraphBlock):
            report.paragraphs += 1
        elif isinstance(block, ListBlock):
            report.lists += 1
            report.list_items += len(block.items)
        elif isinstance(block, TableBlock):
            report.tables += 1
            
            # Categorize table complexity
            if block.complexity_score <= 8:
                report.simple_tables += 1
            elif block.complexity_score <= 18:
                report.medium_tables += 1
            else:
                report.complex_tables += 1
                
        elif isinstance(block, FigureBlock):
            report.figures += 1
        
        # Count policy decisions
        if block.policy == "semantic":
            report.semantic_blocks += 1
        else:
            report.fallback_blocks += 1
            report.fallbacks.append({
                "block_id": block.id,
                "block_type": block.block_type,
                "reason": "complexity" if isinstance(block, TableBlock) else "unsupported",
            })
        
        # Count formatting
        for span in getattr(block, "content", []):
            if span.bold:
                report.bold_spans += 1
            if span.italic:
                report.italic_spans += 1
            if span.underline:
                report.underline_spans += 1
    
    # Add warnings
    if report.complex_tables > 0:
        report.warnings.append(f"{report.complex_tables} table(s) too complex, need image fallback")
    
    if report.fallback_blocks > 0:
        report.warnings.append(f"{report.fallback_blocks} block(s) using fallback rendering")
    
    return report


def save_report(report: ConversionReport, output_path: Path) -> None:
    """Save report to JSON file.
    
    Args:
        report: The conversion report.
        output_path: Path to save the report (should end with .json).
    """
    output_path.write_text(report.to_json(), encoding="utf-8")
