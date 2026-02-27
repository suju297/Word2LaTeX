"""Table parser - Parse tables with complexity scoring."""

from __future__ import annotations

from typing import Dict, List
from lxml import etree

from wordtolatex.ir import TableBlock, TableRow, TableCell, InlineSpan

# OOXML namespaces
NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def parse_table(elem: etree._Element, block_id: int) -> TableBlock:
    """Parse a table element into a TableBlock.

    Args:
        elem: The w:tbl XML element.
        block_id: Block ID for this table.

    Returns:
        TableBlock with parsed rows and cells.
    """
    rows: List[TableRow] = []

    for tr in elem.findall("w:tr", NAMESPACES):
        cells: List[TableCell] = []

        for tc in tr.findall("w:tc", NAMESPACES):
            cell = _parse_table_cell(tc)
            cells.append(cell)

        if cells:
            rows.append(TableRow(cells=cells))

    block = TableBlock(
        id=f"b{block_id}",
        block_type="table",
        rows=rows,
    )

    # Calculate complexity score
    block.complexity_score = calculate_complexity(block)

    return block


def _parse_table_cell(tc: etree._Element) -> TableCell:
    """Parse a single table cell.

    Args:
        tc: The w:tc XML element.

    Returns:
        TableCell with content and span info.
    """
    cell = TableCell()

    # Check for cell properties (spans)
    tcPr = tc.find("w:tcPr", NAMESPACES)
    if tcPr is not None:
        # Column span
        gridSpan = tcPr.find("w:gridSpan", NAMESPACES)
        if gridSpan is not None:
            val = gridSpan.get(f"{{{NAMESPACES['w']}}}val")
            if val:
                cell.col_span = int(val)

        # Vertical merge
        vMerge = tcPr.find("w:vMerge", NAMESPACES)
        if vMerge is not None:
            val = vMerge.get(f"{{{NAMESPACES['w']}}}val")
            if val == "restart":
                cell.row_span = 2  # Will be computed properly in post-processing

    # Parse cell content (paragraphs)
    # IMPORTANT: Preserve paragraph boundaries by adding separator between w:p elements
    content: List[InlineSpan] = []
    paragraphs = tc.findall("w:p", NAMESPACES)
    
    for p_idx, p in enumerate(paragraphs):
        # Check paragraph style (for heading detection)
        pPr = p.find("w:pPr", NAMESPACES)
        pStyle = None
        if pPr is not None:
            pStyleElem = pPr.find("w:pStyle", NAMESPACES)
            if pStyleElem is not None:
                pStyle = pStyleElem.get(f"{{{NAMESPACES['w']}}}val")
        
        for r in p.findall(".//w:r", NAMESPACES):
            t = r.find("w:t", NAMESPACES)
            if t is not None and t.text:
                rPr = r.find("w:rPr", NAMESPACES)
                span = InlineSpan(
                    text=t.text,
                    bold=rPr is not None and rPr.find("w:b", NAMESPACES) is not None,
                    italic=rPr is not None and rPr.find("w:i", NAMESPACES) is not None,
                )
                # Store paragraph style for semantic detection
                if pStyle:
                    span.style_ref = pStyle
                content.append(span)
        
        # Add paragraph separator between different w:p elements (not after last)
        if p_idx < len(paragraphs) - 1 and content:
            # Insert a special "paragraph break" span
            content.append(InlineSpan(text="\n", is_paragraph_break=True))

    cell.content = content
    return cell


def calculate_complexity(table: TableBlock) -> int:
    """Calculate table complexity score per README Appendix A.

    Scoring:
    - merges: +2 per merged region
    - nested table: +10 each (not implemented yet)
    - columns > 8: +3
    - rows > 30: +3
    - mixed border styles: +5 (not implemented yet)

    Buckets:
    - 0-8: simple → semantic
    - 9-18: medium → semantic
    - 19+: complex → fallback
    """
    score = 0

    # Count merges
    for row in table.rows:
        for cell in row.cells:
            if cell.col_span > 1:
                score += 2
            if cell.row_span > 1:
                score += 2

    # Column count
    if table.rows:
        max_cols = max(len(row.cells) for row in table.rows)
        if max_cols > 8:
            score += 3

    # Row count
    if len(table.rows) > 30:
        score += 3

    return score
