"""DOCX Document Parser - Main entry point for parsing .docx files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

from lxml import etree

from wordtolatex.ir import (
    Document,
    Section,
    Block,
    HeadingBlock,
    ParagraphBlock,
    ListBlock,
    ListItem,
    TableBlock,
    TableRow,
    TableCell,
    GridCell,
    GridRow,
    LayoutGridBlock,
    FigureBlock,
    InlineSpan,
    Asset,
    TabStop,
    TextboxBlock,
)
from wordtolatex.docx_parser.styles import get_heading_level, parse_styles as _parse_styles

# OOXML namespaces
NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "v": "urn:schemas-microsoft-com:vml",
    "o": "urn:schemas-microsoft-com:office:office",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}

# Common heading-like patterns for heuristic detection
HEADING_PATTERNS = [
    "education", "experience", "skills", "projects", "summary", "objective",
    "certifications", "awards", "publications", "references", "contact",
    "technical skills", "work experience", "academic projects", "professional experience",
]


def parse_docx(docx_path: Union[str, Path], output_dir: Optional[Path] = None) -> Document:
    """Parse a .docx file and return a Document IR.

    Args:
        docx_path: Path to the .docx file.
        output_dir: Optional directory for extracted assets.

    Returns:
        Document: The parsed document representation.
    """
    docx_path = Path(docx_path)
    if output_dir is None:
        output_dir = docx_path.parent

    with zipfile.ZipFile(docx_path, "r") as zf:
        # Parse main document
        document_xml = zf.read("word/document.xml")
        tree = etree.fromstring(document_xml)

        # Parse styles (if exists)
        styles_map: Dict[str, Dict] = {}
        if "word/styles.xml" in zf.namelist():
            styles_xml = zf.read("word/styles.xml")
            styles_map = _parse_styles(styles_xml)

        # Parse numbering (if exists)
        numbering_map: Dict[str, Dict] = {}
        if "word/numbering.xml" in zf.namelist():
            numbering_xml = zf.read("word/numbering.xml")
            numbering_map = _parse_numbering(numbering_xml)

        # Parse relationships for images
        rels_map: Dict[str, str] = {}
        if "word/_rels/document.xml.rels" in zf.namelist():
            rels_xml = zf.read("word/_rels/document.xml.rels")
            rels_map = _parse_relationships(rels_xml)

        # Parse footnotes (if exists)
        footnotes_map: Dict[str, List[InlineSpan]] = {}
        if "word/footnotes.xml" in zf.namelist():
            footnotes_xml = zf.read("word/footnotes.xml")
            footnotes_map = _parse_footnotes(footnotes_xml, rels_map)

        # Extract media assets
        assets = _extract_media(zf, output_dir)

        # Build document
        doc = Document()
        doc.sections = [_parse_section(tree)]
        doc.blocks = _parse_body(tree, styles_map, numbering_map, rels_map, footnotes_map)
        doc.assets = assets
        doc.styles = styles_map

        # Analyze layout hints (e.g. font family default)
        _analyze_layout_hints(doc)
        
        # Detect if document should use numbered sections (IEEE/academic style)
        _detect_numbered_sections(doc)

        # Count objects for diagnostics
        doc.metadata["object_counts"] = {
            "textboxes": len(tree.xpath("//w:txbxContent", namespaces=NAMESPACES)),
            "shapes": len(tree.xpath("//v:shape", namespaces=NAMESPACES)) + len(tree.xpath("//wps:wsp", namespaces=NAMESPACES)),
            "images": len(tree.xpath("//w:drawing", namespaces=NAMESPACES)),
            "v_images": len(tree.xpath("//v:imagedata", namespaces=NAMESPACES)),
        }

        # Parse headers
        for name in zf.namelist():
            if name.startswith("word/header") and name.endswith(".xml"):
                header_xml = zf.read(name)
                header_tree = etree.fromstring(header_xml)
                header_blocks = _parse_header_footer(header_tree, styles_map)
                doc.header_blocks.extend(header_blocks)

        # Parse footers
        for name in zf.namelist():
            if name.startswith("word/footer") and name.endswith(".xml"):
                footer_xml = zf.read(name)
                footer_tree = etree.fromstring(footer_xml)
                footer_blocks = _parse_header_footer(footer_tree, styles_map)
                doc.footer_blocks.extend(footer_blocks)

    return doc


def _parse_header_footer(tree: etree._Element, styles_map: Dict, rels_map: Dict = None) -> List[Block]:
    """Parse header or footer XML into blocks."""
    if rels_map is None:
        rels_map = {}
    blocks: List[Block] = []
    block_id = 0

    for p in tree.findall(".//w:p", NAMESPACES):
        pPr = p.find("w:pPr", NAMESPACES)
        style_ref = None
        if pPr is not None:
            pStyle = pPr.find("w:pStyle", NAMESPACES)
            if pStyle is not None:
                style_ref = pStyle.get(f"{{{NAMESPACES['w']}}}val")

        content = _parse_runs(p, rels_map)
        layout = _parse_layout_properties(pPr, style_ref, styles_map)
        if layout.get("font_size") is None:
            inferred_size = _dominant_run_font_size(content)
            if inferred_size:
                layout["font_size"] = inferred_size

        if not content:
            blocks.append(ParagraphBlock(
                id=f"hf{block_id}",
                block_type="paragraph",
                content=[],
                spacing_before=layout["spacing_before"],
                spacing_after=layout["spacing_after"],
                line_spacing=layout["line_spacing"],
                line_spacing_type=layout.get("line_spacing_type", "multiplier"),
                tab_stops=layout["tab_stops"],
                justification=layout["justification"],
                font_family=layout.get("font_family"),
                font_size=layout.get("font_size"),
                is_empty=True,
            ))
            block_id += 1
            continue

        blocks.append(ParagraphBlock(
            id=f"hf{block_id}",
            block_type="paragraph",
            style_ref=style_ref,
            content=content,
            spacing_before=layout["spacing_before"],
            spacing_after=layout["spacing_after"],
            line_spacing=layout["line_spacing"],
            line_spacing_type=layout.get("line_spacing_type", "multiplier"),
            tab_stops=layout["tab_stops"],
            justification=layout["justification"],
            font_family=layout.get("font_family"),
            font_size=layout.get("font_size"),
        ))
        block_id += 1

    return blocks



def _parse_section(tree: etree._Element) -> Section:
    """Extract section properties (page setup)."""
    section = Section()

    sect_pr = tree.find(".//w:sectPr", NAMESPACES)
    if sect_pr is not None:
        # Page size
        pg_sz = sect_pr.find("w:pgSz", NAMESPACES)
        if pg_sz is not None:
            w = pg_sz.get(f"{{{NAMESPACES['w']}}}w")
            h = pg_sz.get(f"{{{NAMESPACES['w']}}}h")
            if w:
                section.page_width = int(w) / 1440
            if h:
                section.page_height = int(h) / 1440

        # Margins
        pg_mar = sect_pr.find("w:pgMar", NAMESPACES)
        if pg_mar is not None:
            for attr, prop in [
                ("top", "margin_top"),
                ("bottom", "margin_bottom"),
                ("left", "margin_left"),
                ("right", "margin_right"),
            ]:
                val = pg_mar.get(f"{{{NAMESPACES['w']}}}{attr}")
                if val:
                    setattr(section, prop, int(val) / 1440)

        # Columns
        cols = sect_pr.find("w:cols", NAMESPACES)
        if cols is not None:
             num = cols.get(f"{{{NAMESPACES['w']}}}num")
             space = cols.get(f"{{{NAMESPACES['w']}}}space")
             if num:
                 section.col_count = int(num)
             if space:
                 section.col_spacing = int(space) / 20  # twips to points

    return section




def _parse_numbering(numbering_xml: bytes) -> Dict[str, Dict]:
    """Parse numbering.xml for list definitions."""
    tree = etree.fromstring(numbering_xml)
    
    # Parse abstract numbering definitions
    abstract_nums: Dict[str, Dict] = {}
    for abstract in tree.findall(".//w:abstractNum", NAMESPACES):
        abstract_id = abstract.get(f"{{{NAMESPACES['w']}}}abstractNumId")
        if not abstract_id:
            continue

        levels: Dict[int, Dict] = {}
        for lvl in abstract.findall("w:lvl", NAMESPACES):
            lvl_id = lvl.get(f"{{{NAMESPACES['w']}}}ilvl")
            if lvl_id is None:
                continue

            num_fmt_elem = lvl.find("w:numFmt", NAMESPACES)
            num_fmt = num_fmt_elem.get(f"{{{NAMESPACES['w']}}}val") if num_fmt_elem is not None else "bullet"

            levels[int(lvl_id)] = {
                "num_fmt": num_fmt,
                "ordered": num_fmt in ("decimal", "lowerLetter", "upperLetter", "lowerRoman", "upperRoman"),
            }

        abstract_nums[abstract_id] = {"levels": levels}

    # Parse numbering instances
    numbering_map: Dict[str, Dict] = {}
    for num in tree.findall(".//w:num", NAMESPACES):
        num_id = num.get(f"{{{NAMESPACES['w']}}}numId")
        abstract_ref = num.find("w:abstractNumId", NAMESPACES)
        
        if num_id and abstract_ref is not None:
            abstract_id = abstract_ref.get(f"{{{NAMESPACES['w']}}}val")
            if abstract_id and abstract_id in abstract_nums:
                numbering_map[num_id] = abstract_nums[abstract_id]

    return numbering_map


def _parse_relationships(rels_xml: bytes) -> Dict[str, str]:
    """Parse document relationships for image references."""
    tree = etree.fromstring(rels_xml)
    rels_map: Dict[str, str] = {}

    ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    for rel in tree.findall(".//r:Relationship", ns):
        rel_id = rel.get("Id")
        target = rel.get("Target")
        if rel_id and target:
            rels_map[rel_id] = target

    return rels_map


def _parse_footnotes(footnotes_xml: bytes, rels_map: Dict) -> Dict[str, List[InlineSpan]]:
    """Parse footnotes and return {id: content} map.
    
    Footnotes with id <= 0 are system separators and are skipped.
    """
    tree = etree.fromstring(footnotes_xml)
    footnotes: Dict[str, List[InlineSpan]] = {}
    
    for fn in tree.findall(".//w:footnote", NAMESPACES):
        fn_id = fn.get(f"{{{NAMESPACES['w']}}}id")
        if fn_id:
            try:
                if int(fn_id) <= 0:  # Skip separator footnotes (-1, 0)
                    continue
            except ValueError:
                pass
            
            # Parse all paragraphs in the footnote
            content: List[InlineSpan] = []
            for para in fn.findall(".//w:p", NAMESPACES):
                para_content = _parse_runs(para, rels_map)
                content.extend(para_content)
                # Add space between paragraphs
                if para_content:
                    content.append(InlineSpan(text=" "))
            
            if content:
                footnotes[fn_id] = content
    
    return footnotes


def _extract_media(zf: zipfile.ZipFile, output_dir: Path) -> List[Asset]:
    """Extract media files from the DOCX."""
    assets: List[Asset] = []
    media_dir = output_dir / "media"

    for name in zf.namelist():
        if name.startswith("word/media/"):
            filename = Path(name).name
            output_path = media_dir / filename

            media_dir.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src:
                output_path.write_bytes(src.read())

            assets.append(Asset(
                original_path=name,
                output_path=f"media/{filename}",
                asset_type="image",
            ))

    return assets


def _parse_body(tree: etree._Element, styles_map: Dict, numbering_map: Dict, rels_map: Dict, footnotes_map: Dict = None) -> List[Block]:
    """Parse document body into blocks."""
    if footnotes_map is None:
        footnotes_map = {}
    blocks: List[Block] = []
    body = tree.find(".//w:body", NAMESPACES)

    if body is None:
        return blocks

    block_id = 0
    current_list: Optional[ListBlock] = None
    current_num_id: Optional[str] = None

    for child in body:
        tag = etree.QName(child).localname

        if tag == "p":
            # Check if this paragraph is a list item
            num_pr = child.find(".//w:numPr", NAMESPACES)
            if num_pr is not None:
                num_id_elem = num_pr.find("w:numId", NAMESPACES)
                ilvl_elem = num_pr.find("w:ilvl", NAMESPACES)
                
                if num_id_elem is not None:
                    num_id = num_id_elem.get(f"{{{NAMESPACES['w']}}}val")
                    ilvl = int(ilvl_elem.get(f"{{{NAMESPACES['w']}}}val", "0")) if ilvl_elem is not None else 0
                    
                    # Get list info
                    num_def = numbering_map.get(num_id, {})
                    levels = num_def.get("levels", {})
                    level_info = levels.get(ilvl, {"ordered": False})
                    
                    # Parse list item content
                    content = _parse_runs(child, rels_map)
                    if content:
                        list_item = ListItem(content=content)

                        # Capture layout info from list paragraph to size bullets correctly
                        pPr = child.find("w:pPr", NAMESPACES)
                        style_ref = None
                        if pPr is not None:
                            pStyle = pPr.find("w:pStyle", NAMESPACES)
                            if pStyle is not None:
                                style_ref = pStyle.get(f"{{{NAMESPACES['w']}}}val")

                        layout = _parse_layout_properties(pPr, style_ref, styles_map)
                        if layout.get("font_size") is None:
                            inferred_size = _dominant_run_font_size(content)
                            if inferred_size:
                                layout["font_size"] = inferred_size
                        
                        # Start new list or continue existing
                        if current_list is None or current_num_id != num_id:
                            if current_list is not None:
                                blocks.append(current_list)
                                block_id += 1
                            
                            current_list = ListBlock(
                                id=f"b{block_id}",
                                block_type="list",
                                ordered=level_info.get("ordered", False),
                                items=[list_item],
                                spacing_before=layout["spacing_before"],
                                spacing_after=layout["spacing_after"],
                                line_spacing=layout["line_spacing"],
                                line_spacing_type=layout.get("line_spacing_type", "multiplier"),
                                font_family=layout.get("font_family"),
                                font_size=layout.get("font_size"),
                            )
                            current_num_id = num_id
                        else:
                            current_list.items.append(list_item)
                            if current_list.line_spacing is None and layout["line_spacing"] is not None:
                                current_list.line_spacing = layout["line_spacing"]
                                current_list.line_spacing_type = layout.get("line_spacing_type", "multiplier")
                            if current_list.font_family is None and layout.get("font_family"):
                                current_list.font_family = layout["font_family"]
                            if current_list.font_size is None and layout.get("font_size"):
                                current_list.font_size = layout["font_size"]
                    continue
            
            # Not a list item - flush any pending list
            if current_list is not None:
                blocks.append(current_list)
                block_id += 1
                current_list = None
                current_num_id = None
            
            block = _parse_paragraph(child, styles_map, block_id, rels_map, footnotes_map)
            if block:
                blocks.append(block)
                block_id += 1

        elif tag == "tbl":
            # Flush any pending list
            if current_list is not None:
                blocks.append(current_list)
                block_id += 1
                current_list = None
                current_num_id = None
            
            # Check if this is a layout table (resume/template structure)
            layout_result = compute_layout_score(child)
            
            if layout_result['is_layout']:
                # Build LayoutGridBlock to preserve geometry (don't flatten!)
                layout_grid = _parse_layout_grid(child, block_id, styles_map, rels_map)
                # Build detailed signal trace for debugging
                sig = layout_result['signals']
                signal_str = " ".join(f"{k}={v:.2f}" for k, v in sig.items())
                # Add raw counts for explainability
                counts = f"rows={layout_result['rows']} cols={layout_result['cols']} merged={layout_result['merged_cells']}/{layout_result['total_cells']} digit_pct={layout_result['digit_pct']:.2f}"
                layout_grid.policy_reason = f"LAYOUT_GRID_PRESERVED score={layout_result['score']:.2f} {counts} {signal_str}"
                blocks.append(layout_grid)
                block_id += 1
            else:
                table_block = _parse_table(child, block_id)
                # Add trace for data tables too
                sig = layout_result['signals']
                signal_str = " ".join(f"{k}={v:.2f}" for k, v in sig.items())
                counts = f"rows={layout_result['rows']} cols={layout_result['cols']} merged={layout_result['merged_cells']}/{layout_result['total_cells']} digit_pct={layout_result['digit_pct']:.2f}"
                table_block.policy_reason = f"DATA_TABLE score={layout_result['score']:.2f} {counts} {signal_str}"
                blocks.append(table_block)
                block_id += 1

    # Flush final list if any
    if current_list is not None:
        blocks.append(current_list)

    return blocks


def _parse_paragraph(elem: etree._Element, styles_map: Dict, block_id: int, rels_map: Dict = None, footnotes_map: Dict = None) -> Optional[Block]:
    """Parse a paragraph element."""
    if rels_map is None:
        rels_map = {}
    if footnotes_map is None:
        footnotes_map = {}
    # Get paragraph style
    pPr = elem.find("w:pPr", NAMESPACES)
    style_ref = None
    heading_level = None

    if pPr is not None:
        pStyle = pPr.find("w:pStyle", NAMESPACES)
        if pStyle is not None:
            style_ref = pStyle.get(f"{{{NAMESPACES['w']}}}val")

            # Check if it's a heading using centralized logic
            if style_ref:
                heading_level = get_heading_level(style_ref, styles_map)

    # Parse runs (inline content) with hyperlink and footnote support
    content = _parse_runs(elem, rels_map, footnotes_map)

    # For empty paragraphs, mark as is_empty=True to preserve vertical space
    # Word uses blank paragraphs for spacing - dropping them causes cumulative compaction
    # IMPORTANT: Still need to extract layout properties for line spacing inheritance
    if not content:
        layout = _parse_layout_properties(pPr, style_ref, styles_map)
        return ParagraphBlock(
            id=f"para_{block_id}",
            block_type="paragraph",
            content=[],
            spacing_before=layout["spacing_before"],
            spacing_after=layout["spacing_after"],
            line_spacing=layout["line_spacing"],
            line_spacing_type=layout.get("line_spacing_type", "multiplier"),
            font_family=layout.get("font_family"),
            font_size=layout.get("font_size"),
            is_empty=True,  # Mark as blank paragraph for generator
        )

    # HEURISTIC: Detect headings by content pattern (for resumes, etc.)
    if heading_level is None:
        text = "".join(span.text for span in content).strip()
        text_lower = text.lower()
        
        # Check if it matches common heading patterns
        for pattern in HEADING_PATTERNS:
            if text_lower == pattern or text_lower.startswith(pattern + " "):
                # Also check if it's all bold or short enough to be a heading
                all_bold = all(span.bold for span in content)
                is_short = len(text) < 50
                
                if all_bold or is_short:
                    heading_level = 1  # Section-level heading
                    break

    # Extract Layout Properties (Justification & Spacing & Tabs)
    layout = _parse_layout_properties(pPr, style_ref, styles_map)
    if layout.get("font_size") is None:
        inferred_size = _dominant_run_font_size(content)
        if inferred_size:
            layout["font_size"] = inferred_size
    
    # Create appropriate block type
    if heading_level:
        return HeadingBlock(
            id=f"b{block_id}",
            block_type="heading",
            level=heading_level,
            style_ref=style_ref,
            content=content,
            justification=layout["justification"],
            spacing_before=layout["spacing_before"],
            spacing_after=layout["spacing_after"],
            line_spacing=layout["line_spacing"],
            line_spacing_type=layout.get("line_spacing_type", "multiplier"),
            tab_stops=layout["tab_stops"],
            font_family=layout.get("font_family"),
            font_size=layout.get("font_size"),
        )
    else:
        return ParagraphBlock(
            id=f"b{block_id}",
            block_type="paragraph",
            style_ref=style_ref,
            content=content,
            justification=layout["justification"],
            spacing_before=layout["spacing_before"],
            spacing_after=layout["spacing_after"],
            line_spacing=layout["line_spacing"],
            line_spacing_type=layout.get("line_spacing_type", "multiplier"),
            tab_stops=layout["tab_stops"],
            font_family=layout.get("font_family"),
            font_size=layout.get("font_size"),
        )


def _dominant_run_font_size(spans: List[InlineSpan]) -> Optional[float]:
    """Infer a paragraph font size from dominant run sizes when no style size is set."""
    size_weights: Dict[float, int] = {}
    for span in spans:
        if not span.font_size or not span.text:
            continue
        weight = sum(1 for c in span.text if not c.isspace())
        if weight <= 0:
            continue
        size_weights[span.font_size] = size_weights.get(span.font_size, 0) + weight
    if not size_weights:
        return None
    return max(size_weights.items(), key=lambda item: (item[1], item[0]))[0]


def compute_layout_score(elem: etree._Element, page_width_twips: int = 9360) -> dict:
    """Compute weighted layout table score with signal breakdown.
    
    Signals (weights):
    - borderless: 0.25 (no visible borders)
    - merge_rate: 0.15 (fraction of merged cells)
    - row_variance: 0.10 (variation in columns per row)
    - text_heavy: 0.15 (% cells with mostly text, few numbers)
    - header_like_row: 0.20 (ALLCAPS/short tokens in first col)
    - width_near_page: 0.15 (table width ~ page width)
    
    Returns:
        dict with 'score' (float 0-1), 'is_layout' (bool), and signal breakdown
    """
    signals = {
        'borderless': 0.0,
        'merge_rate': 0.0,
        'row_variance': 0.0,
        'text_heavy': 0.0,
        'header_like_row': 0.0,
        'width_near_page': 0.0,
    }
    weights = {
        'borderless': 0.25,
        'merge_rate': 0.15,
        'row_variance': 0.10,
        'text_heavy': 0.15,
        'header_like_row': 0.20,
        'width_near_page': 0.15,
    }
    
    # 1. Borderless check
    tblPr = elem.find("w:tblPr", NAMESPACES)
    has_borders = False
    table_width = 0
    
    if tblPr is not None:
        tblBorders = tblPr.find("w:tblBorders", NAMESPACES)
        if tblBorders is not None:
            for border_type in ["top", "left", "bottom", "right", "insideH", "insideV"]:
                border = tblBorders.find(f"w:{border_type}", NAMESPACES)
                if border is not None:
                    val = border.get(f"{{{NAMESPACES['w']}}}val")
                    if val and val not in ("nil", "none", ""):
                        has_borders = True
                        break
        
        # Get table width
        tblW = tblPr.find("w:tblW", NAMESPACES)
        if tblW is not None:
            w_type = tblW.get(f"{{{NAMESPACES['w']}}}type", "dxa")
            w_val = tblW.get(f"{{{NAMESPACES['w']}}}w", "0")
            if w_type == "dxa":
                table_width = int(w_val) if w_val.isdigit() else 0
            elif w_type == "pct":
                # Percentage of page width (stored as 50ths of a percent)
                pct = int(w_val) / 5000.0 if w_val.isdigit() else 0
                table_width = int(pct * page_width_twips)
    
    # Fallback: if table_width is 0 (auto), sum gridCol widths
    if table_width == 0:
        tbl_grid = elem.find("w:tblGrid", NAMESPACES)
        if tbl_grid is not None:
            for grid_col in tbl_grid.findall("w:gridCol", NAMESPACES):
                w = grid_col.get(f"{{{NAMESPACES['w']}}}w", "0")
                if w.isdigit():
                    table_width += int(w)
    
    signals['borderless'] = 1.0 if not has_borders else 0.0
    
    # 2-5. Analyze cells
    rows = elem.findall("w:tr", NAMESPACES)
    if not rows:
        return {'score': 0.0, 'is_layout': False, 'signals': signals}
    
    total_cells = 0
    merged_cells = 0
    heading_cells = 0
    text_heavy_cells = 0
    header_like_count = 0
    cols_per_row = []
    
    for row_idx, tr in enumerate(rows):
        row_cells = tr.findall("w:tc", NAMESPACES)
        # Count logical columns (accounting for gridSpan)
        logical_cols = 0
        
        for tc in row_cells:
            total_cells += 1
            
            # Check for merged cells
            tcPr = tc.find("w:tcPr", NAMESPACES)
            grid_span = 1
            if tcPr is not None:
                gs = tcPr.find("w:gridSpan", NAMESPACES)
                if gs is not None:
                    gs_val = gs.get(f"{{{NAMESPACES['w']}}}val", "1")
                    grid_span = int(gs_val) if gs_val.isdigit() else 1
                    if grid_span > 1:
                        merged_cells += 1
                if tcPr.find("w:vMerge", NAMESPACES) is not None:
                    merged_cells += 1
            
            logical_cols += grid_span
            
            # Analyze cell content
            cell_text = ""
            is_heading_style = False
            
            for p in tc.findall("w:p", NAMESPACES):
                pPr = p.find("w:pPr", NAMESPACES)
                if pPr is not None:
                    pStyle = pPr.find("w:pStyle", NAMESPACES)
                    if pStyle is not None:
                        style = pStyle.get(f"{{{NAMESPACES['w']}}}val", "")
                        if "Heading" in style or "heading" in style.lower():
                            is_heading_style = True
                
                # Extract text for analysis
                for t in p.findall(".//w:t", NAMESPACES):
                    if t.text:
                        cell_text += t.text
            
            if is_heading_style:
                heading_cells += 1
            
            # Text-heavy check (mostly letters, few digits)
            if cell_text:
                alpha_count = sum(1 for c in cell_text if c.isalpha())
                digit_count = sum(1 for c in cell_text if c.isdigit())
                if alpha_count > digit_count * 3:  # At least 3x more letters than digits
                    text_heavy_cells += 1
            
            # Header-like row detection (first column with ALLCAPS or short token)
            if row_idx == 0 or len(row_cells) <= 2:
                text_stripped = cell_text.strip()
                if text_stripped:
                    # ALLCAPS short text
                    if text_stripped.isupper() and len(text_stripped) < 30:
                        header_like_count += 1
                    # Common resume section headers
                    elif text_stripped.lower() in HEADING_PATTERNS:
                        header_like_count += 1
        
        cols_per_row.append(logical_cols)
    
    if total_cells == 0:
        return {'score': 0.0, 'is_layout': False, 'signals': signals}
    
    # 2. Merge rate
    signals['merge_rate'] = min(1.0, merged_cells / total_cells * 2)  # Scale up, cap at 1
    
    # 3. Row variance (normalized std deviation of columns per row)
    if len(cols_per_row) > 1:
        mean_cols = sum(cols_per_row) / len(cols_per_row)
        variance = sum((c - mean_cols) ** 2 for c in cols_per_row) / len(cols_per_row)
        std_dev = variance ** 0.5
        signals['row_variance'] = min(1.0, std_dev / max(mean_cols, 1))
    
    # 4. Text-heavy
    signals['text_heavy'] = text_heavy_cells / total_cells if total_cells > 0 else 0
    
    # 5. Header-like row
    signals['header_like_row'] = min(1.0, header_like_count / max(len(rows), 1))
    
    # 6. Width near page width
    if table_width > 0:
        width_ratio = table_width / page_width_twips
        # If width is 80-100% of page width, score high
        if width_ratio >= 0.8:
            signals['width_near_page'] = 1.0
        elif width_ratio >= 0.6:
            signals['width_near_page'] = 0.5
    
    # Compute weighted score
    score = sum(signals[k] * weights[k] for k in signals)
    
    # Heading cells bonus: if any cell has heading style, boost score
    if heading_cells > 0:
        score = min(1.0, score + 0.15)
    
    # Regular-grid guardrail: prevent flattening actual data tables
    # Data tables tend to have regular structure (uniform rows/cols, no heading-like cells)
    # Layout tables have irregular structure or obvious section headers
    num_rows = len(rows)
    num_cols = max(cols_per_row) if cols_per_row else 0
    
    # Count digit-heavy cells (contain numbers/math - common in data tables)
    digit_cells = 0
    for tr in rows:
        for tc in tr.findall("w:tc", NAMESPACES):
            cell_text = ""
            for t in tc.findall(".//w:t", NAMESPACES):
                if t.text:
                    cell_text += t.text
            # Cell is digit-heavy if ≥20% digits or contains math symbols
            if cell_text:
                digit_count = sum(1 for c in cell_text if c.isdigit())
                has_math = any(c in cell_text for c in ['%', '$', '€', '£', '+', '=', '<', '>'])
                if digit_count >= len(cell_text) * 0.2 or has_math:
                    digit_cells += 1
    
    digit_pct = digit_cells / total_cells if total_cells > 0 else 0
    
    is_regular_grid = (
        num_rows >= 3 and 
        num_cols >= 3 and 
        signals['row_variance'] < 0.1 and  # Very uniform rows
        signals['header_like_row'] < 0.1  # No heading-like content
    )
    
    # Digit-heavy guardrail: if ≥40% cells have digits/math, likely a data table
    is_digit_heavy = digit_pct >= 0.40
    
    if is_regular_grid or is_digit_heavy:
        # Only flatten if borderless AND width_near_page are both very strong
        borderless_width_score = signals['borderless'] + signals['width_near_page']
        if borderless_width_score < 1.8:  # Need both to be ~1.0
            score = min(score, 0.40)  # Force below threshold
    
    return {
        'score': score,
        'is_layout': score >= 0.45,  # Threshold
        'signals': signals,
        # Raw counts for debugging
        'rows': num_rows,
        'cols': num_cols,
        'merged_cells': merged_cells,
        'total_cells': total_cells,
        'digit_pct': digit_pct,
    }


def _is_layout_table(elem: etree._Element) -> bool:
    """Detect if a table is a 'layout table' (used for positioning, not data).
    
    Layout tables in Word are used for resume/template layouts and should be
    flattened to semantic blocks rather than rendered as LaTeX tables.
    
    Uses weighted scoring across multiple signals - see compute_layout_score().
    """
    result = compute_layout_score(elem)
    return result['is_layout']


def _parse_layout_grid(elem: etree._Element, block_id: int, 
                       styles_map: Dict, rels_map: Dict) -> LayoutGridBlock:
    """Parse a layout table into a LayoutGridBlock preserving geometry.
    
    Unlike _flatten_layout_table, this preserves the 2-column structure
    so that content ends up at the same X/Y coordinates as in Word.
    """
    # Extract column widths from w:tblGrid
    col_widths_twips = None
    tbl_grid = elem.find("w:tblGrid", NAMESPACES)
    if tbl_grid is not None:
        widths = []
        for grid_col in tbl_grid.findall("w:gridCol", NAMESPACES):
            w = grid_col.get(f"{{{NAMESPACES['w']}}}w")
            if w:
                try:
                    widths.append(int(w))
                except ValueError:
                    pass
        if widths:
            col_widths_twips = widths
    
    # Extract table-level cell margins (tblCellMar)
    cell_margin_twips = None
    tbl_pr = elem.find("w:tblPr", NAMESPACES)
    if tbl_pr is not None:
        cell_mar = tbl_pr.find("w:tblCellMar", NAMESPACES)
        if cell_mar is not None:
            cell_margin_twips = {}
            for side in ['top', 'left', 'bottom', 'right']:
                side_el = cell_mar.find(f"w:{side}", NAMESPACES)
                if side_el is not None:
                    val = side_el.get(f"{{{NAMESPACES['w']}}}w")
                    if val:
                        try:
                            cell_margin_twips[side] = int(val)
                        except ValueError:
                            pass
    
    # Parse rows and cells
    rows = []
    for tr in elem.findall("w:tr", NAMESPACES):
        # Extract row height
        height_twips = None
        height_rule = None
        tr_pr = tr.find("w:trPr", NAMESPACES)
        if tr_pr is not None:
            tr_height = tr_pr.find("w:trHeight", NAMESPACES)
            if tr_height is not None:
                val = tr_height.get(f"{{{NAMESPACES['w']}}}val")
                if val:
                    try:
                        height_twips = int(val)
                    except ValueError:
                        pass
                height_rule = tr_height.get(f"{{{NAMESPACES['w']}}}hRule")
        
        # Parse cells
        row_cells = []
        for tc in tr.findall("w:tc", NAMESPACES):
            # Get gridSpan for merged cells
            col_span = 1
            padding_twips = None
            v_align = "top"  # Default: Word defaults to top alignment
            tc_pr = tc.find("w:tcPr", NAMESPACES)
            if tc_pr is not None:
                grid_span = tc_pr.find("w:gridSpan", NAMESPACES)
                if grid_span is not None:
                    val = grid_span.get(f"{{{NAMESPACES['w']}}}val")
                    if val:
                        try:
                            col_span = int(val)
                        except ValueError:
                            pass
                
                # Extract cell-level padding overrides (tcMar)
                tc_mar = tc_pr.find("w:tcMar", NAMESPACES)
                if tc_mar is not None:
                    padding_twips = {}
                    for side in ['top', 'left', 'bottom', 'right']:
                        side_el = tc_mar.find(f"w:{side}", NAMESPACES)
                        if side_el is not None:
                            val = side_el.get(f"{{{NAMESPACES['w']}}}w")
                            if val:
                                try:
                                    padding_twips[side] = int(val)
                                except ValueError:
                                    pass
                
                # Extract vertical alignment (w:vAlign)
                v_align_el = tc_pr.find("w:vAlign", NAMESPACES)
                if v_align_el is not None:
                    val = v_align_el.get(f"{{{NAMESPACES['w']}}}val")
                    if val:
                        # Map OOXML values to our IR values
                        # Standard: top, center, bottom
                        # Rare: both, distributed, justify -> treat as center
                        if val in ("center", "bottom"):
                            v_align = val
                        elif val in ("both", "distributed", "justify"):
                            v_align = "center"
                        # else keep "top" (default)
            
            # Parse cell content as blocks
            cell_blocks = []
            for p in tc.findall("w:p", NAMESPACES):
                parsed = _parse_paragraph(p, styles_map, block_id, rels_map)
                if parsed is not None:
                    cell_blocks.append(parsed)
                    block_id += 1
            
            row_cells.append(GridCell(
                blocks=cell_blocks,
                col_span=col_span,
                padding_twips=padding_twips,
                v_align=v_align,
            ))
        
        rows.append(GridRow(
            cells=row_cells,
            height_twips=height_twips,
            height_rule=height_rule,
        ))
    
    return LayoutGridBlock(
        id=f"layout_grid_{block_id}",
        block_type="layout_grid",
        col_widths_twips=col_widths_twips,
        rows=rows,
        cell_margin_twips=cell_margin_twips,
    )


def _flatten_layout_table(elem: etree._Element, start_block_id: int, 
                          styles_map: Dict, rels_map: Dict) -> List[Block]:
    """Flatten a layout table into a sequence of semantic blocks.
    
    Converts table cells into HeadingBlocks and ParagraphBlocks,
    preserving the semantic structure of resumé/template content.
    
    Handles various two-column resume patterns:
    - 2 cells: left body + right date/location
    - 3 cells with empty first: icon/spacer + content + right
    - 3 cells with empty middle: content + spacer + right
    """
    import re
    
    blocks: List[Block] = []
    block_id = start_block_id
    
    def _get_cell_text(cell) -> str:
        """Extract all text from a cell."""
        text = ""
        for t in cell.findall(".//w:t", NAMESPACES):
            if t.text:
                text += t.text
        return text.strip()
    
    def _is_tiny_cell(cell) -> bool:
        """Check if cell is empty or tiny (icon, spacer, etc)."""
        text = _get_cell_text(cell)
        return len(text) < 5  # Less than 5 chars = tiny
    
    def _is_right_element_pattern(text: str) -> bool:
        """Check if text matches date/location/metadata patterns.
        
        Returns True if text looks like a right-aligned element (date, location, etc).
        Returns False if text looks like actual content (multi-sentence, bullets, long prose).
        """
        if not text or len(text) > 120:  # Sanity max length
            return False
        
        # VETO rules: prevent treating content cells as right_element
        
        # Veto 1: Multiple sentences (2+ sentence-ending punctuation)
        sentence_ends = text.count('.') + text.count('!') + text.count('?')
        if sentence_ends >= 2:
            return False
        
        # Veto 2: Bullet-like markers (2+ bullets/checkmarks/list markers)
        bullet_markers = text.count('•') + text.count('●') + text.count('○') + text.count('✓') + text.count('*')
        if bullet_markers >= 2:
            return False
        
        # Veto 3: Too many words without date/location pattern (prose content)
        words = text.split()
        if len(words) > 12:
            # Check if it at least has date/location before rejecting
            has_date = bool(re.search(r'\b(19|20)\d{2}\b', text) or 
                           re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|current)\b', text.lower()))
            has_location = bool(re.search(r',\s*[A-Z]{2}\b', text))
            if not (has_date or has_location):
                return False
        
        text_lower = text.lower()
        
        # Pattern 1: Date-ish (months, years, Present, en-dash)
        date_patterns = [
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b',  # Month abbrev
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\b(19|20)\d{2}\b',  # Year (1900s/2000s)
            r'\bpresent\b',
            r'\bcurrent\b',
            r'–|—|--',  # En-dash, em-dash, double hyphen
        ]
        for pattern in date_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Pattern 2: Location (city, state/country)
        # "San Francisco, CA" or "New York, NY" or "London, UK"
        location_pattern = r',\s*[A-Z]{2}\b'  # ", CA", ", NY", ", UK"
        if re.search(location_pattern, text):
            return True
        
        # Pattern 3: High punctuation density (|, –, ,)
        punct_chars = text.count('|') + text.count('–') + text.count('—') + text.count(',')
        if punct_chars >= 2 and len(text) < 80:
            return True
        
        # Pattern 4: Numeric-heavy AND short word count (≤8 tokens)
        if len(words) <= 8:
            digit_count = sum(1 for c in text if c.isdigit())
            if digit_count >= 4:  # At least 4 digits (like a year range)
                return True
        
        return False
    
    def _get_right_col_fraction(right_text: str) -> float:
        """Compute adaptive right column width fraction based on content length.
        
        Returns a value between 0.20 and 0.38.
        """
        length = len(right_text)
        if length < 15:
            return 0.22  # Short dates: "2020-2024"
        elif length < 35:
            return 0.28  # Medium: "Jan 2020 - Present"
        else:
            return 0.34  # Long: "September 2018 – January 2021 | San Francisco, CA"
    
    def _find_content_and_right_cells(cells):
        """Find main content cell and right-aligned cell from a row.
        
        Returns (content_cell_index, right_cell_index) or (None, None) if not a two-column pattern.
        """
        if len(cells) == 2:
            return (0, 1)
        
        if len(cells) == 3:
            # Pattern: (empty/tiny) + content + right
            if _is_tiny_cell(cells[0]):
                return (1, 2)
            # Pattern: content + (empty/tiny) + right
            if _is_tiny_cell(cells[1]):
                return (0, 2)
        
        return (None, None)
    
    for tr in elem.findall("w:tr", NAMESPACES):
        cells = tr.findall("w:tc", NAMESPACES)
        
        # Try to detect two-column resume pattern
        content_idx, right_idx = _find_content_and_right_cells(cells)
        
        if content_idx is not None:
            content_cell = cells[content_idx]
            right_cell = cells[right_idx]
            
            left_paragraphs = content_cell.findall("w:p", NAMESPACES)
            right_text = _get_cell_text(right_cell)
            
            # Use pattern-based detection instead of length-only
            if right_text and _is_right_element_pattern(right_text) and len(left_paragraphs) > 0:
                # Store the right column fraction for the generator
                right_col_frac = _get_right_col_fraction(right_text)
                
                # Parse left cell paragraphs with right_element set on first
                first_para = True
                for p in left_paragraphs:
                    parsed = _parse_paragraph(p, styles_map, block_id, rels_map)
                    if parsed is not None:
                        # Set right_element only on first non-empty paragraph
                        if first_para and isinstance(parsed, ParagraphBlock):
                            parsed.right_element = right_text
                            # Store fraction as attribute for generator
                            parsed._right_col_frac = right_col_frac
                            first_para = False
                        blocks.append(parsed)
                        block_id += 1
                continue
        
        # Standard flattening: process each cell's paragraphs
        for tc in cells:
            for p in tc.findall("w:p", NAMESPACES):
                parsed = _parse_paragraph(p, styles_map, block_id, rels_map)
                if parsed is not None:
                    blocks.append(parsed)
                    block_id += 1
    
    return blocks


def _parse_table(elem: etree._Element, block_id: int) -> TableBlock:
    """Parse a table element."""
    column_widths = []
    tblGrid = elem.find("w:tblGrid", NAMESPACES)
    if tblGrid is not None:
        for col in tblGrid.findall("w:gridCol", NAMESPACES):
            w = col.get(f"{{{NAMESPACES['w']}}}w")
            if w:
                column_widths.append(int(w) / 1440.0) # twips to inches

    rows: List[TableRow] = []

    for tr in elem.findall("w:tr", NAMESPACES):
        cells: List[TableCell] = []

        for tc in tr.findall("w:tc", NAMESPACES):
            cell_content: List[InlineSpan] = []
            
            # Parse cell properties for spans
            tcPr = tc.find("w:tcPr", NAMESPACES)
            col_span = 1
            row_span = 1
            
            if tcPr is not None:
                gridSpan = tcPr.find("w:gridSpan", NAMESPACES)
                if gridSpan is not None:
                    val = gridSpan.get(f"{{{NAMESPACES['w']}}}val")
                    if val:
                        col_span = int(val)

            # Parse cell paragraphs - preserve paragraph boundaries
            paragraphs = tc.findall("w:p", NAMESPACES)
            for p_idx, p in enumerate(paragraphs):
                cell_content.extend(_parse_runs(p))
                
                # Insert paragraph separator between w:p elements (not after last)
                if p_idx < len(paragraphs) - 1:
                    cell_content.append(InlineSpan(text="\n", is_paragraph_break=True))

            cells.append(TableCell(content=cell_content, col_span=col_span, row_span=row_span))

        if cells:
            rows.append(TableRow(cells=cells))

    block = TableBlock(
        id=f"b{block_id}",
        block_type="table",
        rows=rows,
        column_widths=column_widths,
    )

    # Calculate complexity score
    block.complexity_score = _calculate_table_complexity(block)

    return block


def _calculate_table_complexity(table: TableBlock) -> int:
    """Calculate table complexity score."""
    score = 0

    for row in table.rows:
        for cell in row.cells:
            if cell.col_span > 1:
                score += 2
            if cell.row_span > 1:
                score += 2

    if table.rows:
        max_cols = max(len(row.cells) for row in table.rows)
        if max_cols > 8:
            score += 3

    if len(table.rows) > 30:
        score += 3

    return score


def _parse_runs(elem: etree._Element, rels_map: Dict[str, str] = None, footnotes_map: Dict[str, List[InlineSpan]] = None) -> List[InlineSpan]:
    """Parse runs within a paragraph, including tab characters, hyperlinks, and content controls.
    
    Args:
        elem: The paragraph element to parse
        rels_map: Relationship map for resolving hyperlink URLs
        footnotes_map: Footnotes map for resolving footnote references
    """
    if rels_map is None:
        rels_map = {}
    if footnotes_map is None:
        footnotes_map = {}
    
    spans: List[InlineSpan] = []
    
    # Iterate through direct children to catch runs, hyperlinks, and content controls
    for child in elem:
        tag = etree.QName(child).localname
        
        if tag == "hyperlink":
            # This is a hyperlink - extract URL from relationship
            rel_id = child.get(f"{{{NAMESPACES['r']}}}id")
            url = rels_map.get(rel_id) if rel_id else None
            
            # Parse runs inside the hyperlink
            for run in child.findall("w:r", NAMESPACES):
                run_spans = _parse_single_run(run, rels_map, url=url, footnotes_map=footnotes_map)
                spans.extend(run_spans)
                
        elif tag == "sdt":
            # Structured Document Tag (Content Control) - common in Microsoft templates
            # Extract runs from sdtContent, which contains the actual text
            sdt_content = child.find("w:sdtContent", NAMESPACES)
            if sdt_content is not None:
                # Recursively parse the sdtContent (handles nested structures)
                spans.extend(_parse_runs(sdt_content, rels_map))
                
        elif tag == "drawing":
            # Drawing object - use helper to handle Absolute Textboxes or fallback
            span = _parse_drawing(child, rels_map)
            if span:
                spans.append(span)

        elif tag == "AlternateContent":
            # Handle Markup Compatibility (e.g. drawings hidden in choice)
            # Look for Choice first, then Fallback
            choice = child.find("mc:Choice", NAMESPACES)
            if choice is not None:
                # Recursively parse content inside Choice
                # Choice usually contains w:drawing directly or w:p
                # But here we are in _parse_runs, so children are run-level
                spans.extend(_parse_runs(choice, rels_map, footnotes_map))
            else:
                fallback = child.find("mc:Fallback", NAMESPACES)
                if fallback is not None:
                    spans.extend(_parse_runs(fallback, rels_map, footnotes_map))

        elif tag == "r":
            # Regular run (not in a hyperlink)
            run_spans = _parse_single_run(child, rels_map, url=None, footnotes_map=footnotes_map)
            spans.extend(run_spans)
            
        else:
            # print(f"DEBUG: Unhandled tag in _parse_runs: {tag}")
            pass

    # Research Fix #6: Coalesce adjacent spans with same formatting
    return _coalesce_spans(spans)


def _parse_single_run(run: etree._Element, rels_map: Dict[str, str], url: Optional[str] = None, footnotes_map: Dict[str, List[InlineSpan]] = None) -> List[InlineSpan]:
    """Parse a single run element into InlineSpans."""
    if footnotes_map is None:
        footnotes_map = {}
    spans = []
    
    # Get run properties (apply to all content in this run)
    rPr = run.find("w:rPr", NAMESPACES)
    bold = False
    italic = False
    underline = False

    if rPr is not None:
        bold = rPr.find("w:b", NAMESPACES) is not None
        italic = rPr.find("w:i", NAMESPACES) is not None
        underline = rPr.find("w:u", NAMESPACES) is not None

    # Parse font information
    font_name = None
    font_size = None
    if rPr is not None:
        rFonts = rPr.find("w:rFonts", NAMESPACES)
        if rFonts is not None:
            # Prefer ascii, then hAnsi, then cs
            font_name = rFonts.get(f"{{{NAMESPACES['w']}}}ascii")
            if not font_name:
                font_name = rFonts.get(f"{{{NAMESPACES['w']}}}hAnsi")
            if not font_name:
                font_name = rFonts.get(f"{{{NAMESPACES['w']}}}cs")
        sz = rPr.find("w:sz", NAMESPACES)
        if sz is None:
            sz = rPr.find("w:szCs", NAMESPACES)
        if sz is not None:
            val = sz.get(f"{{{NAMESPACES['w']}}}val")
            if val:
                try:
                    font_size = float(val) / 2.0
                except ValueError:
                    font_size = None

    # Iterate over children to handle mixed content (text, tabs, breaks) in order
    for child in run:
        tag = etree.QName(child).localname
        
        if tag == "t":
            if child.text:
                spans.append(InlineSpan(
                    text=child.text,
                    bold=bold,
                    italic=italic,
                    underline=underline,
                    font_name=font_name,
                    font_size=font_size,
                    url=url,
                ))
        
        elif tag == "tab":
            spans.append(InlineSpan(text="\t", is_tab=True, font_name=font_name, font_size=font_size))
            
        elif tag == "br":
            spans.append(InlineSpan(text="\n", font_name=font_name, font_size=font_size))
            
        elif tag == "drawing":
            span = _parse_drawing(child, rels_map)
            if span:
                spans.append(span)

        elif tag == "object":
            # OLE Object - may contain v:imagedata for embedded images
            # Look for v:imagedata inside v:shape
            imagedata = child.find(".//v:imagedata", NAMESPACES)
            if imagedata is not None:
                rel_id = imagedata.get(f"{{{NAMESPACES['r']}}}id")
                image_path = rels_map.get(rel_id) if rel_id else None
                
                if image_path:
                    # Get dimensions from w:object attributes (in twips)
                    dxa_orig = child.get(f"{{{NAMESPACES['w']}}}dxaOrig")
                    dya_orig = child.get(f"{{{NAMESPACES['w']}}}dyaOrig")
                    
                    # Convert twips to inches (1 inch = 1440 twips)
                    width = float(dxa_orig) / 1440.0 if dxa_orig else 2.0
                    height = float(dya_orig) / 1440.0 if dya_orig else 2.0
                    
                    fb_block = FigureBlock(
                        id=f"fig_obj_{id(child)}",
                        block_type="figure",
                        image_path=image_path,
                        caption="",
                        width=width,
                        height=height,
                        floating=False
                    )
                    spans.append(InlineSpan(text="", floating_element=fb_block))

        elif tag == "footnoteReference":
            # Footnote reference - look up content in footnotes_map
            fn_id = child.get(f"{{{NAMESPACES['w']}}}id")
            if fn_id and fn_id in footnotes_map:
                spans.append(InlineSpan(text="", footnote_content=footnotes_map[fn_id]))

        elif tag == "AlternateContent":
            # Handle mc:AlternateContent inside runs (common for images)
            choice = child.find("mc:Choice", NAMESPACES)
            if choice is not None:
                # Look for drawing directly inside Choice
                for choice_child in choice:
                    choice_tag = etree.QName(choice_child).localname
                    if choice_tag == "drawing":
                        span = _parse_drawing(choice_child, rels_map)
                        if span:
                            spans.append(span)


    return spans


def _coalesce_spans(spans: List[InlineSpan]) -> List[InlineSpan]:
    """Merge consecutive spans with identical formatting.
    
    This reduces fragmented bold/italic output like:
        \\textbf{He}\\textbf{llo} -> \\textbf{Hello}
    """
    if not spans:
        return spans
    
    coalesced: List[InlineSpan] = []
    current = spans[0]
    
    for span in spans[1:]:
        # Check if styles match and neither is a tab
        same_style = (
            span.bold == current.bold and
            span.italic == current.italic and
            span.underline == current.underline and
            not span.is_tab and
            not current.is_tab and
            not span.floating_element and
            not current.floating_element and
            span.font_name == current.font_name and
            span.font_size == current.font_size and
            span.url == current.url and
            span.footnote_content == current.footnote_content
        )
        
        if same_style:
            # Merge text into current span
            current = InlineSpan(
                text=current.text + span.text,
                bold=current.bold,
                italic=current.italic,
                underline=current.underline,
                font_name=current.font_name,
                font_size=current.font_size,
                url=current.url,
                is_tab=current.is_tab,
                floating_element=current.floating_element,
                is_paragraph_break=current.is_paragraph_break,
                style_ref=current.style_ref,
                footnote_content=current.footnote_content,
            )
        else:
            # Different style or tab - save current and start new
            coalesced.append(current)
            current = span
    
    # Don't forget the last span
    coalesced.append(current)
    
    return coalesced



def _parse_layout_properties(pPr: etree._Element, style_ref: str = None, styles_map: Dict = None) -> Dict:
    """Extract layout properties from paragraph properties with style inheritance.
    
    If line spacing is not explicitly set in pPr, falls back to the paragraph's
    style (and its base styles) to get effective line spacing. This is critical
    for layout grid cells where paragraphs inherit from Normal style.
    """
    layout: Dict = {
        "tab_stops": [],
        "justification": None,
        "spacing_before": 0,
        "spacing_after": 0,
        "line_spacing": None,
        "line_spacing_type": "multiplier",
        "font_family": None,
        "font_size": None,
    }
    
    if pPr is not None:
        # Parse tab stops
        tabs_elem = pPr.find("w:tabs", NAMESPACES)
        if tabs_elem is not None:
            for tab in tabs_elem.findall("w:tab", NAMESPACES):
                tab_val = tab.get(f"{{{NAMESPACES['w']}}}val", "left")
                tab_pos = tab.get(f"{{{NAMESPACES['w']}}}pos", "0")
                layout["tab_stops"].append(TabStop(
                    position=int(tab_pos),
                    alignment=tab_val,
                ))
        
        # Parse justification
        jc = pPr.find("w:jc", NAMESPACES)
        if jc is not None:
            layout["justification"] = jc.get(f"{{{NAMESPACES['w']}}}val")
        
        # Parse explicit spacing from pPr
        spacing = pPr.find("w:spacing", NAMESPACES)
        if spacing is not None:
            before = spacing.get(f"{{{NAMESPACES['w']}}}before")
            after = spacing.get(f"{{{NAMESPACES['w']}}}after")
            line = spacing.get(f"{{{NAMESPACES['w']}}}line")
            line_rule = spacing.get(f"{{{NAMESPACES['w']}}}lineRule")
            
            if before:
                layout["spacing_before"] = int(before) / 20  # twips to points
            if after:
                layout["spacing_after"] = int(after) / 20
                
            if line:
                line_val = int(line)
                if line_rule == "exact":
                    layout["line_spacing_type"] = "exact"
                    layout["line_spacing"] = line_val / 20.0
                elif line_rule == "atLeast":
                    layout["line_spacing_type"] = "atLeast"  
                    layout["line_spacing"] = line_val / 20.0
                else:
                    layout["line_spacing_type"] = "multiplier"
                    layout["line_spacing"] = line_val / 240.0
    
    # STYLE INHERITANCE: Use style chain for spacing and font defaults.
    # Fall back to "Normal" style if no style_ref is provided.
    if styles_map:
        effective_style = style_ref if style_ref else "Normal"
        effective = _resolve_style_spacing(effective_style, styles_map)
        if layout["line_spacing"] is None and effective.get("line_twips"):
            line_twips = effective["line_twips"]
            line_rule = effective.get("lineRule", "auto")
            if line_rule == "exact":
                layout["line_spacing_type"] = "exact"
                layout["line_spacing"] = line_twips / 20.0
            elif line_rule == "atLeast":
                layout["line_spacing_type"] = "atLeast"
                layout["line_spacing"] = line_twips / 20.0
            else:
                layout["line_spacing_type"] = "multiplier"
                layout["line_spacing"] = line_twips / 240.0
        # Also inherit before/after if not set
        if layout["spacing_before"] == 0 and effective.get("before_twips"):
            layout["spacing_before"] = effective["before_twips"] / 20.0
        if layout["spacing_after"] == 0 and effective.get("after_twips"):
            layout["spacing_after"] = effective["after_twips"] / 20.0
        if effective.get("font_family"):
            layout["font_family"] = effective["font_family"]
        if effective.get("font_size"):
            layout["font_size"] = effective["font_size"]

        if layout["font_family"] is None or layout["font_size"] is None or layout["line_spacing"] is None:
            doc_defaults = styles_map.get("_docDefaults", {})
            if layout["font_family"] is None and doc_defaults.get("font_family"):
                layout["font_family"] = doc_defaults["font_family"]
            if layout["font_size"] is None and doc_defaults.get("font_size"):
                layout["font_size"] = doc_defaults["font_size"]
            if layout["line_spacing"] is None and doc_defaults.get("line_twips"):
                line_twips = doc_defaults["line_twips"]
                line_rule = doc_defaults.get("lineRule", "auto")
                if line_rule == "exact":
                    layout["line_spacing_type"] = "exact"
                    layout["line_spacing"] = line_twips / 20.0
                elif line_rule == "atLeast":
                    layout["line_spacing_type"] = "atLeast"
                    layout["line_spacing"] = line_twips / 20.0
                else:
                    layout["line_spacing_type"] = "multiplier"
                    layout["line_spacing"] = line_twips / 240.0

    return layout


def _resolve_style_spacing(style_id: str, styles_map: Dict, visited: set = None) -> Dict:
    """Resolve effective properties (spacing, font) by walking the style inheritance chain."""
    if visited is None:
        visited = set()
    
    if not style_id or style_id in visited:
        return {}
    visited.add(style_id)
    
    style = styles_map.get(style_id, {})
    result = {}
    
    # Get values from this style
    if "line_twips" in style:
        result["line_twips"] = style["line_twips"]
    if "lineRule" in style:
        result["lineRule"] = style["lineRule"]
    if "before_twips" in style:
        result["before_twips"] = style["before_twips"]
    if "after_twips" in style:
        result["after_twips"] = style["after_twips"]
    if "font_family" in style:
        result["font_family"] = style["font_family"]
    if "font_size" in style:
        result["font_size"] = style["font_size"]
    
    # If basedOn exists and we're missing values, inherit
    based_on = style.get("based_on") or style.get("basedOn")
    if based_on:
        parent = _resolve_style_spacing(based_on, styles_map, visited)
        for key in ["line_twips", "lineRule", "before_twips", "after_twips", "font_family", "font_size"]:
            if key not in result and key in parent:
                result[key] = parent[key]
    
    return result


def _emu_to_inch(emu_str: str) -> float:
    """Convert EMUs to inches (1 inch = 914400 EMUs)."""
    try:
        return int(emu_str) / 914400.0
    except (ValueError, TypeError):
        return 0.0


def _parse_drawing(elem: etree._Element, rels_map: Dict[str, str], styles_map: Dict = None) -> Optional[InlineSpan]:
    """Parse a drawing element, handling absolute textboxes or fallback content."""
    if styles_map is None:
        styles_map = {}
    
    # 1. Try to find absolute anchor
    anchor = elem.find(".//wp:anchor", NAMESPACES)
    if anchor is not None:
        # Extract Geometry
        simple_pos = anchor.get("simplePos", "0")
        
        # Position can be simpler or relative. We try to find offset.
        # wp:positionH/wp:posOffset
        pos_h = anchor.find("wp:positionH", NAMESPACES)
        pos_v = anchor.find("wp:positionV", NAMESPACES)
        extent = anchor.find("wp:extent", NAMESPACES)
        
        if pos_h is not None and pos_v is not None and extent is not None:
            # Extract Origins
            origin_x = pos_h.get("relativeFrom", "page")
            origin_y = pos_v.get("relativeFrom", "page")

            # Extract Offsets (convert EMU to Inch)
            # Find posOffset child
            offset_h = pos_h.find("wp:posOffset", NAMESPACES)
            offset_v = pos_v.find("wp:posOffset", NAMESPACES)
            
            # Default to 0 if offset missing (though structure usually guarantees it or align)
            x = _emu_to_inch(offset_h.text) if offset_h is not None else 0.0
            y = _emu_to_inch(offset_v.text) if offset_v is not None else 0.0
            cx = _emu_to_inch(extent.get("cx"))
            cy = _emu_to_inch(extent.get("cy"))
            
            # Extract Content Blocks (recursively)
            blocks = []
            
            # Check for Picture
            pic = anchor.find(".//pic:pic", NAMESPACES)
            if pic is not None:
                # Extract image path from blip
                blip = pic.find(".//a:blip", NAMESPACES)
                if blip is not None:
                    embed_id = blip.get(f"{{{NAMESPACES['r']}}}embed")
                    image_path = rels_map.get(embed_id) if embed_id else None
                    
                    if image_path:
                        # Extract description if available
                        descr = pic.find(".//pic:nvPicPr/pic:cNvPr", NAMESPACES)
                        caption = descr.get("descr", "") if descr is not None else ""
                        
                        fb_block = FigureBlock(
                            id=f"fig_{id(elem)}",
                            block_type="figure",
                            image_path=image_path,
                            caption=caption,
                            width=cx,
                            height=cy,
                            # Positioning
                            floating=True,
                            x=x,
                            y=y,
                            origin_x=origin_x,
                            origin_y=origin_y,
                        )
                        return InlineSpan(text="", floating_element=fb_block)

            # Check for Textbox
            txbx = anchor.find(".//w:txbxContent", NAMESPACES)
            if txbx is not None:
                # We need to call _parse_paragraph for children.
                # Since _parse_paragraph is available in module scope, we can call it.
                # However, avoid circular import issues if logic moves.
                
                # Note: We need a block_id. We can generate one or use a counter/hash.
                # For simplicity here, we'll use 'tb_content' + simple index
                idx = 0
                for child in txbx:
                    tag = etree.QName(child).localname
                    if tag == "p":
                        blocks.append(_parse_paragraph(child, styles_map, f"tb_{id(elem)}_{idx}", rels_map))
                        idx += 1
            
            # If no content found (no pic, no text blocks), return None (Filter Decorative Shapes)
            if not blocks:
                return None
                        
            # Create TextboxBlock
            tb_block = TextboxBlock(
                id=f"tb_{id(elem)}",
                block_type="textbox",
                x=x,
                y=y,
                width=cx,
                height=cy,
                origin_x=origin_x,
                origin_y=origin_y,
                content_blocks=blocks
            )
            
            # Return as anchor span
            return InlineSpan(text="", floating_element=tb_block)
            
    # 2. Fallback: Extract content (Inline or complex group)
    
    # Check for Inline Picture
    pic = elem.find(".//pic:pic", NAMESPACES)
    if pic is not None:
         blip = pic.find(".//a:blip", NAMESPACES)
         if blip is not None:
             embed_id = blip.get(f"{{{NAMESPACES['r']}}}embed")
             image_path = rels_map.get(embed_id) if embed_id else None
             # print(f"DEBUG: Checking drawing blip embed_id={embed_id}, resolved path={image_path}")

             if image_path:
                 descr = pic.find(".//pic:nvPicPr/pic:cNvPr", NAMESPACES)
                 
                 caption = descr.get("descr", "") if descr is not None else ""
                 
                 # Get Extent (size)
                 extent = elem.find(".//wp:extent", NAMESPACES)
                 cx = _emu_to_inch(extent.get("cx")) if extent is not None else 0.0
                 cy = _emu_to_inch(extent.get("cy")) if extent is not None else 0.0

                 fb_block = FigureBlock(
                     id=f"fig_{id(elem)}",
                     block_type="figure",
                     image_path=image_path,
                     caption=caption,
                     width=cx,
                     height=cy,
                     floating=False
                 )
                 # print(f"DEBUG: Returning InlineSpan with fb_block: {fb_block}")
                 return InlineSpan(text="", floating_element=fb_block)
                 
             else:
                 # print(f"DEBUG: Missing image path for embed_id: {embed_id}")
                 pass

    # Check for Inline Textbox
    text_content = []
    for txbx in elem.findall(".//w:txbxContent", NAMESPACES):
        for p in txbx.findall("w:p", NAMESPACES):
            # We can't use _parse_paragraph because we need flat text for InlineSpan
            # So uses _parse_runs directly
            p_spans = _parse_runs(p, rels_map)
            if p_spans:
                text_content.extend([s.text for s in p_spans])
                text_content.append("\n")
    
    if text_content:
        return InlineSpan(text="".join(text_content))
        
    return None


def _analyze_layout_hints(doc: Document) -> None:
    """Analyze document content and styles to update layout hints."""
    # 1. Detect Body Font Family
    # Priority 1: Explicit "Normal" style font (from styles.xml)
    normal_style = doc.styles.get("Normal")
    if normal_style and normal_style.get("font_family"):
        doc.layout_hints.body_font_family = normal_style["font_family"]
        return

    # Priority 2: "Default Paragraph Font"
    default_para = doc.styles.get("Default Paragraph Font")
    if default_para and default_para.get("font_family"):
        doc.layout_hints.body_font_family = default_para["font_family"]
        return

    # Priority 3: Document Defaults (w:docDefaults)
    doc_defaults = doc.styles.get("_docDefaults")
    if doc_defaults and doc_defaults.get("font_family"):
        # Map "Times New Roman" or similar serif fonts if detected
        doc.layout_hints.body_font_family = doc_defaults["font_family"]
        return


def _detect_numbered_sections(doc: Document) -> None:
    """Detect if document should use numbered sections (IEEE/academic style).
    
    Heuristics:
    1. Two-column layout (col_count > 1)
    2. Multiple Heading1 blocks AND at least one has academic content
    3. Academic content keywords: INTRODUCTION, CONCLUSION, APPENDIX, REFERENCES, etc.
    
    Sets doc.use_numbered_sections = True if detected.
    """
    # Check for two-column layout
    is_two_column = False
    if doc.sections:
        is_two_column = doc.sections[0].col_count > 1
    
    # Count Heading1 blocks and check for academic patterns
    heading1_count = 0
    academic_headings = 0
    
    ACADEMIC_PATTERNS = [
        "introduction", "conclusion", "abstract", "appendix", "references",
        "acknowledgment", "acknowledgement", "methodology", "results",
        "discussion", "related work", "background", "evaluation", "experiments",
        "nomenclature", "paper preparation", "template"
    ]
    
    for block in doc.blocks:
        if isinstance(block, HeadingBlock) and block.level == 1:
            heading1_count += 1
            # Extract text content
            text = "".join(span.text for span in block.content).strip().lower()
            # Check for academic patterns
            for pattern in ACADEMIC_PATTERNS:
                if pattern in text:
                    academic_headings += 1
                    break
    
    # Decision: Use numbered sections if:
    # - Two-column layout AND 2+ Heading1 blocks, OR
    # - 3+ academic headings (even in single column)
    if (is_two_column and heading1_count >= 2) or academic_headings >= 3:
        doc.use_numbered_sections = True
