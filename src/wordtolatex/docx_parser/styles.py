"""Styles parser - Parse styles.xml to build a style map."""

from __future__ import annotations

from typing import Dict, Optional
from lxml import etree

# OOXML namespaces
NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def parse_styles(styles_xml: bytes) -> Dict[str, Dict]:
    """Parse styles.xml to build a comprehensive style map.

    Args:
        styles_xml: Raw bytes of styles.xml content.

    Returns:
        Dict mapping style IDs to style information.
    """
    tree = etree.fromstring(styles_xml)
    styles_map: Dict[str, Dict] = {}

    # Parse Default Settings (rPrDefault + pPrDefault)
    doc_defaults = tree.find("w:docDefaults", NAMESPACES)
    if doc_defaults is not None:
        rPrDefault = doc_defaults.find(".//w:rPrDefault/w:rPr", NAMESPACES)
        if rPrDefault is not None:
            rFonts = rPrDefault.find("w:rFonts", NAMESPACES)
            if rFonts is not None:
                font_family = rFonts.get(f"{{{NAMESPACES['w']}}}ascii")
                if not font_family:
                    font_family = rFonts.get(f"{{{NAMESPACES['w']}}}hAnsi")
                
                if font_family:
                    # Store as a reserved style key
                    styles_map["_docDefaults"] = {
                        "name": "_docDefaults",
                        "font_family": font_family
                    }
            sz = rPrDefault.find("w:sz", NAMESPACES)
            if sz is None:
                sz = rPrDefault.find("w:szCs", NAMESPACES)
            if sz is not None:
                val = sz.get(f"{{{NAMESPACES['w']}}}val")
                if val:
                    try:
                        font_size = float(val) / 2.0
                    except ValueError:
                        font_size = None
                    if font_size:
                        styles_map.setdefault("_docDefaults", {"name": "_docDefaults"})
                        styles_map["_docDefaults"]["font_size"] = font_size
        pPrDefault = doc_defaults.find(".//w:pPrDefault/w:pPr", NAMESPACES)
        if pPrDefault is not None:
            spacing = pPrDefault.find("w:spacing", NAMESPACES)
            if spacing is not None:
                before = spacing.get(f"{{{NAMESPACES['w']}}}before")
                after = spacing.get(f"{{{NAMESPACES['w']}}}after")
                line = spacing.get(f"{{{NAMESPACES['w']}}}line")
                line_rule = spacing.get(f"{{{NAMESPACES['w']}}}lineRule")
                styles_map.setdefault("_docDefaults", {"name": "_docDefaults"})
                if before:
                    styles_map["_docDefaults"]["before_twips"] = int(before)
                if after:
                    styles_map["_docDefaults"]["after_twips"] = int(after)
                if line:
                    styles_map["_docDefaults"]["line_twips"] = int(line)
                if line_rule:
                    styles_map["_docDefaults"]["lineRule"] = line_rule

    for style in tree.findall(".//w:style", NAMESPACES):
        style_id = style.get(f"{{{NAMESPACES['w']}}}styleId")
        if not style_id:
            continue

        style_name_elem = style.find("w:name", NAMESPACES)
        style_type = style.get(f"{{{NAMESPACES['w']}}}type")
        based_on_elem = style.find("w:basedOn", NAMESPACES)

        style_info = {
            "name": style_name_elem.get(f"{{{NAMESPACES['w']}}}val") if style_name_elem is not None else style_id,
            "type": style_type,
            "based_on": based_on_elem.get(f"{{{NAMESPACES['w']}}}val") if based_on_elem is not None else None,
        }

        # Extract paragraph properties
        pPr = style.find("w:pPr", NAMESPACES)
        if pPr is not None:
            spacing = pPr.find("w:spacing", NAMESPACES)
            if spacing is not None:
                before = spacing.get(f"{{{NAMESPACES['w']}}}before")
                after = spacing.get(f"{{{NAMESPACES['w']}}}after")
                line = spacing.get(f"{{{NAMESPACES['w']}}}line")
                line_rule = spacing.get(f"{{{NAMESPACES['w']}}}lineRule")
                if before:
                    style_info["spacing_before"] = int(before) / 20  # twips to points
                    style_info["before_twips"] = int(before)
                if after:
                    style_info["spacing_after"] = int(after) / 20
                    style_info["after_twips"] = int(after)
                if line:
                    style_info["line_spacing"] = int(line) / 240  # line units
                    style_info["line_twips"] = int(line)
                if line_rule:
                    style_info["lineRule"] = line_rule

            outlineLvl = pPr.find("w:outlineLvl", NAMESPACES)
            if outlineLvl is not None:
                val = outlineLvl.get(f"{{{NAMESPACES['w']}}}val")
                if val:
                    style_info["outline_level"] = int(val)

        # Extract run properties (fonts)
        rPr = style.find("w:rPr", NAMESPACES)
        if rPr is not None:
            rFonts = rPr.find("w:rFonts", NAMESPACES)
            if rFonts is not None:
                # Prioritize ascii, then hAnsi, then eastAsia
                font_family = rFonts.get(f"{{{NAMESPACES['w']}}}ascii")
                if not font_family:
                    font_family = rFonts.get(f"{{{NAMESPACES['w']}}}hAnsi")
                
                if font_family:
                    style_info["font_family"] = font_family
            sz = rPr.find("w:sz", NAMESPACES)
            if sz is None:
                sz = rPr.find("w:szCs", NAMESPACES)
            if sz is not None:
                val = sz.get(f"{{{NAMESPACES['w']}}}val")
                if val:
                    try:
                        style_info["font_size"] = float(val) / 2.0
                    except ValueError:
                        pass

        # For TableNormal, extract default cell margins
        if style_id == "TableNormal":
            tbl_pr = style.find("w:tblPr", NAMESPACES)
            if tbl_pr is not None:
                cell_mar = tbl_pr.find("w:tblCellMar", NAMESPACES)
                if cell_mar is not None:
                    style_info["tblCellMar"] = _parse_tbl_cell_mar(cell_mar)

        styles_map[style_id] = style_info

    return styles_map


def _parse_tbl_cell_mar(cell_mar_elem) -> Dict[str, int]:
    """Parse w:tblCellMar element to extract cell margins in twips."""
    result = {}
    for side in ("top", "left", "bottom", "right"):
        side_el = cell_mar_elem.find(f"w:{side}", NAMESPACES)
        if side_el is not None:
            w = side_el.get(f"{{{NAMESPACES['w']}}}w")
            if w and w.isdigit():
                result[side] = int(w)
    return result


def get_heading_level(style_id: Optional[str], styles_map: Dict[str, Dict]) -> Optional[int]:
    """Determine heading level from style information.

    Args:
        style_id: The style ID to check.
        styles_map: The parsed styles map.

    Returns:
        Heading level (1-6) or None if not a heading.
    """
    if not style_id:
        return None

    style_info = styles_map.get(style_id, {})
    style_name = style_info.get("name", "")

    # Check for outline level
    if "outline_level" in style_info:
        return style_info["outline_level"] + 1

    # Check style name pattern
    name_lower = style_name.lower()
    if name_lower.startswith("heading"):
        try:
            level_str = style_name.split()[-1]
            return int(level_str)
        except (ValueError, IndexError):
            return 1

    # Check for TOC/Title
    if name_lower == "title" or name_lower == "titledocument":
        return 1

    # Check for common academic style patterns
    # "Head1", "Head 1", "Heading1"
    if name_lower.startswith("head") or name_lower.startswith("heading"):
        # Try to extract number from end
        import re
        match = re.search(r"(\d+)$", name_lower)
        if match:
            return int(match.group(1))
            
    # Academic/Journal specific headers
    # AbsHead -> Abstract -> Level 1
    # AckHead -> Acknowledgments -> Level 1
    # CCSHead -> CCS Concepts -> Level 1
    # KeyWordHead -> Keywords -> Level 1
    if name_lower.endswith("head") or name_lower.endswith("header"):
        if any(x in name_lower for x in ["abs", "ack", "ccs", "keyword", "ref", "meta"]):
            return 1
    
    # Fallback: Check style_id pattern directly (for when styles_map is empty)
    # This handles cases like style_id="Heading1" without styles_map lookup
    import re
    style_id_lower = style_id.lower()
    if style_id_lower.startswith("heading"):
        match = re.search(r"(\d+)$", style_id_lower)
        if match:
            return int(match.group(1))
        return 1  # Default to level 1 if "Heading" without number
            
    return None
