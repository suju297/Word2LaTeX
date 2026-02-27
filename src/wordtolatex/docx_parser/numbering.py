"""Numbering parser - Parse numbering.xml for list definitions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from lxml import etree

# OOXML namespaces
NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def parse_numbering(numbering_xml: bytes) -> Dict[str, Dict]:
    """Parse numbering.xml to build numbering definitions.

    Args:
        numbering_xml: Raw bytes of numbering.xml content.

    Returns:
        Dict mapping numId to numbering information.
    """
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
            lvl_text_elem = lvl.find("w:lvlText", NAMESPACES)
            start_elem = lvl.find("w:start", NAMESPACES)

            num_fmt = num_fmt_elem.get(f"{{{NAMESPACES['w']}}}val") if num_fmt_elem is not None else "bullet"
            lvl_text = lvl_text_elem.get(f"{{{NAMESPACES['w']}}}val") if lvl_text_elem is not None else ""
            start = int(start_elem.get(f"{{{NAMESPACES['w']}}}val")) if start_elem is not None else 1

            levels[int(lvl_id)] = {
                "num_fmt": num_fmt,
                "lvl_text": lvl_text,
                "start": start,
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


def get_list_info(num_id: Optional[str], ilvl: int, numbering_map: Dict[str, Dict]) -> Optional[Dict]:
    """Get list information for a given numId and level.

    Args:
        num_id: The numbering ID.
        ilvl: The indentation level (0-based).
        numbering_map: The parsed numbering map.

    Returns:
        Dict with list info or None if not a list.
    """
    if not num_id or num_id not in numbering_map:
        return None

    num_def = numbering_map[num_id]
    levels = num_def.get("levels", {})

    if ilvl in levels:
        return levels[ilvl]

    return None
