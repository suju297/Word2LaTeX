"""Media extractor - Extract images from word/media/ folder."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from wordtolatex.ir import Asset


def extract_media(zf: zipfile.ZipFile, output_dir: Path) -> Dict[str, Asset]:
    """Extract all media files from the DOCX.

    Args:
        zf: Open zipfile of the DOCX.
        output_dir: Directory to extract media to.

    Returns:
        Dict mapping relationship IDs to Asset objects.
    """
    media_map: Dict[str, Asset] = {}
    media_dir = output_dir / "media"

    for name in zf.namelist():
        if name.startswith("word/media/"):
            filename = Path(name).name
            output_path = media_dir / filename

            # Ensure media directory exists
            media_dir.mkdir(parents=True, exist_ok=True)

            # Extract file
            with zf.open(name) as src:
                output_path.write_bytes(src.read())

            # Map by filename (we'll resolve rId later)
            media_map[filename] = Asset(
                original_path=name,
                output_path=str(output_path.relative_to(output_dir)),
                asset_type="image",
            )

    return media_map


def get_relationship_targets(zf: zipfile.ZipFile) -> Dict[str, str]:
    """Parse document relationships to map rId to targets.

    Args:
        zf: Open zipfile of the DOCX.

    Returns:
        Dict mapping rId to target paths.
    """
    from lxml import etree

    rels_map: Dict[str, str] = {}

    rels_path = "word/_rels/document.xml.rels"
    if rels_path not in zf.namelist():
        return rels_map

    rels_xml = zf.read(rels_path)
    tree = etree.fromstring(rels_xml)

    ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
    for rel in tree.findall(".//r:Relationship", ns):
        rel_id = rel.get("Id")
        target = rel.get("Target")
        if rel_id and target:
            rels_map[rel_id] = target

    return rels_map
