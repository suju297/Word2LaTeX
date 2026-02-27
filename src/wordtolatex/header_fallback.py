"""Header image fallback utilities.

Applies a header-region image fallback using a reference PDF crop, intended
for resume-style documents where branding must be preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import uuid

import fitz

from wordtolatex.calibration import detect_document_type
from wordtolatex.ir import Document, FigureBlock, LayoutGridBlock
from wordtolatex.pdf_cropper import crop_pdf_region


@dataclass(frozen=True)
class HeaderFallbackConfig:
    """Configuration for header image fallback."""

    header_ratio: float = 0.25
    min_confidence: float = 0.6
    max_header_ratio: float = 0.5
    min_header_ratio: float = 0.08
    pad_ratio: float = 0.05
    require_complex_header: bool = True
    min_complex_elements: int = 1


def apply_header_image_fallback(
    doc: Document,
    ref_pdf: Path,
    output_dir: Path,
    config: HeaderFallbackConfig | None = None,
) -> bool:
    """Apply header image fallback for resume-style documents.

    Returns True if a fallback was applied.
    """
    if config is None:
        config = HeaderFallbackConfig()

    if not ref_pdf or not ref_pdf.exists():
        return False

    doc_type = detect_document_type(doc)
    if doc_type != "resume":
        return False

    page_width, page_height = _get_page_size(doc)

    bbox, header_ratio = _detect_header_bbox(ref_pdf, config)
    if bbox is None:
        header_ratio = _estimate_header_ratio_from_grid(doc, page_height)
        if header_ratio is None:
            header_ratio = config.header_ratio
        bbox = (0.0, 0.0, 1.0, header_ratio)
    bbox, header_ratio = _expand_header_bbox(bbox, header_ratio, config)

    if config.require_complex_header:
        if not _has_visual_complexity(ref_pdf, bbox, config):
            return False

    filename = f"header_fallback_{uuid.uuid4().hex[:8]}.png"
    output_path = output_dir / "media" / filename

    if not crop_pdf_region(ref_pdf, 0, list(bbox), output_path):
        return False

    header_height = header_ratio * page_height

    header_block = FigureBlock(
        id=f"header-fallback-{uuid.uuid4().hex[:8]}",
        block_type="figure",
        image_path=f"media/{filename}",
        width=page_width,
        height=header_height,
        floating=True,
        x=0.0,
        y=0.0,
        origin_x="page",
        origin_y="page",
    )
    header_block.render_as_image = False
    header_block.image_fallback_path = f"media/{filename}"
    header_block.policy = "fallback"
    header_block.policy_reason = "HEADER_IMAGE_FALLBACK"

    header_text = _extract_header_text(doc, header_ratio)
    _strip_layout_grid_header_rows(doc, header_ratio, page_height)
    _strip_leading_contact_blocks(doc)
    _insert_header_block(doc, header_block)

    # Avoid duplicate semantic headers if present
    doc.header_blocks = []
    doc.metadata["header_fallback"] = {
        "bbox": bbox,
        "ratio": header_ratio,
        "height_in": header_height,
        "image": f"media/{filename}",
    }
    if header_text:
        doc.metadata["header_text_overlay"] = header_text
    return True


def _detect_header_bbox(
    ref_pdf: Path,
    config: HeaderFallbackConfig,
) -> Tuple[Tuple[float, float, float, float], float] | Tuple[None, None]:
    """Use layout ML header detection when available."""
    try:
        from wordtolatex.layout_ml import analyze_document
    except Exception:
        return None, None

    try:
        analysis = analyze_document(ref_pdf)
    except Exception:
        return None, None

    candidates = [
        r for r in analysis.regions
        if r.label.lower() == "header" and r.confidence >= config.min_confidence
    ]
    candidates = [r for r in candidates if r.bbox[1] <= 0.35]

    if not candidates:
        return None, None

    candidates.sort(key=lambda r: (-r.confidence, r.bbox[1]))
    best = candidates[0]

    y0, y1 = best.bbox[1], best.bbox[3]
    height = y1 - y0

    if height < config.min_header_ratio or height > config.max_header_ratio:
        return None, None

    return (0.0, y0, 1.0, y1), height


def _expand_header_bbox(
    bbox: Tuple[float, float, float, float],
    header_ratio: float,
    config: HeaderFallbackConfig,
) -> Tuple[Tuple[float, float, float, float], float]:
    """Pad header region downward to capture contact lines."""
    x0, y0, x1, y1 = bbox
    padded_y1 = min(1.0, y1 + config.pad_ratio)
    max_y1 = min(1.0, y0 + config.max_header_ratio)
    padded_y1 = min(padded_y1, max_y1)
    padded_ratio = max(header_ratio, padded_y1 - y0)
    return (x0, y0, x1, padded_y1), padded_ratio


def _has_visual_complexity(
    ref_pdf: Path,
    bbox: Tuple[float, float, float, float],
    config: HeaderFallbackConfig,
) -> bool:
    """Detect visual complexity in the header region (shapes/images)."""
    try:
        doc = fitz.open(ref_pdf)
    except Exception:
        return False

    page = doc[0]
    rect = page.rect
    header_rect = fitz.Rect(
        bbox[0] * rect.width,
        bbox[1] * rect.height,
        bbox[2] * rect.width,
        bbox[3] * rect.height,
    )

    complex_count = 0

    # Images in the header region.
    for img in page.get_images():
        for r in page.get_image_rects(img[0]):
            if _rect_overlap_ratio(r, header_rect) > 0.05:
                complex_count += 1
                if complex_count >= config.min_complex_elements:
                    doc.close()
                    return True

    # Drawings (filled or stroked shapes).
    for d in page.get_drawings():
        r = d.get("rect")
        if not r or _rect_overlap_ratio(r, header_rect) <= 0.05:
            continue
        if d.get("fill") is not None or d.get("color") is not None:
            complex_count += 1
            if complex_count >= config.min_complex_elements:
                doc.close()
                return True

    doc.close()
    return False


def _rect_overlap_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    """Return overlap ratio between rectangles (intersection / b)."""
    inter = a & b
    if inter.is_empty:
        return 0.0
    b_area = max(1.0, (b.x1 - b.x0) * (b.y1 - b.y0))
    inter_area = max(0.0, (inter.x1 - inter.x0) * (inter.y1 - inter.y0))
    return inter_area / b_area


def _get_page_size(doc: Document) -> Tuple[float, float]:
    if doc.sections:
        return doc.sections[0].page_width, doc.sections[0].page_height
    return doc.layout_hints.page_width, doc.layout_hints.page_height


def _estimate_header_ratio_from_grid(doc: Document, page_height: float) -> Optional[float]:
    """Estimate header height from layout grid rows."""
    for block in doc.blocks:
        if not isinstance(block, LayoutGridBlock) or not block.rows:
            continue

        # Header-only grid: sum all rows.
        if _is_header_only_grid(block):
            total_twips = sum(row.height_twips or 0 for row in block.rows)
            if total_twips:
                return (total_twips / 1440.0) / page_height
            return None

        # Header rows before first body heading.
        heading_idx = _find_heading_row_index(block)
        if heading_idx is None or heading_idx <= 0:
            return None

        total_twips = sum(
            (row.height_twips or 0) for row in block.rows[:heading_idx]
        )
        if total_twips:
            return (total_twips / 1440.0) / page_height
        return None

    return None


def _strip_layout_grid_header_rows(
    doc: Document,
    header_ratio: float,
    page_height: float,
) -> None:
    """Remove header rows from the first layout grid to avoid duplicates."""
    header_height_in = header_ratio * page_height

    for idx, block in enumerate(list(doc.blocks)):
        if not isinstance(block, LayoutGridBlock) or not block.rows:
            continue
        if _is_header_only_grid(block):
            doc.blocks.pop(idx)
            break
        heading_row_index = _find_heading_row_index(block)
        if heading_row_index is not None and heading_row_index > 0:
            block.rows = block.rows[heading_row_index:]
            break

        row_heights = []
        known_heights = []
        for row in block.rows:
            if row.height_twips:
                height_in = row.height_twips / 1440.0
                row_heights.append(height_in)
                known_heights.append(height_in)
            else:
                row_heights.append(None)

        rows_to_remove = 1
        if known_heights:
            avg_height = sum(known_heights) / len(known_heights)
            total_height = 0.0
            for idx, height in enumerate(row_heights):
                total_height += height if height is not None else avg_height
                if total_height >= header_height_in:
                    rows_to_remove = idx + 1
                    break
        else:
            rows_to_remove = max(1, int(header_ratio * len(block.rows) + 0.5))
            if header_ratio >= 0.2:
                rows_to_remove = max(rows_to_remove, 2)

        rows_to_remove = min(rows_to_remove, len(block.rows) - 1)
        if rows_to_remove > 0:
            block.rows = block.rows[rows_to_remove:]
        break


def _find_heading_row_index(block: LayoutGridBlock) -> Optional[int]:
    """Find the first row containing a resume section heading."""
    heading_terms = (
        "profile",
        "work experience",
        "experience",
        "education",
        "skills",
        "projects",
        "hobbies",
        "certifications",
        "summary",
    )

    for idx, row in enumerate(block.rows):
        text = _row_text(row).lower()
        if any(term in text for term in heading_terms):
            return idx
    return None


def _is_header_only_grid(block: LayoutGridBlock) -> bool:
    """Heuristic: header-only grid contains contact info but no body headings."""
    text = " ".join(_row_text(row) for row in block.rows).lower()

    header_terms = (
        "contact",
        "phone",
        "website",
        "email",
        "linkedin",
        "github",
    )
    body_terms = (
        "profile",
        "work experience",
        "experience",
        "education",
        "skills",
        "projects",
        "hobbies",
        "certifications",
        "summary",
    )

    if any(term in text for term in body_terms):
        return False
    return any(term in text for term in header_terms)


def _row_text(row) -> str:
    parts = []
    for cell in row.cells:
        for block in cell.blocks:
            content = getattr(block, "content", None)
            if not content:
                continue
            parts.append("".join(span.text for span in content if hasattr(span, "text")))
    return " ".join(parts)


def _strip_leading_contact_blocks(doc: Document) -> None:
    """Remove leading contact blocks to avoid duplicates under header image."""
    contact_terms = (
        "contact",
        "phone",
        "website",
        "email",
        "linkedin",
        "github",
    )

    trimmed = []
    stripping = True

    for block in doc.blocks:
        if not stripping:
            trimmed.append(block)
            continue

        if isinstance(block, LayoutGridBlock):
            trimmed.append(block)
            stripping = False
            continue

        text = ""
        content = getattr(block, "content", None)
        if content:
            text = "".join(span.text for span in content if hasattr(span, "text"))

        normalized = text.strip().lower()
        if not normalized:
            continue

        if any(term in normalized for term in contact_terms):
            continue

        trimmed.append(block)
        stripping = False

    doc.blocks = trimmed


def _extract_header_text(doc: Document, header_ratio: float) -> str:
    """Extract contact/header text to overlay on the image fallback."""
    collected = []
    max_blocks = 6

    for block in doc.blocks[:max_blocks]:
        content = getattr(block, "content", None)
        if not content:
            continue
        text = "".join(span.text for span in content if hasattr(span, "text")).strip()
        if text:
            collected.append(text)

    if not collected:
        return ""

    return " \\\\ ".join(_escape_latex(text) for text in collected if text)


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters for overlay text."""
    if not text:
        return ""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


def _insert_header_block(doc: Document, header_block: FigureBlock) -> None:
    """Insert header block before the first layout grid (or at top)."""
    insert_idx = 0
    for idx, block in enumerate(doc.blocks):
        if isinstance(block, LayoutGridBlock):
            insert_idx = idx
            break
    doc.blocks.insert(insert_idx, header_block)
