"""Dynamic LaTeX Generator - Template-free IR-driven generation with PDF oracle.

This module generates LaTeX programmatically from Document IR without using
fixed templates. It analyzes document features and builds appropriate LaTeX
structure dynamically, using PDF reference for layout calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import re
from collections import defaultdict

from wordtolatex.ir import (
    Document, Block, HeadingBlock, ParagraphBlock, ListBlock,
    TableBlock, FigureBlock, InlineSpan, TextboxBlock,
    TableBlock, FigureBlock, InlineSpan, TextboxBlock,
    LayoutGridBlock, GridCell, GridRow, ListItem
)

from wordtolatex.utils import escape_latex

# Import layout pattern detection for PDF-based sidebar detection
try:
    from wordtolatex.layout_oracle import detect_layout_pattern, LayoutPattern, LayoutPatternInfo
    HAS_LAYOUT_ORACLE = True
except ImportError:
    HAS_LAYOUT_ORACLE = False
    LayoutPattern = None
    LayoutPatternInfo = None


# =============================================================================
# Document Feature Detection
# =============================================================================

@dataclass
class DocumentFeatures:
    """Detected features for generation decisions."""
    
    # Document class selection
    doc_class: str = "article"          # article, IEEEtran, acmart, letter
    class_options: List[str] = field(default_factory=list)  # 10pt, twocolumn, etc.
    
    # Layout
    column_count: int = 1
    page_width: float = 8.5             # inches
    page_height: float = 11.0
    margin_top: float = 1.0
    margin_bottom: float = 1.0
    margin_left: float = 1.0
    margin_right: float = 1.0
    
    # Typography
    font_family: str = "Arial"
    font_size: float = 11.0
    line_spacing: float = 1.15
    heading_style: str = "unnumbered"   # numbered, unnumbered, ruled
    heading_font_sizes: Dict[int, float] = field(default_factory=dict)
    
    # Content features
    has_math: bool = False
    has_tables: bool = False
    has_figures: bool = False
    has_floating_elements: bool = False
    has_hyperlinks: bool = False
    has_footnotes: bool = False
    has_lists: bool = False
    needs_ltablex: bool = False         # Large tables needing page breaks
    
    # Detected document type
    doc_type: str = "generic"           # academic_twocol, resume, letter, generic
    
    # PDF-detected layout pattern (for sidebar/multi-column rendering)
    layout_pattern: Optional[Any] = None  # LayoutPatternInfo from PDF analysis


@dataclass
class RenderContext:
    """Context passed through block rendering for stateful awareness."""
    
    page_width: float = 8.5
    text_width: float = 6.5             # page_width - margins
    column_width: float = 6.5           # For single column
    
    in_list: bool = False
    in_table: bool = False
    section_depth: int = 0
    previous_block_type: Optional[str] = None
    
    # Margins for tab stop calculations
    margin_left: float = 1.0
    margin_top: float = 1.0
    
    # Use numbered sections
    use_numbered_sections: bool = False
    
    # Features (for layout awareness)
    features: Optional[DocumentFeatures] = None


@dataclass
class PDFLayout:
    """Layout metrics extracted from reference PDF."""
    
    page_size: Tuple[float, float] = (8.5, 11.0)
    margins: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # t,b,l,r
    body_font_size: float = 11.0
    heading_font_sizes: Dict[int, float] = field(default_factory=dict)
    line_spacing: float = 1.15
    column_count: int = 1
    column_spacing: float = 0.25        # inches


# =============================================================================
# Feature Detection
# =============================================================================

def analyze_document(doc: Document, ref_pdf: Optional[Path] = None) -> DocumentFeatures:
    """Analyze document IR and optional PDF to detect features."""
    features = DocumentFeatures()
    
    # Extract from OOXML-derived IR
    _analyze_ir_features(doc, features)
    
    # Extract from PDF if available
    if ref_pdf and ref_pdf.exists():
        _analyze_pdf_features(ref_pdf, features, doc)
    
    # Determine document class and heading style
    _determine_doc_class(doc, features)
    
    return features


def _analyze_ir_features(doc: Document, features: DocumentFeatures) -> None:
    """Extract features from document IR."""
    
    # Page setup from sections
    if doc.sections:
        sec = doc.sections[0]
        features.page_width = sec.page_width
        features.page_height = sec.page_height
        features.margin_top = sec.margin_top
        features.margin_bottom = sec.margin_bottom
        features.margin_left = sec.margin_left
        features.margin_right = sec.margin_right
        features.column_count = sec.col_count
    
    # Layout hints
    hints = doc.layout_hints
    features.font_family = hints.body_font_family
    features.font_size = hints.suggested_font_size or hints.body_font_size
    
    # Apply global squeeze factor if present
    squeeze = hints.suggested_squeeze or 1.0
    features.line_spacing = hints.line_spacing * squeeze
    
    # Scan blocks for features
    layout_grid_count = 0
    large_grid_rows = 0
    heading_size_weights: Dict[int, Dict[float, int]] = defaultdict(lambda: defaultdict(int))

    content_blocks = list(doc.blocks)
    header_footer_blocks = list(doc.header_blocks) + list(doc.footer_blocks)
    for block in content_blocks:
        if isinstance(block, TableBlock):
            features.has_tables = True
        elif isinstance(block, ListBlock):
            features.has_lists = True
        elif isinstance(block, FigureBlock):
            features.has_figures = True
            if getattr(block, 'floating', False):
                features.has_floating_elements = True
        elif isinstance(block, TextboxBlock):
            features.has_floating_elements = True
        elif isinstance(block, LayoutGridBlock):
            layout_grid_count += 1
            large_grid_rows = max(large_grid_rows, len(block.rows))
            features.has_tables = True  # Need tabularx for layout grids
        
        # Check for hyperlinks in content
        if hasattr(block, 'content'):
            for span in block.content:
                if span.url:
                    features.has_hyperlinks = True
                if span.footnote_content:
                    features.has_footnotes = True

    for block in header_footer_blocks:
        if isinstance(block, TableBlock):
            features.has_tables = True
        elif isinstance(block, ListBlock):
            features.has_lists = True
        elif isinstance(block, FigureBlock):
            features.has_figures = True
            if getattr(block, 'floating', False):
                features.has_floating_elements = True
        elif isinstance(block, TextboxBlock):
            features.has_floating_elements = True
        elif isinstance(block, LayoutGridBlock):
            features.has_tables = True
        if hasattr(block, 'content'):
            for span in block.content:
                if span.url:
                    features.has_hyperlinks = True
                if span.footnote_content:
                    features.has_footnotes = True

    _collect_heading_sizes(doc.blocks, heading_size_weights)
    if heading_size_weights:
        features.heading_font_sizes = {
            level: max(sizes.items(), key=lambda item: (item[1], item[0]))[0]
            for level, sizes in heading_size_weights.items()
            if sizes
        }
    
    # Large layout grids need ltablex for page breaks
    if large_grid_rows > 8:
        features.needs_ltablex = True


def _collect_heading_sizes(blocks: List[Block], weights: Dict[int, Dict[float, int]]) -> None:
    """Collect weighted heading font sizes from nested blocks."""
    for block in blocks:
        if isinstance(block, HeadingBlock) and block.font_size:
            weight = _text_weight(block.content)
            if weight > 0:
                key = round(block.font_size, 1)
                weights[block.level][key] += weight
        if isinstance(block, LayoutGridBlock):
            for row in block.rows:
                for cell in row.cells:
                    _collect_heading_sizes(cell.blocks, weights)
        elif isinstance(block, TextboxBlock):
            _collect_heading_sizes(block.content_blocks, weights)


def _text_weight(spans: List[InlineSpan]) -> int:
    """Estimate weight based on non-space characters."""
    text = "".join(s.text for s in spans if getattr(s, "text", None))
    return sum(1 for c in text if not c.isspace())


def _analyze_pdf_features(ref_pdf: Path, features: DocumentFeatures, doc: Optional[Document] = None) -> None:
    """Extract layout metrics from reference PDF."""
    try:
        from wordtolatex.pdf_metrics import extract_metrics
        metrics = extract_metrics(ref_pdf)
        
        # Override with measured values
        if metrics.avg_font_size > 0:
            features.font_size = metrics.avg_font_size
        
        # Estimate line spacing from line height / font size
        # BUT: If manual squeeze calibration is active, do NOT overwrite with PDF metrics
        is_squeezed = False
        if doc and doc.layout_hints:
             # Check if squeeze is active (not 1.0)
             squeeze = doc.layout_hints.suggested_squeeze
             if squeeze and squeeze < 0.99:
                 is_squeezed = True
        
        if not is_squeezed and metrics.avg_line_height > 0 and metrics.avg_font_size > 0:
            features.line_spacing = metrics.avg_line_height / metrics.avg_font_size
            
    except Exception:
        pass  # Use IR-derived values if PDF analysis fails
    
    # Detect layout pattern (sidebar, two-column, etc.)
    if HAS_LAYOUT_ORACLE:
        try:
            pattern_info = detect_layout_pattern(ref_pdf)
            features.layout_pattern = pattern_info
        except Exception:
            pass  # Layout detection failed, use defaults


def _determine_doc_class(doc: Document, features: DocumentFeatures) -> None:
    """Determine document class and heading style from content analysis."""
    override = doc.metadata.get("slm_doc_type") if hasattr(doc, "metadata") else None
    layout_style = doc.metadata.get("slm_layout_style") if hasattr(doc, "metadata") else None
    if override:
        if layout_style == "two_column" and override == "academic_singlecol":
            override = "academic_twocol"

        if override == "academic_twocol":
            features.doc_type = "academic_twocol"
            features.doc_class = "article"
            features.class_options = ["10pt"]
            features.heading_style = "numbered"
            features.column_count = max(features.column_count, 2)
            if features.layout_pattern is None and HAS_LAYOUT_ORACLE and LayoutPatternInfo:
                features.layout_pattern = LayoutPatternInfo(
                    pattern=LayoutPattern.TWO_COLUMN_EQUAL,
                    column_gap_ratio=0.04,
                )
        elif override == "academic_singlecol":
            features.doc_type = "academic_singlecol"
            features.doc_class = "article"
            features.class_options = ["11pt"]
            features.heading_style = "numbered"
        elif override == "resume":
            features.doc_type = "resume"
            features.doc_class = "article"
            features.class_options = ["10pt"]
            features.heading_style = "ruled"
        else:
            features.doc_type = "generic"
            features.doc_class = "article"
            features.class_options = ["11pt"]
            features.heading_style = "unnumbered"

        doc.use_numbered_sections = (features.heading_style == "numbered")
        return

    # Count heading patterns
    academic_score = 0
    resume_score = 0
    heading_count = 0
    
    ACADEMIC_PATTERNS = [
        "introduction", "conclusion", "abstract", "appendix", "references",
        "acknowledgment", "methodology", "results", "discussion", "background",
        "related work", "evaluation", "experiments", "literature review",
    ]
    
    RESUME_PATTERNS = [
        "education", "experience", "skills", "projects", "summary", "objective",
        "certifications", "awards", "publications", "contact", "employment",
    ]
    
    for block in doc.blocks:
        if isinstance(block, HeadingBlock):
            heading_count += 1
            text = "".join(s.text for s in block.content).lower().strip()
            
            for pattern in ACADEMIC_PATTERNS:
                if pattern in text:
                    academic_score += 2 if block.level == 1 else 1
                    break
            
            for pattern in RESUME_PATTERNS:
                if pattern in text:
                    resume_score += 2 if block.level == 1 else 1
                    break
    
    # Count layout grids (common in resumes)
    layout_grids = sum(1 for b in doc.blocks if isinstance(b, LayoutGridBlock))
    if layout_grids >= 2:
        resume_score += 3
    
    # Determine type and class
    if features.column_count > 1 and academic_score >= 2:
        features.doc_type = "academic_twocol"
        features.doc_class = "article"  # Could use IEEEtran if available
        features.class_options = ["10pt"] # Remove 'twocolumn', we handle it manually with multicol
        features.heading_style = "numbered"
        
    elif academic_score >= 4:
        features.doc_type = "academic_singlecol"
        features.doc_class = "article"
        features.class_options = ["11pt"]
        features.heading_style = "numbered"
        
    elif resume_score >= 3 or layout_grids >= 2:
        features.doc_type = "resume"
        features.doc_class = "article"
        features.class_options = ["10pt"]
        features.heading_style = "ruled"  # Section with titlerule
        
    else:
        features.doc_type = "generic"
        features.doc_class = "article"
        features.class_options = ["11pt"]
        features.heading_style = "unnumbered"
    
    # Set IR flag for numbered sections
    doc.use_numbered_sections = (features.heading_style == "numbered")


# =============================================================================
# Preamble Builder
# =============================================================================

def build_preamble(features: DocumentFeatures) -> str:
    """Build LaTeX preamble dynamically based on document features."""
    lines = []
    
    # Document class
    options = ", ".join(features.class_options) if features.class_options else ""
    if options:
        lines.append(f"\\documentclass[{options}]{{{features.doc_class}}}")
    else:
        lines.append(f"\\documentclass{{{features.doc_class}}}")
    
    lines.append("")
    lines.append("% --- Packages ---")
    
    # Core packages (always needed)
    lines.append("\\usepackage{fontspec}")
    lines.append("\\usepackage{geometry}")
    lines.append("\\usepackage{setspace}")
    lines.append("\\usepackage{titlesec}")
    lines.append("\\usepackage{titlesec}")
    lines.append("\\usepackage{enumitem}")
    lines.append("\\usepackage{graphicx}")
    lines.append("\\usepackage{caption}")
    
    # Conditional packages
    if features.has_hyperlinks:
        lines.append("\\usepackage[hidelinks]{hyperref}")
        lines.append("\\usepackage{xcolor}")
    
    if features.has_tables or features.needs_ltablex:
        if features.needs_ltablex:
            lines.append("\\usepackage{ltablex}")
            lines.append("\\keepXColumns")
        else:
            lines.append("\\usepackage{tabularx}")
        lines.append("\\usepackage{array}")
    
    if features.has_figures:
        lines.append("\\usepackage{graphicx}")
        lines.append("\\usepackage{float}")
    
    if features.has_floating_elements:
        lines.append("\\usepackage[absolute,overlay]{textpos}")
        lines.append("\\setlength{\\TPHorizModule}{1in}")
        lines.append("\\setlength{\\TPVertModule}{1in}")
    
    if features.has_math:
        lines.append("\\usepackage{amsmath}")
        lines.append("\\usepackage{amssymb}")

    # Layout packages
    lines.append("\\usepackage{multicol}")
    lines.append("\\setlength{\\columnsep}{0.2in}")
    
    # Additional layout packages
    lines.append("\\usepackage{ragged2e}")
    lines.append("\\usepackage{tabto}")
    
    # Font setting
    lines.append("")
    lines.append("% --- Font ---")
    font_cmd = _get_font_command(features.font_family)
    lines.append(font_cmd)
    
    # Geometry
    lines.append("")
    lines.append("% --- Page Geometry ---")
    lines.append("\\geometry{")
    lines.append(f"    paperwidth={features.page_width}in,")
    lines.append(f"    paperheight={features.page_height}in,")
    lines.append(f"    top={features.margin_top}in,")
    lines.append(f"    bottom={features.margin_bottom}in,")
    lines.append(f"    left={features.margin_left}in,")
    lines.append(f"    right={features.margin_right}in,")
    lines.append("    headheight=0pt,")
    lines.append("    headsep=0pt,")
    lines.append("    footskip=0pt")
    lines.append("}")
    
    # Spacing and formatting
    lines.append("")
    lines.append("% --- Spacing ---")
    lines.append(f"\\setstretch{{{features.line_spacing:.2f}}}")
    lines.append("\\setlength{\\parindent}{0pt}")
    lines.append("\\setlength{\\parskip}{0pt}")
    
    # List formatting
    lines.append("\\setlist[itemize]{leftmargin=*, topsep=1pt, itemsep=0pt, parsep=0pt}")
    lines.append("\\setlist[enumerate]{leftmargin=*, topsep=1pt, itemsep=0pt, parsep=0pt}")
    
    # Section formatting based on heading style
    lines.append("")
    lines.append("% --- Section Formatting ---")
    section_size, section_baseline = _resolve_heading_size(features, 1, 12.0)
    subsection_size, subsection_baseline = _resolve_heading_size(features, 2, 11.0)
    subsubsection_size, subsubsection_baseline = _resolve_heading_size(features, 3, 10.0)
    if features.heading_style == "numbered":
        lines.append("\\renewcommand{\\thesection}{\\arabic{section}}")
        lines.append("\\renewcommand{\\thesubsection}{\\thesection.\\arabic{subsection}}")
        lines.append(
            f"\\titleformat{{\\section}}{{\\bfseries\\fontsize{{{section_size:.1f}}}{{{section_baseline:.1f}}}\\selectfont}}{{\\thesection}}{{0.5em}}{{}}"
        )
        lines.append(
            f"\\titleformat{{\\subsection}}{{\\bfseries\\fontsize{{{subsection_size:.1f}}}{{{subsection_baseline:.1f}}}\\selectfont}}{{\\thesubsection}}{{0.5em}}{{}}"
        )
        lines.append(
            f"\\titleformat{{\\subsubsection}}{{\\bfseries\\fontsize{{{subsubsection_size:.1f}}}{{{subsubsection_baseline:.1f}}}\\selectfont}}{{\\thesubsubsection}}{{0.5em}}{{}}"
        )
    elif features.heading_style == "ruled":
        lines.append(
            f"\\titleformat{{\\section}}{{\\bfseries\\fontsize{{{section_size:.1f}}}{{{section_baseline:.1f}}}\\selectfont}}{{}}{{0em}}{{}}[\\titlerule]"
        )
        lines.append(
            f"\\titleformat{{\\subsection}}{{\\bfseries\\fontsize{{{subsection_size:.1f}}}{{{subsection_baseline:.1f}}}\\selectfont}}{{}}{{0em}}{{}}"
        )
        lines.append(
            f"\\titleformat{{\\subsubsection}}{{\\bfseries\\fontsize{{{subsubsection_size:.1f}}}{{{subsubsection_baseline:.1f}}}\\selectfont}}{{}}{{0em}}{{}}"
        )
    else:  # unnumbered
        lines.append(
            f"\\titleformat{{\\section}}{{\\bfseries\\fontsize{{{section_size:.1f}}}{{{section_baseline:.1f}}}\\selectfont}}{{}}{{0em}}{{}}"
        )
        lines.append(
            f"\\titleformat{{\\subsection}}{{\\bfseries\\fontsize{{{subsection_size:.1f}}}{{{subsection_baseline:.1f}}}\\selectfont}}{{}}{{0em}}{{}}"
        )
        lines.append(
            f"\\titleformat{{\\subsubsection}}{{\\bfseries\\fontsize{{{subsubsection_size:.1f}}}{{{subsubsection_baseline:.1f}}}\\selectfont}}{{}}{{0em}}{{}}"
        )
    
    lines.append("\\titlespacing*{\\section}{0pt}{4pt}{2pt}")
    lines.append("\\titlespacing*{\\subsection}{0pt}{3pt}{1pt}")
    
    # Misc settings
    lines.append("")
    lines.append("% --- Misc ---")
    lines.append("\\pagenumbering{gobble}")
    lines.append("\\hyphenpenalty=10000")
    lines.append("\\exhyphenpenalty=10000")
    lines.append("\\sloppy")
    
    return "\n".join(lines)


def _resolve_heading_size(features: DocumentFeatures, level: int, default_size: float) -> Tuple[float, float]:
    """Resolve heading font size and baseline for a heading level."""
    size = features.heading_font_sizes.get(level)
    if not size:
        size = default_size
    return size, size * 1.2


def _map_font_family(font_family: str) -> str:
    """Map common fonts to local/system equivalents."""
    if not font_family:
        return "TeX Gyre Heros"

    lower = font_family.strip().lower()
    font_dir = Path.cwd() / "fonts"
    has_carlito = (font_dir / "Carlito-Regular.ttf").exists()

    if "calibri" in lower:
        return "Carlito" if has_carlito else "TeX Gyre Heros"
    if "courier" in lower or "mono" in lower:
        return "TeX Gyre Cursor"
    if "helvetica" in lower or "arial" in lower or "sans" in lower:
        return "TeX Gyre Heros"
    if "times" in lower or "serif" in lower:
        return "TeX Gyre Termes"
    if "verdana" in lower or "tahoma" in lower or "trebuchet" in lower:
        return "TeX Gyre Heros"

    return font_family


def _escape_caption(text: str) -> str:
    """Collapse whitespace and escape caption text for \\caption."""
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return escape_latex(collapsed)


def _get_font_command(font_family: str) -> str:
    """Generate font setting command."""
    mapped = _map_font_family(font_family)

    # Check for local Carlito
    font_dir = Path.cwd() / "fonts"
    if mapped == "Carlito" and (font_dir / "Carlito-Regular.ttf").exists():
        abs_path = str(font_dir).replace("\\", "/") + "/"
        return (
            f"\\setmainfont{{Carlito}}[\n"
            f"    Path={abs_path},\n"
            "    Extension=.ttf,\n"
            "    UprightFont=*-Regular,\n"
            "    BoldFont=*-Bold,\n"
            "    ItalicFont=*-Italic,\n"
            "    BoldItalicFont=*-BoldItalic\n"
            "]"
        )

    return f"\\setmainfont{{{mapped}}}"


def _get_inline_fontspec(font_family: str) -> str:
    """Generate inline \\fontspec for font overrides."""
    mapped = _map_font_family(font_family)
    font_dir = Path.cwd() / "fonts"

    if mapped == "Carlito" and (font_dir / "Carlito-Regular.ttf").exists():
        abs_path = str(font_dir).replace("\\", "/") + "/"
        return (
            "\\fontspec[\n"
            f"    Path={abs_path},\n"
            "    Extension=.ttf,\n"
            "    UprightFont=*-Regular,\n"
            "    BoldFont=*-Bold,\n"
            "    ItalicFont=*-Italic,\n"
            "    BoldItalicFont=*-BoldItalic\n"
            "]{Carlito}"
        )

    return f"\\fontspec{{{mapped}}}"


def _get_font_wrappers(block: Block, context: RenderContext) -> Tuple[str, str]:
    """Return prefix/suffix for font overrides."""
    font_cmds: List[str] = []

    base_font = None
    if context.features:
        base_font = _map_font_family(context.features.font_family)

    if getattr(block, "font_family", None):
        mapped = _map_font_family(block.font_family)
        if not base_font or mapped != base_font:
            font_cmds.append(_get_inline_fontspec(mapped))

    if getattr(block, "font_size", None):
        font_size = block.font_size
        baseline = font_size * 1.2
        line_spacing_type = getattr(block, "line_spacing_type", "multiplier")
        line_spacing = getattr(block, "line_spacing", None)
        if line_spacing_type in ("exact", "atLeast") and line_spacing:
            baseline = line_spacing
        elif line_spacing_type == "multiplier" and line_spacing:
            baseline = font_size * line_spacing
        font_cmds.append(f"\\fontsize{{{font_size:.1f}}}{{{baseline:.1f}}}\\selectfont")

    if not font_cmds:
        return "", ""

    return "{" + " ".join(font_cmds) + " ", "}"


# =============================================================================
# Block Rendering
# =============================================================================

def render_document(doc: Document, features: DocumentFeatures) -> str:
    """Render entire document body."""
    context = RenderContext(
        page_width=features.page_width,
        text_width=features.page_width - features.margin_left - features.margin_right,
        column_width=(features.page_width - features.margin_left - features.margin_right) / features.column_count,
        margin_left=features.margin_left,
        margin_top=features.margin_top,
        use_numbered_sections=(features.heading_style == "numbered"),
        features=features,
    )
    
    lines = []
    lines.append("")
    lines.append("\\begin{document}")
    
    # Apply font size if different from class default
    if features.font_size and features.font_size != 11.0:
        baseline = features.font_size * 1.2
        lines.append(f"\\fontsize{{{features.font_size:.1f}}}{{{baseline:.1f}}}\\selectfont")
    
    lines.append("")

    header_content = _render_header_footer_blocks(doc.header_blocks, context)
    if header_content:
        lines.append(header_content)
        lines.append("")
    
    # Check for sidebar layout from PDF pattern detection
    is_sidebar = (features.layout_pattern and 
                  HAS_LAYOUT_ORACLE and 
                  features.layout_pattern.pattern in (LayoutPattern.SIDEBAR_LEFT, LayoutPattern.SIDEBAR_RIGHT))
    
    if is_sidebar:
        # Use minipage-based sidebar layout
        sidebar_content = _render_sidebar_layout(doc, features, context)
        lines.append(sidebar_content)
    elif (features.layout_pattern and 
          features.layout_pattern.pattern == LayoutPattern.TWO_COLUMN_EQUAL):
        # Use multicol for balanced two-column
        # Set separation based on detected gap
        if features.layout_pattern.column_gap_ratio > 0:
            lines.append(f"\\setlength{{\\columnsep}}{{{features.layout_pattern.column_gap_ratio:.3f}\\linewidth}}")
        
        # Check for Abstract to split Header vs Body
        split_idx = 0
        for i, block in enumerate(doc.blocks[:25]): # Scan first 25 blocks
            # Crude text extraction
            text = ""
            if hasattr(block, 'text'): text = block.text
            elif hasattr(block, 'content'): text = "".join([s.text for s in block.content if hasattr(s, 'text')])
            
            if text and ("abstract" in text.lower() or "introduction" in text.lower()):
                split_idx = i
                break
                
        # Render Header (Full Width)
        for block in doc.blocks[:split_idx]:
             rendered = render_block(block, context)
             if rendered:
                 lines.append(rendered)
                 lines.append("")
             context.previous_block_type = block.block_type

        # Render Body (Multicol)
        lines.append("\\begin{multicols}{2}")
        for block in doc.blocks[split_idx:]:
            rendered = render_block(block, context)
            if rendered:
                lines.append(rendered)
                lines.append("")
            context.previous_block_type = block.block_type
            
        lines.append("\\end{multicols}")
    else:
        # Standard block-by-block rendering
        for i, block in enumerate(doc.blocks):
            rendered = render_block(block, context)
            if rendered:
                lines.append(rendered)
                lines.append("")
            
            # Update context
            context.previous_block_type = block.block_type
    
    footer_content = _render_header_footer_blocks(doc.footer_blocks, context)
    if footer_content:
        lines.append("\\vfill")
        lines.append(footer_content)
        lines.append("")

    lines.append("\\end{document}")
    
    return "\n".join(lines)


def _wrap_justification(rendered: str, justification: Optional[str]) -> str:
    """Wrap rendered content in a justification environment when specified."""
    if not justification:
        return rendered
    env_map = {
        "right": "FlushRight",
        "center": "Center",
        "left": "FlushLeft",
        "both": "justify",
        "justify": "justify",
    }
    env = env_map.get(str(justification).lower())
    if not env:
        return rendered
    return f"\\begin{{{env}}}\n{rendered}\n\\end{{{env}}}"


def _render_header_footer_blocks(blocks: List[Block], context: RenderContext) -> str:
    """Render header/footer blocks as editable content."""
    rendered_blocks = []
    for block in blocks:
        rendered = render_block(block, context)
        if not rendered:
            continue
        rendered = _wrap_justification(rendered, getattr(block, "justification", None))
        rendered_blocks.append(rendered)
    if not rendered_blocks:
        return ""
    return "\n\n".join(rendered_blocks)


def _render_sidebar_layout(doc: Document, features: DocumentFeatures, context: RenderContext) -> str:
    """Render document using minipage-based sidebar + main layout.
    
    This is used for resume-style documents with a narrow left sidebar.
    """
    pattern = features.layout_pattern
    
    # Debug detected pattern
    if pattern:
        print(f"   [Dynamic] Sidebar Layout Detected: {pattern.pattern} (Sidebar: {pattern.sidebar_width_ratio:.3f}, Main: {pattern.main_width_ratio:.3f})")
    
    # Get width ratios from pattern detection (or use defaults)
    if pattern:
        # Resumes typically have Narrow Left, Wide Right
        # Use min/max to enforce this heuristic regardless of detection orientation
        r1 = pattern.sidebar_width_ratio
        r2 = pattern.main_width_ratio

        # Heuristic: Left column is usually narrower
        sidebar_ratio = min(r1, r2)
        main_ratio = max(r1, r2)

        # Use detected gap if reasonable
        gap_ratio = pattern.column_gap_ratio if pattern.column_gap_ratio > 0 else 0.04

        # Ensure they fit
        if sidebar_ratio + main_ratio + gap_ratio > 1.0:
             # Scale down proportionally if overflow
             scale = 0.96 / (sidebar_ratio + main_ratio)
             sidebar_ratio *= scale
             main_ratio *= scale
    else:
        sidebar_ratio = 0.28
        main_ratio = 0.68
    
    # Clamp to reasonable values
    sidebar_ratio = max(0.15, min(0.45, sidebar_ratio))
    main_ratio = max(0.45, min(0.85, main_ratio))
    
    # Separate header fallback blocks so spacing applies to both columns.
    header_blocks = []
    content_blocks = []
    for block in doc.blocks:
        if isinstance(block, FigureBlock) and getattr(block, "policy_reason", "") == "HEADER_IMAGE_FALLBACK":
            header_blocks.append(block)
        else:
            content_blocks.append(block)

    # Classify blocks into sidebar vs main content
    # For LayoutGrids, use column index from the grid structure
    # For other blocks, use heading patterns as hints
    
    sidebar_blocks = []
    main_blocks = []
    
    # Simple heuristic: Use LayoutGrid column structure if present
    # Otherwise, all blocks go to main

    sidebar_on_right = False
    sidebar_index_hint = None
    main_index_hint = None
    hint_content_cols = None
    for block in content_blocks:
        if not isinstance(block, LayoutGridBlock) or not block.col_widths_twips:
            continue
        if len(block.col_widths_twips) < 2:
            continue

        content_cols = set()
        for row in block.rows:
            for idx, cell in enumerate(row.cells):
                if cell.blocks:
                    content_cols.add(idx)

        if len(content_cols) >= 2:
            widths = block.col_widths_twips
            left_idx = min(content_cols)
            right_idx = max(content_cols)

            if pattern and pattern.pattern == LayoutPattern.SIDEBAR_LEFT:
                sidebar_index_hint = left_idx
                main_index_hint = right_idx
                sidebar_on_right = False
            else:
                sidebar_index_hint = right_idx
                main_index_hint = left_idx
                sidebar_on_right = True

            hint_content_cols = content_cols
            content_total = 0
            if left_idx < len(widths):
                content_total += widths[left_idx]
            if right_idx < len(widths):
                content_total += widths[right_idx]
            if content_total:
                sidebar_ratio = widths[sidebar_index_hint] / content_total
                main_ratio = widths[main_index_hint] / content_total
            break

    if sidebar_index_hint is None and pattern and pattern.pattern == LayoutPattern.SIDEBAR_RIGHT:
        sidebar_on_right = True

    for block in content_blocks:
        if isinstance(block, LayoutGridBlock) and block.rows:
            content_cols = hint_content_cols
            if content_cols is None:
                content_cols = set()
                for row in block.rows:
                    for idx, cell in enumerate(row.cells):
                        if cell.blocks:
                            content_cols.add(idx)
            # Extract content from each column of the layout grid
            for row in block.rows:
                if len(row.cells) >= 2:
                    # Multi-column row - map sidebar column based on detected layout
                    sidebar_index = sidebar_index_hint
                    main_index = main_index_hint
                    if sidebar_index is None:
                        sidebar_index = len(row.cells) - 1 if sidebar_on_right else 0
                    if main_index is None:
                        main_index = 0 if sidebar_on_right else len(row.cells) - 1
                    if sidebar_index >= len(row.cells):
                        sidebar_index = len(row.cells) - 1
                    if main_index >= len(row.cells):
                        main_index = 0
                    for idx, cell in enumerate(row.cells):
                        if not cell.blocks:
                            continue
                        if idx == sidebar_index:
                            sidebar_blocks.extend(cell.blocks)
                        elif idx == main_index or idx in content_cols:
                            main_blocks.extend(cell.blocks)
                        else:
                            main_blocks.extend(cell.blocks)
                else:
                    # Single cell - goes to main
                    for cell in row.cells:
                        main_blocks.extend(cell.blocks)
        else:
            # Non-grid blocks go to main
            main_blocks.append(block)
    
    # Build sidebar minipage
    sidebar_lines = []
    for block in sidebar_blocks:
        rendered = render_block(block, context)
        if rendered:
            sidebar_lines.append(rendered)
    
    # Build main minipage  
    main_lines = []
    for block in main_blocks:
        rendered = render_block(block, context)
        if rendered:
            main_lines.append(rendered)
    
    # Render header fallback blocks before minipages
    header_lines = []
    for block in header_blocks:
        rendered = render_block(block, context)
        if rendered:
            header_lines.append(rendered)

    # Combine into minipage layout
    sidebar_content = "\n".join(sidebar_lines) if sidebar_lines else "~"
    main_content = "\n".join(main_lines) if main_lines else "~"

    if sidebar_on_right:
        layout = f"""\\noindent
\\begin{{minipage}}[t]{{{main_ratio:.2f}\\linewidth}}
{main_content}
\\end{{minipage}}%
\\hfill
\\begin{{minipage}}[t]{{{sidebar_ratio:.2f}\\linewidth}}
\\raggedright
{sidebar_content}
\\end{{minipage}}"""
    else:
        layout = f"""\\noindent
\\begin{{minipage}}[t]{{{sidebar_ratio:.2f}\\linewidth}}
\\raggedright
{sidebar_content}
\\end{{minipage}}%
\\hfill
\\begin{{minipage}}[t]{{{main_ratio:.2f}\\linewidth}}
{main_content}
\\end{{minipage}}"""

    if header_lines:
        return "\n".join(header_lines) + "\n" + layout
    return layout


def render_block(block: Block, context: RenderContext) -> str:
    """Render a single block to LaTeX."""
    if isinstance(block, FigureBlock) and getattr(block, "policy_reason", "") == "HEADER_IMAGE_FALLBACK":
        spacer = f"\\vspace*{{{block.height:.2f}in}}" if block.height else ""
        return "\n".join(filter(None, [_render_figure(block, context), spacer]))
    
    # Check for image fallback
    if getattr(block, 'render_as_image', False) and getattr(block, 'image_fallback_path', None):
        return _render_image_fallback(block)
    
    if isinstance(block, HeadingBlock):
        return _render_heading(block, context)
    elif isinstance(block, ParagraphBlock):
        return _render_paragraph(block, context)
    elif isinstance(block, ListBlock):
        return _render_list(block, context)
    elif isinstance(block, TableBlock):
        return _render_table(block, context)
    elif isinstance(block, LayoutGridBlock):
        return _render_layout_grid(block, context)
    elif isinstance(block, FigureBlock):
        return _render_figure(block, context)
    elif isinstance(block, TextboxBlock):
        return _render_textbox(block, context)
    else:
        # Generic block - render content
        return "".join(_render_span(s, context) for s in block.content)


def _render_image_fallback(block: Block) -> str:
    """Render block as image fallback."""
    path = block.image_fallback_path
    width = getattr(block, 'width', None)
    width_str = f"{width}in" if width else "\\linewidth"
    
    return (
        "\\begin{center}\n"
        f"\\includegraphics[width={width_str}]{{{path}}}\n"
        "\\end{center}"
    )


def _render_heading(block: HeadingBlock, context: RenderContext) -> str:
    """Render heading block."""
    content = "".join(_render_span(s, context) for s in block.content if not s.is_tab)
    font_prefix, font_suffix = _get_font_wrappers(block, context)
    if font_prefix and content:
        content = f"{font_prefix}{content}{font_suffix}"
    
    use_star = not context.use_numbered_sections
    star = "*" if use_star else ""
    
    level_map = {
        1: "section",
        2: "subsection", 
        3: "subsubsection",
        4: "paragraph",
        5: "subparagraph",
        6: "subparagraph",
    }
    cmd = level_map.get(block.level, "section")
    
    return f"\\{cmd}{star}{{{content}}}"


def _render_paragraph(block: ParagraphBlock, context: RenderContext) -> str:
    """Render paragraph with tabs, justification, spacing."""
    
    # Segment by tabs
    segments = []
    current = []
    
    for span in block.content:
        if span.is_tab:
            segments.append(current)
            segments.append(span)
            current = []
        else:
            current.append(span)
    segments.append(current)
    
    # Check for overlay method (right/center tabs)
    use_overlay = any(ts.alignment in ("right", "center") for ts in block.tab_stops)
    
    if use_overlay and block.tab_stops:
        content = _render_paragraph_overlay(segments, block.tab_stops, context)
    else:
        content = _render_paragraph_flow(segments, block.tab_stops, context)
    
    # Handle right_element (resume two-column layout)
    if hasattr(block, 'right_element') and block.right_element:
        right_text = escape_latex(block.right_element)
        right_frac = getattr(block, '_right_col_frac', 0.28)
        right_frac = max(0.20, min(0.38, right_frac))
        
        content = (
            "\\noindent\n"
            f"\\begin{{tabularx}}{{\\linewidth}}{{@{{}}X >{{\\raggedleft\\arraybackslash}}p{{{right_frac:.2f}\\linewidth}}@{{}}}}\n"
            f"{content} & {right_text} \\\\\n"
            "\\end{tabularx}"
        )

    font_prefix, font_suffix = _get_font_wrappers(block, context)
    if font_prefix and content:
        content = f"{font_prefix}{content}{font_suffix}"
    
    # Apply spacing
    result = ""
    if block.spacing_before > 0:
        result += f"\\addvspace{{{block.spacing_before}pt}}\n"
    
    result += content
    
    if block.spacing_after > 0:
        result += f"\n\\addvspace{{{block.spacing_after}pt}}"
    
    return result


def _render_paragraph_overlay(segments: list, tab_stops: list, context: RenderContext) -> str:
    """Render paragraph using overlay method for precise tab positioning."""
    parts = ["\\noindent"]
    current_x = 0.0
    
    # First segment at origin
    if segments and segments[0]:
        text = "".join(_render_span(s, context) for s in segments[0])
        parts.append(f"\\makebox[0pt][l]{{{text}}}")
    
    # Process tabs
    tab_idx = 0
    i = 1
    while i < len(segments):
        if tab_idx < len(tab_stops):
            ts = tab_stops[tab_idx]
            pos_in = ts.position / 1440.0
            delta = pos_in - current_x
            
            if delta > 0:
                parts.append(f"\\hspace*{{{delta:.2f}in}}")
                current_x = pos_in
            
            # Get next content
            next_content = segments[i + 1] if i + 1 < len(segments) else []
            text = "".join(_render_span(s, context) for s in next_content)
            
            if text:
                align = {"right": "r", "center": "c"}.get(ts.alignment, "l")
                parts.append(f"\\makebox[0pt][{align}]{{{text}}}")
        
        tab_idx += 1
        i += 2
    
    return "".join(parts)


def _render_paragraph_flow(segments: list, tab_stops: list, context: RenderContext) -> str:
    """Render paragraph using flow method with tabto."""
    parts = []
    
    # First segment
    if segments:
        parts.append("".join(_render_span(s, context) for s in segments[0]))
    
    # Process tabs
    tab_idx = 0
    i = 1
    while i < len(segments):
        next_content = segments[i + 1] if i + 1 < len(segments) else []
        text = "".join(_render_span(s, context) for s in next_content)
        
        if tab_idx < len(tab_stops):
            ts = tab_stops[tab_idx]
            pos_in = ts.position / 1440.0
            parts.append(f"\\tabto{{{pos_in:.2f}in}}")
            
            if ts.alignment == "right":
                parts.append(f"\\llap{{{text}}}")
            elif ts.alignment == "center":
                parts.append(f"\\clap{{{text}}}")
            else:
                parts.append(text)
        else:
            parts.append("\\hfill ")
            parts.append(text)
        
        tab_idx += 1
        i += 2
    
    return "".join(parts)


def _render_list(block: ListBlock, context: RenderContext) -> str:
    """Render list block."""
    env = "enumerate" if block.ordered else "itemize"
    
    lines = [f"\\begin{{{env}}}"]
    
    for item in block.items:
        content = "".join(_render_span(s, context) for s in item.content)
        lines.append(f"\\item {content}")
        
        # Nested lists (if any)
        if item.children:
            for child in item.children:
                child_content = "".join(_render_span(s, context) for s in child.content)
                lines.append(f"  \\item {child_content}")
    
    lines.append(f"\\end{{{env}}}")
    
    list_output = "\n".join(lines)
    font_prefix, font_suffix = _get_font_wrappers(block, context)
    if font_prefix:
        list_output = f"{font_prefix}{list_output}{font_suffix}"
    return list_output


def _render_table(block: TableBlock, context: RenderContext) -> str:
    """Render data table."""
    if not block.rows:
        return ""
    
    num_cols = max(len(row.cells) for row in block.rows)
    
    # Build column spec
    if block.column_widths:
        col_spec = "".join(f"p{{{w}in}}" for w in block.column_widths)
    else:
        col_spec = "l" * num_cols
    
    lines = [f"\\begin{{tabular}}{{{col_spec}}}"]
    lines.append("\\hline")
    
    for row in block.rows:
        cells = []
        for cell in row.cells:
            content = "".join(_render_span(s, context) for s in cell.content)
            cells.append(content)
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    
    return "\n".join(lines)


def _render_layout_grid(block: LayoutGridBlock, context: RenderContext) -> str:
    """Render layout grid (borderless table for positioning).
    
    LayoutGrids are typically used in resumes for multi-column content.
    We need to avoid section commands inside tabular cells.
    """
    if not block.rows:
        return ""
    
    # Calculate column widths
    if block.col_widths_twips:
        total = sum(block.col_widths_twips)
        col_widths = [w / total for w in block.col_widths_twips]
    else:
        num_cols = max(len(row.cells) for row in block.rows)
        col_widths = [1.0 / num_cols] * num_cols
    
    # Build column spec
    col_spec = "".join(f"p{{{w:.3f}\\linewidth}}" for w in col_widths)
    
    lines = ["\\noindent"]
    lines.append(f"\\begin{{tabularx}}{{\\linewidth}}{{@{{}}{col_spec}@{{}}}}")
    
    for row in block.rows:
        cells = []
        for cell in row.cells:
            # Render nested blocks in cell - use table-safe rendering
            cell_content = []
            for nested in cell.blocks:
                rendered = _render_block_in_table(nested, context)
                if rendered:
                    cell_content.append(rendered)
            # Join with empty string - each block already has its own line breaks
            cells.append("".join(cell_content) if cell_content else "")
        
        lines.append(" & ".join(cells) + " \\\\")
    
    lines.append("\\end{tabularx}")
    
    return "\n".join(lines)


def _render_block_in_table(block: Block, context: RenderContext) -> str:
    """Render a block inside a table cell (no section commands).
    
    For headings, we use bold text with underline (similar to titlerule).
    Paragraphs get proper spacing.
    """
    if isinstance(block, HeadingBlock):
        content = "".join(_render_span(s, context) for s in block.content if not s.is_tab)
        font_prefix, font_suffix = _get_font_wrappers(block, context)
        if font_prefix and content:
            content = f"{font_prefix}{content}{font_suffix}"
        # Use bold text with underline for section-like appearance
        if block.level == 1:
            return f"\\textbf{{{content}}}\\\\[-0.5ex]\\rule{{\\linewidth}}{{0.4pt}}\\\\[0.5ex]"
        elif block.level == 2:
            return f"\\textbf{{{content}}}\\\\[0.3ex]"
        else:
            return f"\\textit{{{content}}}"
    
    elif isinstance(block, ParagraphBlock):
        # Paragraph with proper spacing
        content = "".join(_render_span(s, context) for s in block.content)
        font_prefix, font_suffix = _get_font_wrappers(block, context)
        if font_prefix and content:
            content = f"{font_prefix}{content}{font_suffix}"
        if content.strip():
            return f"{content}\\\\[0.5ex]"
        return ""
    
    elif isinstance(block, ListBlock):
        # Compact list with bullets
        items = []
        for item in block.items:
            item_text = "".join(_render_span(s, context) for s in item.content)
            if block.ordered:
                items.append(f"\\textbullet\\ {item_text}")
            else:
                items.append(f"\\textbullet\\ {item_text}")
        return "\\\\".join(items) + "\\\\[0.5ex]"
    
    else:
        # Generic: just get content
        if hasattr(block, 'content'):
            content = "".join(_render_span(s, context) for s in block.content)
            if content.strip():
                return f"{content}\\\\[0.3ex]"
        return ""


def _render_figure(block: FigureBlock, context: RenderContext) -> str:
    """Render figure/image."""
    path = block.image_path
    
    # Determine width spec
    if block.width:
        width_spec = f"width={block.width}in"
    else:
        width_spec = "width=\\linewidth"

    # Filter unsupported formats (WMF/EMF cause XeLaTeX errors)
    if path.lower().endswith(('.wmf', '.emf')):
        return f"\\fbox{{[Image format {Path(path).suffix} not supported]}}"

    # Check for two-column academic layout
    is_two_col = (context.features and 
                  context.features.layout_pattern and 
                  context.features.layout_pattern.pattern == LayoutPattern.TWO_COLUMN_EQUAL)
    
    is_academic = context.features and context.features.doc_type.startswith("academic")
    
    env = "figure"
    placement = "[htbp]"
    
    # If spanning is needed (figure width > column width)
    # If spanning is needed (figure width > column width)
    if is_two_col:
        if block.width and block.width > (context.column_width * 1.1):
            env = "figure*"
            placement = "[t]"  # figure* usually flows to top
            width_spec = "width=\\textwidth"
        else:
            # Single column in multicol -> Non-floating (multicol forbids floats)
            content = []
            content.append("\\begin{center}")
            content.append(f"    \\includegraphics[{width_spec}]{{{path}}}")
            if block.caption:
                caption_safe = _escape_caption(block.caption)
                content.append(f"    \\captionof{{figure}}{{{caption_safe}}}")
                # Basic label generation
                content.append(f"    \\label{{fig:{Path(path).stem}}}")
            content.append("\\end{center}")
            return "\n".join(content)

    if not is_academic and (block.floating or block.origin_x != "page"):
        # Resume/Standard Layout -> Use textblock for absolute positioning if desired
        return (
            f"\\begin{{textblock*}}{{\\linewidth}}({block.x}in,{block.y}in)\\noindent\n"
            f"\\includegraphics[{width_spec}]{{{path}}}\n"
            "\\end{textblock*}"
        )
    
    # Semantic Figure Environment (Academic / Standard)
    content = []
    content.append(f"\\begin{{{env}}}{placement}")
    content.append("    \\centering")
    content.append(f"    \\includegraphics[{width_spec}]{{{path}}}")
    if block.caption:
         caption_safe = _escape_caption(block.caption)
         content.append(f"    \\caption{{{caption_safe}}}")
         # Basic label generation
         content.append(f"    \\label{{fig:{Path(path).stem}}}")
    content.append(f"\\end{{{env}}}")
    
    return "\n".join(content)


def _render_textbox(block: TextboxBlock, context: RenderContext) -> str:
    """Render floating textbox."""
    # Textboxes are typically hoisted and rendered at document level
    # Return empty for inline position
    return ""


def _render_span(span: InlineSpan, context: RenderContext) -> str:
    """Render inline span to LaTeX."""
    if getattr(span, 'is_paragraph_break', False):
        return " \\newline{}\\n"
    
    if span.is_tab:
        return "\\hfill "
    
    text = span.text
    
    # Handle floating elements (Figures/Textboxes) attached to span
    if getattr(span, 'floating_element', None):
        elem = span.floating_element
        if isinstance(elem, FigureBlock):
            return _render_figure(elem, context)
        elif isinstance(elem, TextboxBlock):
            return _render_textbox(elem, context)
    
    if not text and not span.footnote_content:
        return ""
    
    if text:
        text = escape_latex(text)
        
        if span.bold and text.strip():
            text = f"\\textbf{{{text}}}"
        if span.italic and text.strip():
            text = f"\\textit{{{text}}}"
        if span.underline:
            text = f"\\underline{{{text}}}"
    
    if span.url and text:
        text = f"\\href{{{span.url}}}{{\\textcolor{{blue}}{{{text}}}}}"
    
    if span.footnote_content:
        fn_text = "".join(_render_span(s, context) for s in span.footnote_content)
        text += f"\\footnote{{{fn_text.strip()}}}"
    
    return text





# =============================================================================
# Main Entry Point
# =============================================================================

def generate_latex(doc: Document, ref_pdf: Optional[Path] = None) -> str:
    """Generate LaTeX from Document IR without templates.
    
    Args:
        doc: Document IR from OOXML parser
        ref_pdf: Optional reference PDF for layout calibration
        
    Returns:
        Complete LaTeX document as string
    """
    # Analyze document features
    features = analyze_document(doc, ref_pdf)
    
    # Build preamble
    preamble = build_preamble(features)
    
    # Render document body
    body = render_document(doc, features)
    
    return preamble + "\n" + body
