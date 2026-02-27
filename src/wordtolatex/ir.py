"""Intermediate Representation (IR) for Word documents.

This module defines the canonical data structures used throughout the pipeline:
OOXML Parser → IR → LaTeX Generator
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict


@dataclass
class InlineSpan:
    """A run of text with inline formatting."""

    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    font_name: Optional[str] = None
    font_size: Optional[float] = None  # in points
    is_tab: bool = False  # True if this is a tab character
    url: Optional[str] = None  # Hyperlink URL (if this span is a link)
    floating_element: Optional["Block"] = None  # Attached floating block (e.g. Textbox)
    is_paragraph_break: bool = False  # True if this represents a paragraph boundary
    style_ref: Optional[str] = None  # Paragraph style (e.g., "Heading1")
    footnote_content: Optional[List["InlineSpan"]] = None  # Content of a footnote


@dataclass
class TabStop:
    """A tab stop definition."""
    position: int  # in twips
    alignment: str = "left"  # left, right, center, decimal


@dataclass
class Block:
    """Base class for all block-level elements."""

    id: str
    block_type: str
    style_ref: Optional[str] = None
    content: List[InlineSpan] = field(default_factory=list)
    policy: Literal["semantic", "fallback"] = "semantic"
    
    # Layout attributes
    tab_stops: List[TabStop] = field(default_factory=list)
    justification: Optional[str] = None  # left, right, center, both
    spacing_before: float = 0  # in points
    spacing_after: float = 0  # in points
    line_spacing: Optional[float] = None  # line height value (multiplier OR points depending on type)
    line_spacing_type: Literal["multiplier", "exact", "atLeast"] = "multiplier" # Default to multiplier
    font_family: Optional[str] = None
    font_size: Optional[float] = None  # in points
    
    # Policy Audit
    policy_reason: Optional[str] = None # Reason code for the policy decision
    
    # Visual Fallback
    render_as_image: bool = False # If True, render as \includegraphics
    image_fallback_path: Optional[str] = None # Path to extracted image crop from Ref PDF


@dataclass
class HeadingBlock(Block):
    """A heading block (H1-H6)."""

    level: int = 1

    def __post_init__(self):
        self.block_type = "heading"


@dataclass
class ParagraphBlock(Block):
    """A standard paragraph."""
    
    # For two-column resume rows: right-aligned element (e.g., dates)
    right_element: Optional[str] = None
    # True for blank paragraphs (no visible text) - used for vertical spacing
    is_empty: bool = False

    def __post_init__(self):
        self.block_type = "paragraph"


@dataclass
class ListItem:
    """A single list item."""

    content: List[InlineSpan] = field(default_factory=list)
    children: List["ListItem"] = field(default_factory=list)


@dataclass
class ListBlock(Block):
    """A list (ordered or unordered)."""

    ordered: bool = False
    items: List[ListItem] = field(default_factory=list)

    def __post_init__(self):
        self.block_type = "list"


@dataclass
class TableCell:
    """A single table cell."""

    content: List[InlineSpan] = field(default_factory=list)
    col_span: int = 1
    row_span: int = 1


@dataclass
class TableRow:
    """A table row."""

    cells: List[TableCell] = field(default_factory=list)


@dataclass
class TableBlock(Block):
    """A table."""

    rows: List[TableRow] = field(default_factory=list)
    complexity_score: int = 0
    column_widths: List[float] = field(default_factory=list)  # Explicit column widths in inches

    def __post_init__(self):
        self.block_type = "table"


@dataclass
class GridCell:
    """A cell in a layout grid, containing nested blocks."""
    
    blocks: List["Block"] = field(default_factory=list)  # paragraphs/headings inside cell
    col_span: int = 1
    row_span: int = 1
    padding_twips: Optional[Dict[str, int]] = None  # from w:tcPr/w:tcMar overrides
    v_align: str = "top"  # Vertical alignment: "top" | "center" | "bottom"


@dataclass
class GridRow:
    """A row in a layout grid with height information."""
    
    cells: List[GridCell] = field(default_factory=list)
    height_twips: Optional[int] = None  # from w:trHeight/@w:val
    height_rule: Optional[str] = None   # "exact" | "atLeast" | None


@dataclass
class LayoutGridBlock(Block):
    """A layout grid (borderless table used for positioning).
    
    Unlike TableBlock (for data tables), this preserves Word's 
    2-column layout geometry for resumes and templates.
    """
    
    col_widths_twips: Optional[List[int]] = None  # from w:tblGrid if present
    rows: List[GridRow] = field(default_factory=list)
    cell_margin_twips: Optional[Dict[str, int]] = None  # top/left/bottom/right
    
    def __post_init__(self):
        self.block_type = "layout_grid"


@dataclass
class FigureBlock(Block):
    """An embedded image/figure."""

    image_path: str = ""
    caption: str = ""
    width: Optional[float] = None  # in inches
    height: Optional[float] = None # in inches
    
    # Positioning
    floating: bool = False
    x: float = 0.0
    y: float = 0.0
    origin_x: str = "page"
    origin_y: str = "page"

    def __post_init__(self):
        self.block_type = "figure"


@dataclass
class Section:
    """Document section with page setup."""

    page_width: float = 8.5  # inches
    page_height: float = 11.0  # inches
    margin_top: float = 1.0
    margin_bottom: float = 1.0
    margin_left: float = 1.0
    margin_right: float = 1.0
    
    # Columns
    col_count: int = 1
    col_spacing: float = 36.0  # Default 0.5 inches in points


@dataclass
class TextboxBlock(Block):
    """A floating textbox with absolute position."""
    
    x: float = 0.0  # inches from page left
    y: float = 0.0  # inches from page top
    width: float = 0.0 # inches
    height: float = 0.0 # inches
    content_blocks: List[Block] = field(default_factory=list) # Nested blocks
    
    # Positioning context
    origin_x: str = "page"  # page, margin, column, character, leftMargin, rightMargin, insideMargin, outsideMargin
    origin_y: str = "page"  # page, margin, paragraph, line, topMargin, bottomMargin, insideMargin, outsideMargin

    def __post_init__(self):
        self.block_type = "textbox"


@dataclass
class Asset:
    """An extracted asset (image, etc.)."""

    original_path: str
    output_path: str
    asset_type: str = "image"


@dataclass
class LayoutHints:
    """Detected layout parameters from the layout oracle."""
    
    body_font_size: float = 11.0
    body_font_family: str = "Calibri"
    line_spacing: float = 1.0
    margin_top: float = 1.0
    margin_bottom: float = 1.0
    margin_left: float = 1.0
    margin_right: float = 1.0
    page_width: float = 8.5
    page_height: float = 11.0
    
    # Dynamic Feedback Parameters
    suggested_squeeze: float = 1.0  # Adaptive line spacing multiplier (0.8 - 1.2)
    suggested_columns: Optional[int] = None  # Detected number of columns (1 or 2)
    suggested_font_size: Optional[float] = None  # Target base font size from PDF metrics
    suggested_font_size: Optional[float] = None  # Target base font size from PDF metrics
    suggested_section_spacing: Optional[float] = None  # Target section spacing from PDF metrics
    
    # Fidelity Elements
    visual_elements: List[Dict] = field(default_factory=list) # Horizontal lines, shapes, etc.


@dataclass
class Document:
    """The complete document representation."""

    metadata: Dict = field(default_factory=dict)
    sections: List[Section] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    assets: List[Asset] = field(default_factory=list)
    header_blocks: List[Block] = field(default_factory=list)
    footer_blocks: List[Block] = field(default_factory=list)
    styles: Dict[str, Dict] = field(default_factory=dict)
    layout_hints: LayoutHints = field(default_factory=LayoutHints)
    
    # Section numbering style
    use_numbered_sections: bool = False  # If True, use \section instead of \section*
    
    # Audit Trace
    policy_trace: List[Dict] = field(default_factory=list)
