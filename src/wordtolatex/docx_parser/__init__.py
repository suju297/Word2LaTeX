"""DOCX Parser Package - OOXML extraction modules."""

from .document import parse_docx, compute_layout_score
from .styles import parse_styles, get_heading_level
from .numbering import parse_numbering, get_list_info
from .media import extract_media, get_relationship_targets
from .tables import parse_table, calculate_complexity

__all__ = [
    "parse_docx",
    "compute_layout_score",
    "parse_styles",
    "get_heading_level",
    "parse_numbering",
    "get_list_info",
    "extract_media",
    "get_relationship_targets",
    "parse_table",
    "calculate_complexity",
]

