import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock fitz BEFORE importing src.solver because it imports fitz at top level
sys.modules["fitz"] = MagicMock()

from wordtolatex.solver import find_optimal_layout, get_page_count

@pytest.fixture
def mock_dependencies():
    with patch("wordtolatex.solver.parse_docx") as mock_parse, \
         patch("wordtolatex.solver.analyze_document") as mock_analyze, \
         patch("wordtolatex.solver.get_page_count") as mock_get_pages, \
         patch("wordtolatex.solver.compile_latex") as mock_compile, \
         patch("wordtolatex.solver.shutil.which") as mock_which: # Mock pdflatex check
        
        # Setup mocks
        mock_doc = MagicMock()
        mock_parse.return_value = mock_doc

        # Mock sections[0] properties for generator.py
        mock_section = MagicMock()
        mock_section.col_count = 1  # Default to 1 column
        mock_section.col_spacing = 36.0
        mock_section.margin_left = 1.0
        mock_section.margin_top = 1.0
        mock_section.page_width = 8.5
        mock_section.page_height = 11.0
        mock_doc.sections = [mock_section]
        
        mock_which.return_value = "/usr/bin/pdflatex" # Simulate installed
        
        mock_layout = MagicMock()
        mock_layout.body_font_size = 10
        mock_layout.line_spacing = 1.0
        mock_layout.margin_top = 1.0
        mock_layout.margin_bottom = 1.0
        mock_layout.margin_left = 1.0
        mock_layout.margin_right = 1.0
        mock_layout.width = 612
        mock_layout.height = 792
        mock_analyze.return_value = mock_layout
        
        # Default behavior for compile/pages
        mock_compile.return_value = "dummy.pdf"
        
        yield mock_parse, mock_analyze, mock_get_pages, mock_compile, mock_which

def test_solver_immediate_success(mock_dependencies):
    mock_parse, _, mock_get_pages, mock_compile, _ = mock_dependencies
    
    # Scenario: Target is 1 page. Generated is 1 page immediately.
    # Note: find_optimal_layout gets page count from compiled PDF.
    mock_get_pages.side_effect = [1] 
    
    # We pass a dummy doc_ir (mock_parse.return_value)
    find_optimal_layout(mock_parse.return_value, 1, Path("."))
    
    # Should compile once
    assert mock_compile.call_count == 1

def test_solver_squeeze_loop(mock_dependencies):
    mock_parse, _, mock_get_pages, mock_compile, _ = mock_dependencies
    
    # Scenario: Target 1 page.
    # Attempt 1: 2 pages (Too long)
    # Attempt 2: 1 page (Success)
    mock_get_pages.side_effect = [2, 1] 
    
    find_optimal_layout(mock_parse.return_value, 1, Path("."))
    
    assert mock_compile.call_count == 2
