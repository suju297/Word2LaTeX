"""Tests for the dynamic LaTeX generator."""

import pytest
from pathlib import Path

from wordtolatex.dynamic_generator import (
    DocumentFeatures,
    RenderContext,
    analyze_document,
    build_preamble,
    render_block,
    generate_latex,
    _render_heading,
    _render_paragraph,
    _escape_latex,
)
from wordtolatex.ir import (
    Document, Section, HeadingBlock, ParagraphBlock, InlineSpan,
    LayoutHints, LayoutGridBlock, GridRow, GridCell
)


class TestDocumentFeatures:
    """Tests for DocumentFeatures detection."""
    
    def test_default_features(self):
        """Test default feature values."""
        features = DocumentFeatures()
        assert features.doc_class == "article"
        assert features.column_count == 1
        assert features.font_size == 11.0
        assert not features.has_tables
        assert not features.has_figures
    
    def test_features_from_sections(self):
        """Test feature extraction from section props."""
        doc = Document(
            sections=[Section(
                page_width=8.5,
                page_height=11.0,
                col_count=2,
                margin_top=0.5,
                margin_left=0.75,
            )],
            blocks=[],
            layout_hints=LayoutHints(body_font_family="Times New Roman"),
        )
        features = analyze_document(doc)
        assert features.column_count == 2
        assert features.margin_top == 0.5
        assert features.margin_left == 0.75


class TestDocumentTypeDetection:
    """Tests for automatic document type detection."""
    
    def _make_heading(self, text: str, level: int = 1) -> HeadingBlock:
        return HeadingBlock(
            id=f"h_{text}",
            block_type="heading",
            level=level,
            content=[InlineSpan(text=text)],
        )
    
    def test_detect_resume_by_headings(self):
        """Resume keywords → resume type."""
        doc = Document(
            sections=[Section()],
            blocks=[
                self._make_heading("Education"),
                self._make_heading("Experience"),
                self._make_heading("Skills"),
            ],
            layout_hints=LayoutHints(),
        )
        features = analyze_document(doc)
        assert features.doc_type == "resume"
        assert features.heading_style == "ruled"
    
    def test_detect_academic_singlecol(self):
        """Academic keywords → academic_singlecol."""
        doc = Document(
            sections=[Section(col_count=1)],
            blocks=[
                self._make_heading("Abstract"),
                self._make_heading("Introduction"),
                self._make_heading("Methodology"),
                self._make_heading("Results"),
                self._make_heading("Conclusion"),
            ],
            layout_hints=LayoutHints(),
        )
        features = analyze_document(doc)
        assert features.doc_type == "academic_singlecol"
        assert features.heading_style == "numbered"
    
    def test_detect_generic_fallback(self):
        """No strong signals → generic."""
        doc = Document(
            sections=[Section()],
            blocks=[
                ParagraphBlock(id="p1", block_type="paragraph"),
            ],
            layout_hints=LayoutHints(),
        )
        features = analyze_document(doc)
        assert features.doc_type == "generic"


class TestPreambleBuilder:
    """Tests for dynamic preamble generation."""
    
    def test_basic_preamble(self):
        """Test minimal preamble."""
        features = DocumentFeatures(font_family="Arial")
        preamble = build_preamble(features)
        
        assert "\\documentclass" in preamble
        assert "\\usepackage{fontspec}" in preamble
        assert "\\usepackage{geometry}" in preamble
    
    def test_preamble_with_tables(self):
        """Tables feature adds tabularx."""
        features = DocumentFeatures(has_tables=True)
        preamble = build_preamble(features)
        assert "\\usepackage{tabularx}" in preamble
    
    def test_preamble_with_hyperlinks(self):
        """Hyperlinks add hyperref and xcolor."""
        features = DocumentFeatures(has_hyperlinks=True)
        preamble = build_preamble(features)
        assert "\\usepackage[hidelinks]{hyperref}" in preamble
        assert "\\usepackage{xcolor}" in preamble
    
    def test_two_column_class_options(self):
        """Two-column sets class options."""
        features = DocumentFeatures(
            doc_type="academic_twocol",
            class_options=["10pt", "twocolumn"],
        )
        preamble = build_preamble(features)
        assert "\\documentclass[10pt, twocolumn]{article}" in preamble


class TestBlockRendering:
    """Tests for block rendering functions."""
    
    def test_render_heading_unnumbered(self):
        """Unnumbered headings use star."""
        context = RenderContext(use_numbered_sections=False)
        block = HeadingBlock(
            id="h1",
            block_type="heading",
            level=1,
            content=[InlineSpan(text="Introduction")],
        )
        result = _render_heading(block, context)
        assert result == "\\section*{Introduction}"
    
    def test_render_heading_numbered(self):
        """Numbered headings omit star."""
        context = RenderContext(use_numbered_sections=True)
        block = HeadingBlock(
            id="h1",
            block_type="heading",
            level=2,
            content=[InlineSpan(text="Methods")],
        )
        result = _render_heading(block, context)
        assert result == "\\subsection{Methods}"
    
    def test_escape_latex_special_chars(self):
        """Special characters escaped."""
        assert _escape_latex("10% off") == "10\\% off"
        assert _escape_latex("$5 & tax") == "\\$5 \\& tax"
        assert _escape_latex("file_name") == "file\\_name"
        assert _escape_latex("#{hash}") == "\\#\\{hash\\}"


class TestIntegration:
    """Integration tests for full document generation."""
    
    def test_generate_minimal_document(self):
        """Generate LaTeX for minimal document."""
        doc = Document(
            sections=[Section(page_width=8.5, page_height=11.0)],
            blocks=[
                HeadingBlock(
                    id="h1",
                    block_type="heading",
                    level=1,
                    content=[InlineSpan(text="Hello")],
                ),
                ParagraphBlock(
                    id="p1",
                    block_type="paragraph",
                    content=[InlineSpan(text="World")],
                ),
            ],
            layout_hints=LayoutHints(),
        )
        
        latex = generate_latex(doc)
        
        assert "\\documentclass" in latex
        assert "\\begin{document}" in latex
        assert "\\end{document}" in latex
        assert "Hello" in latex
        assert "World" in latex
