"""Integration tests using real DOCX files."""

import pytest
from pathlib import Path

from wordtolatex.docx_parser import parse_docx
from wordtolatex.generator import generate_latex
from wordtolatex.policy import decide_policy


SAMPLES_DIR = Path(__file__).parent / "samples"


class TestResumeDocument:
    """Test parsing and conversion of a real resume document."""

    @pytest.fixture
    def resume_path(self):
        return SAMPLES_DIR / "resume.docx"

    def test_resume_exists(self, resume_path):
        """Ensure the test document exists."""
        assert resume_path.exists(), f"Resume not found at {resume_path}"

    def test_parse_resume(self, resume_path):
        """Test that the resume can be parsed without errors."""
        doc = parse_docx(resume_path)
        
        # Should have at least one section
        assert len(doc.sections) >= 1
        
        # Should have blocks (paragraphs, headings)
        assert len(doc.blocks) > 0
        
        # Print stats for debugging
        print(f"\nParsed {len(doc.blocks)} blocks")
        print(f"Page size: {doc.sections[0].page_width}x{doc.sections[0].page_height} inches")

    def test_resume_has_content(self, resume_path):
        """Test that parsed blocks contain text."""
        doc = parse_docx(resume_path)
        
        # Count blocks with actual text content
        blocks_with_text = [b for b in doc.blocks if b.content]
        assert len(blocks_with_text) > 0, "No blocks with text content found"
        
        # Print first few blocks for debugging
        print("\nFirst 5 blocks:")
        for i, block in enumerate(doc.blocks[:5]):
            text = "".join(span.text for span in block.content)
            print(f"  [{block.block_type}] {text[:50]}...")

    def test_generate_latex_from_resume(self, resume_path):
        """Test that LaTeX can be generated from the resume."""
        doc = parse_docx(resume_path)
        
        # Apply policy to all blocks
        for block in doc.blocks:
            block.policy = decide_policy(block)
        
        # Generate LaTeX
        latex = generate_latex(doc)
        
        # Basic validation
        assert "\\documentclass" in latex
        assert "\\begin{document}" in latex
        assert "\\end{document}" in latex
        
        # Should have some actual content
        assert len(latex) > 500, "LaTeX output seems too short"
        
        print(f"\nGenerated {len(latex)} characters of LaTeX")

    def test_save_latex_output(self, resume_path, tmp_path):
        """Test that LaTeX output can be saved to a file."""
        doc = parse_docx(resume_path)
        
        for block in doc.blocks:
            block.policy = decide_policy(block)
        
        latex = generate_latex(doc)
        
        # Save to temp file
        output_path = tmp_path / "resume_output.tex"
        output_path.write_text(latex, encoding="utf-8")
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        print(f"\nSaved LaTeX to: {output_path}")


class TestBlockTypes:
    """Test that different block types are detected."""

    @pytest.fixture
    def resume_doc(self):
        resume_path = SAMPLES_DIR / "resume.docx"
        if not resume_path.exists():
            pytest.skip("Resume document not available")
        return parse_docx(resume_path)

    def test_detect_block_types(self, resume_doc):
        """Check what block types are present."""
        block_types = {}
        for block in resume_doc.blocks:
            block_types[block.block_type] = block_types.get(block.block_type, 0) + 1
        
        print("\nBlock type counts:")
        for bt, count in sorted(block_types.items()):
            print(f"  {bt}: {count}")
        
        # Should have at least paragraphs
        assert "paragraph" in block_types or "heading" in block_types

    def test_inline_formatting(self, resume_doc):
        """Check that inline formatting is detected."""
        bold_count = 0
        italic_count = 0
        
        for block in resume_doc.blocks:
            for span in block.content:
                if span.bold:
                    bold_count += 1
                if span.italic:
                    italic_count += 1
        
        print(f"\nFormatting: {bold_count} bold spans, {italic_count} italic spans")

    def test_layout_hints_integration(self, resume_doc):
        """Test that layout hints are correctly applied to generated LaTeX."""
        # 1. Create artificial hints
        from wordtolatex.ir import LayoutHints
        hints = LayoutHints(
            body_font_size=12.5,
            line_spacing=1.8,
            margin_left=0.75,
            margin_right=0.75,
            margin_top=0.75,
            margin_bottom=0.75
        )
        resume_doc.layout_hints = hints
        # Also update Section margins, as generator prioritizes them IF sections exist
        if resume_doc.sections:
            resume_doc.sections[0].margin_left = 0.75
            resume_doc.sections[0].margin_right = 0.75
            resume_doc.sections[0].margin_top = 0.75
            resume_doc.sections[0].margin_bottom = 0.75
        
        # 2. Generate
        latex = generate_latex(resume_doc)
        
        # 3. Verify - Note: documentclass now uses fixed 10pt for stability
        # The custom font size is no longer injected into documentclass
        # And letterpaper is handled by geometry package now
        assert "\\documentclass[10pt]{article}" in latex
        # Line spacing is used RAW in the preamble (squeeze factor applies to blocks if used there)
        assert "\\setstretch{1.8}" in latex
        assert "left=0.75in" in latex
