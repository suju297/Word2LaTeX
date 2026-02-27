
from wordtolatex.ir import ParagraphBlock, TextboxBlock, InlineSpan, TabStop, Document, Section
from wordtolatex.generator import _format_paragraph, _format_textbox, generate_latex, _CURRENT_MARGINS

class TestFixesVerification:
    """Verify the 3 concrete fixes requested."""

    def test_fix1_tab_overlay_method(self):
        """Fix #1: Tabs with Right alignment should use Overlay Method (makebox)."""
        # Paragraph: "Company" \t[Right, 6in] "Date"
        block = ParagraphBlock(
            id="p1",
            block_type="paragraph",
            content=[
                InlineSpan(text="Company"),
                InlineSpan(text="\t", is_tab=True),
                InlineSpan(text="2020-2024")
            ],
            tab_stops=[TabStop(position=8640, alignment="right")] # 8640 twips = 6.0 in
        )
        
        latex = _format_paragraph(block)
        
        # Expect Overlay Method commands
        assert "\\noindent" in latex
        assert "\\makebox[0pt][l]{Company}" in latex
        assert "\\hspace*{6.00in}" in latex
        assert "\\makebox[0pt][r]{2020-2024}" in latex
        # Ensure fallback \tabto is NOT used
        assert "\\tabto" not in latex

    def test_fix1_tab_fallback_method(self):
        """Fix #1: Simple Left tabs should use Fallback Method (tabto)."""
        # Paragraph: "Item" \t[Left, 1in] "Value"
        block = ParagraphBlock(
            id="p2",
            block_type="paragraph",
            content=[
                InlineSpan(text="Item"),
                InlineSpan(text="\t", is_tab=True),
                InlineSpan(text="Value")
            ],
            tab_stops=[TabStop(position=1440, alignment="left")] # 1440 twips = 1.0 in
        )
        
        latex = _format_paragraph(block)
        
        # Expect Fallback Method commands
        assert "\\tabto" in latex
        assert "\\makebox[0pt]" not in latex

    def test_fix2_textbox_margin_offset(self):
        """Fix #2: Textboxes relative to margin should include margin offset."""
        # 1. Update Global Margins (simulate generate_latex behavior)
        _CURRENT_MARGINS["left"] = 1.0
        _CURRENT_MARGINS["top"] = 1.0
        
        # Textbox at 1.0, 1.0 relative to margin
        box = TextboxBlock(
            id="tb1",
            block_type="textbox",
            x=1.0, 
            y=1.0, 
            width=2.0,
            origin_x="margin",
            origin_y="margin",
            content_blocks=[]
        )
        
        # Use floating renderer directly
        from wordtolatex.generator import _format_floating_block
        latex = _format_floating_block(box)
        
        # Expected Absolute Position: 1.0 + 1.0 = 2.0
        assert "(2.00in,2.00in)" in latex

    def test_fix2_textbox_page_absolute(self):
        """Fix #2: Textboxes relative to page should NOT include margin offset."""
        _CURRENT_MARGINS["left"] = 1.0
        
        # Textbox at 1.0, 1.0 relative to page
        box = TextboxBlock(
            id="tb2",
            block_type="textbox",
            x=1.0, 
            y=1.0, 
            width=2.0,
            origin_x="page",
            origin_y="page",
            content_blocks=[]
        )
        
        # Use floating renderer directly
        from wordtolatex.generator import _format_floating_block
        latex = _format_floating_block(box)
        
        # Expected Absolute Position: 1.0 (no offset)
        assert "(1.00in,1.00in)" in latex

    def test_generate_latex_sets_context(self):
        """Verify generate_latex sets the global margin context."""
        # Create doc with weird margins
        doc = Document(
            sections=[Section(margin_left=0.5, margin_top=0.5)],
            blocks=[]
        )
        
        generate_latex(doc)
        
        # Check if global context was updated
        assert _CURRENT_MARGINS["left"] == 0.5
        assert _CURRENT_MARGINS["top"] == 0.5
        
        # Reset to default for other tests if needed
        _CURRENT_MARGINS["left"] = 1.0
        _CURRENT_MARGINS["top"] = 1.0
