
from wordtolatex.ir import ParagraphBlock, TextboxBlock, InlineSpan
from wordtolatex.generator import _format_paragraph, _format_textbox

class TestTextboxRendering:
    """Tests for Absolute Textbox rendering."""

    def test_textbox_standalone(self):
        """Test direct formatting of a TextboxBlock (Suppressed in main flow, handled by floating logic)."""
        box = TextboxBlock(
            id="tb1",
            block_type="textbox",
            x=1.5,
            y=2.0,
            width=3.0,
            height=1.0,
            content_blocks=[
                ParagraphBlock(id="p_inner", block_type="paragraph", content=[InlineSpan(text="Inner Text")])
            ]
        )
        
        # In main flow, textboxes are suppressed (empty string)
        latex = _format_textbox(box)
        assert latex.strip() == ""

        # Use _format_floating_block to verify actual rendering
        from wordtolatex.generator import _format_floating_block
        float_latex = _format_floating_block(box)
        
        # Check textpos wrapper
        assert "\\begin{textblock*}{3.00in}(1.50in,2.00in)" in float_latex
        assert "\\end{textblock*}" in float_latex
        
        # Check tcolorbox wrapper
        assert "\\begin{tcolorbox}" in float_latex
        assert "colback=white" in float_latex
        assert "boxrule=0.5pt" in float_latex
        
        # Check content
        assert "Inner Text" in float_latex

    def test_textbox_anchored_in_span(self):
        """Test textbox suppression when anchored in span."""
        box = TextboxBlock(
            id="tb2",
            block_type="textbox",
            x=5.0,
            y=1.0,
            width=2.0,
            content_blocks=[
                ParagraphBlock(id="p_inner2", block_type="paragraph", content=[InlineSpan(text="Floating")])
            ]
        )
        
        parent_block = ParagraphBlock(
            id="p_outer",
            block_type="paragraph",
            content=[
                InlineSpan(text="Anchor Text"),
                InlineSpan(text="", floating_element=box)
            ]
        )
        
        latex = _format_paragraph(parent_block)
        
        assert "Anchor Text" in latex
        # The floating element should be SUPPRESSED in the paragraph flow
        assert "textblock*" not in latex
        assert "Floating" not in latex
