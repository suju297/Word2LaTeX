
from wordtolatex.ir import ParagraphBlock, TabStop, InlineSpan
from wordtolatex.generator import _format_paragraph

class TestTabHandling:
    """Tests for Tab Stop rendered in paragraphs."""

    def test_left_tab(self):
        """Single Left Tab should render as \\tabto."""
        block = ParagraphBlock(
            id="p1",
            block_type="paragraph",
            content=[
                InlineSpan(text="Name:"),
                InlineSpan(text="\t", is_tab=True),
                InlineSpan(text="John Doe")
            ],
            tab_stops=[TabStop(position=1440, alignment="left")] # 1 inch
        )
        
        latex = _format_paragraph(block)
        # Expected: Name:\tabto{1.00in}John Doe
        assert "\\tabto{1.00in}" in latex
        assert "Name:" in latex
        assert "John Doe" in latex

    def test_right_tab_date(self):
        """Right Tab should render using precision layout (makebox/hspace)."""
        block = ParagraphBlock(
            id="p2",
            block_type="paragraph",
            content=[
                InlineSpan(text="Company"),
                InlineSpan(text="\t", is_tab=True),
                InlineSpan(text="2020-Present")
            ],
            tab_stops=[TabStop(position=7200, alignment="right")] # 5 inches
        )
        
        latex = _format_paragraph(block)
        # Expected: ...Company...\hspace*{5.00in}\makebox[0pt][r]{2020-Present}
        assert r"\hspace*{5.00in}" in latex
        assert r"\makebox[0pt][r]{2020-Present}" in latex

    def test_multiple_tabs(self):
        """Multiple tabs mixed with text."""
        block = ParagraphBlock(
            id="p3",
            block_type="paragraph",
            content=[
                InlineSpan(text="A"),
                InlineSpan(text="\t", is_tab=True),
                InlineSpan(text="B"),
                InlineSpan(text="\t", is_tab=True),
                InlineSpan(text="C")
            ],
            tab_stops=[
                TabStop(position=1440, alignment="left"),  # 1in
                TabStop(position=2880, alignment="left")   # 2in
            ]
        )
        
        latex = _format_paragraph(block)
        # Standard left tabs use \tabto
        assert r"\tabto{1.00in}" in latex
        assert r"\tabto{2.00in}" in latex

    def test_tab_fallback(self):
        """Tab without definition depends on fallback (quad or hfill)."""
        block = ParagraphBlock(
            id="p4",
            block_type="paragraph",
            content=[
                InlineSpan(text="Start"),
                InlineSpan(text="\t", is_tab=True),
                InlineSpan(text="End")
            ],
            tab_stops=[] # No stops defined
        )
        
        latex = _format_paragraph(block)
        # Fallback to quad by default
        assert "\\quad" in latex or "\\hfill" in latex
