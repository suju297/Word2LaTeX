
from wordtolatex.ir import TableBlock, TableRow, TableCell, InlineSpan
from wordtolatex.generator import _format_table

class TestFixedTables:
    """Tests for Fixed Width Table rendering."""

    def test_fixed_width_simple(self):
        """Table with explicit widths uses tabular and p{}."""
        table = TableBlock(
            id="t1",
            block_type="table",
            column_widths=[1.0, 2.0, 1.5], # in inches
            rows=[
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="A")]),
                    TableCell(content=[InlineSpan(text="B")]),
                    TableCell(content=[InlineSpan(text="C")])
                ])
            ]
        )
        
        latex = _format_table(table)
        
        # Expect tabular not tabularx
        assert "\\begin{tabular}" in latex
        assert "{|p{1.00in}|p{2.00in}|p{1.50in}|}" in latex
        assert "tabularx" not in latex

    def test_multicolumn_width_sum(self):
        """Multicolumn cell should sum widths of spanned columns."""
        # 3 columns: 1in, 1in, 1in. Total 3in.
        table = TableBlock(
            id="t2",
            block_type="table",
            column_widths=[1.0, 1.0, 1.0],
            rows=[
                TableRow(cells=[
                    # Spans first 2 columns. Width should be 1.0 + 1.0 = 2.0
                    TableCell(content=[InlineSpan(text="Span")], col_span=2),
                    TableCell(content=[InlineSpan(text="Single")])
                ])
            ]
        )
        
        latex = _format_table(table)
        
        # Check multicolumn p width
        # \multicolumn{2}{|p{2.00in}|}{...}
        assert "\\multicolumn{2}{|p{2.00in}|}{Span}" in latex

    def test_fallback_auto(self):
        """Table without widths falls back to tabularx."""
        table = TableBlock(
            id="t3",
            block_type="table",
            column_widths=[], # Empty
            rows=[
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="A")]),
                    TableCell(content=[InlineSpan(text="B")])
                ])
            ]
        )
        
        latex = _format_table(table)
        assert "\\begin{tabularx}" in latex
        assert "{|X|X|}" in latex

