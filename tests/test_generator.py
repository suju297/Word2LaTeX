"""Tests for the LaTeX generator - specifically table formatting."""

import pytest
from wordtolatex.ir import TableBlock, TableRow, TableCell, InlineSpan
from wordtolatex.generator import _format_table


class TestMulticolumnTableFormatting:
    """Regression tests for multicolumn table rendering (fixes Extra alignment tab error)."""

    def test_simple_table_no_multicolumn(self):
        """Basic table without any col_span should work."""
        table = TableBlock(
            id="t1",
            block_type="table",
            rows=[
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="A")]),
                    TableCell(content=[InlineSpan(text="B")]),
                    TableCell(content=[InlineSpan(text="C")]),
                ]),
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="1")]),
                    TableCell(content=[InlineSpan(text="2")]),
                    TableCell(content=[InlineSpan(text="3")]),
                ]),
            ]
        )
        
        latex = _format_table(table)
        
        # Should have 3 columns spec
        assert "{|X|X|X|}" in latex
        # Each row should have exactly 2 ampersands (3 columns = 2 separators)
        data_lines = [l for l in latex.split("\n") if "&" in l]
        for line in data_lines:
            assert line.count("&") == 2, f"Expected 2 ampersands, got {line.count('&')} in: {line}"

    def test_multicolumn_span_2_plus_1(self):
        """REGRESSION TEST: col_span=2 + col_span=1 should produce correct ampersand count.
        
        This is the exact pattern that caused 'Extra alignment tab' errors.
        """
        table = TableBlock(
            id="t2",
            block_type="table",
            rows=[
                # Row 1: 3 normal cells (logical 3 cols)
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="A")]),
                    TableCell(content=[InlineSpan(text="B")]),
                    TableCell(content=[InlineSpan(text="C")]),
                ]),
                # Row 2: multicolumn(2) + 1 normal cell = 3 logical cols
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="Merged")], col_span=2),
                    TableCell(content=[InlineSpan(text="Single")]),
                ]),
            ]
        )
        
        latex = _format_table(table)
        
        # Should have 3 columns spec (determined by max logical cols)
        assert "{|X|X|X|}" in latex
        
        # Row 1: 2 ampersands (A & B & C)
        # Row 2: 1 ampersand (\\multicolumn{2}{...}{Merged} & Single)
        lines = latex.split("\n")
        # Filter for rows containing ampersands (data rows)
        data_lines = [l for l in lines if "&" in l]
        
        # First data row: 2 ampersands (A & B & C)
        assert len(data_lines) >= 2
        assert data_lines[0].count("&") == 2, f"Row 0: expected 2 &, got {data_lines[0]}"
        # Second data row: 1 ampersand (multicolumn consumes 2 logical cols)
        assert data_lines[1].count("&") == 1, f"Row 1: expected 1 &, got {data_lines[1]}"

    def test_multicolumn_full_span(self):
        """Row with single multicolumn spanning all columns should work."""
        table = TableBlock(
            id="t3",
            block_type="table",
            rows=[
                # Row 1: 4 normal cells
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="A")]),
                    TableCell(content=[InlineSpan(text="B")]),
                    TableCell(content=[InlineSpan(text="C")]),
                    TableCell(content=[InlineSpan(text="D")]),
                ]),
                # Row 2: multicolumn(4) - spans everything
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="Full Width")], col_span=4),
                ]),
            ]
        )
        
        latex = _format_table(table)
        
        # Should have 4 columns spec
        assert "{|X|X|X|X|}" in latex
        
        # Row with multicolumn(4) should have 0 ampersands
        lines = [l for l in latex.split("\n") if "Full Width" in l]
        assert len(lines) == 1
        assert lines[0].count("&") == 0, f"Full span row should have 0 &: {lines[0]}"

    def test_multicolumn_mixed_spans(self):
        """Complex row: multicolumn(2) + multicolumn(2) in 4-column table."""
        table = TableBlock(
            id="t4",
            block_type="table",
            rows=[
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="A")]),
                    TableCell(content=[InlineSpan(text="B")]),
                    TableCell(content=[InlineSpan(text="C")]),
                    TableCell(content=[InlineSpan(text="D")]),
                ]),
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="Left")], col_span=2),
                    TableCell(content=[InlineSpan(text="Right")], col_span=2),
                ]),
            ]
        )
        
        latex = _format_table(table)
        
        # Row with two multicolumn(2) should have 1 ampersand
        lines = [l for l in latex.split("\n") if "Left" in l and "Right" in l]
        assert len(lines) == 1
        assert lines[0].count("&") == 1

    def test_row_padding_with_multicolumn(self):
        """Row with multicolumn + fewer cells than max should be padded correctly."""
        table = TableBlock(
            id="t5",
            block_type="table",
            rows=[
                # Row 1: 4 cells = 4 logical cols
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="A")]),
                    TableCell(content=[InlineSpan(text="B")]),
                    TableCell(content=[InlineSpan(text="C")]),
                    TableCell(content=[InlineSpan(text="D")]),
                ]),
                # Row 2: multicolumn(2) + 1 cell = 3 logical cols, needs 1 padding
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="Merged")], col_span=2),
                    TableCell(content=[InlineSpan(text="X")]),
                ]),
            ]
        )
        
        latex = _format_table(table)
        
        # 4-column table
        assert "{|X|X|X|X|}" in latex
        
        # Row 2 should have 2 ampersands: multicolumn(2) & X & (empty padding)
        lines = [l for l in latex.split("\n") if "Merged" in l]
        assert len(lines) == 1
        assert lines[0].count("&") == 2, f"Expected 2 & for padding: {lines[0]}"

    def test_fallback_table_still_renders(self):
        """Tables with policy='fallback' should still render (with warning comment)."""
        table = TableBlock(
            id="t6",
            block_type="table",
            policy="fallback",
            complexity_score=50,
            rows=[
                TableRow(cells=[
                    TableCell(content=[InlineSpan(text="Content")]),
                ]),
            ]
        )
        
        latex = _format_table(table)
        
        # Should contain warning comment
        assert "WARNING" in latex
        assert "complex" in latex.lower()
        # But should still render the table content
        assert "Content" in latex
        assert "tabularx" in latex.lower()


class TestEmptyTableHandling:
    """Tests for edge cases with empty tables."""

    def test_empty_table(self):
        """Empty table should return comment, not error."""
        table = TableBlock(id="t_empty", block_type="table", rows=[])
        latex = _format_table(table)
        assert "Empty table" in latex

    def test_table_with_empty_cells(self):
        """Table with cells containing no text should still render structure."""
        table = TableBlock(
            id="t_sparse",
            block_type="table",
            rows=[
                TableRow(cells=[
                    TableCell(content=[]),
                    TableCell(content=[InlineSpan(text="X")]),
                    TableCell(content=[]),
                ]),
            ]
        )
        
        latex = _format_table(table)
        assert "{|X|X|X|}" in latex
        assert "X" in latex
