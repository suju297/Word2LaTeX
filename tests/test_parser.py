"""Tests for the DOCX parser."""

import pytest
from wordtolatex.ir import HeadingBlock, ParagraphBlock, InlineSpan


class TestInlineSpan:
    """Tests for InlineSpan dataclass."""

    def test_basic_span(self):
        span = InlineSpan(text="Hello")
        assert span.text == "Hello"
        assert span.bold is False
        assert span.italic is False

    def test_formatted_span(self):
        span = InlineSpan(text="Bold", bold=True, italic=True)
        assert span.bold is True
        assert span.italic is True


class TestHeadingBlock:
    """Tests for HeadingBlock."""

    def test_heading_type(self):
        block = HeadingBlock(
            id="b1",
            block_type="heading",  # Required by base Block class
            level=1,
            content=[InlineSpan(text="Title")],
        )
        assert block.block_type == "heading"
        assert block.level == 1


class TestParagraphBlock:
    """Tests for ParagraphBlock."""

    def test_paragraph_type(self):
        block = ParagraphBlock(
            id="b2",
            block_type="paragraph",  # Required by base Block class
            content=[InlineSpan(text="Body text")],
        )
        assert block.block_type == "paragraph"
