
import pytest
from lxml import etree
from wordtolatex.docx_parser.document import _parse_drawing, NAMESPACES
from wordtolatex.ir import TextboxBlock

class TestDrawingParser:
    """Tests for _parse_drawing extraction logic."""

    def test_absolute_textbox_complete(self):
        """Test extraction of a fully specified absolute textbox."""
        # 1 inch = 914400 EMUs
        # x = 1.0 in, y = 2.0 in
        # w = 3.0 in, h = 1.0 in
        xml = """
        <w:drawing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                   xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
            <wp:anchor simplePos="0" relativeHeight="0" behindDoc="0" locked="0" layoutInCell="1" allowOverlap="1">
                <wp:simplePos x="0" y="0"/>
                <wp:positionH relativeFrom="page">
                    <wp:posOffset>914400</wp:posOffset>
                </wp:positionH>
                <wp:positionV relativeFrom="page">
                    <wp:posOffset>1828800</wp:posOffset>
                </wp:positionV>
                <wp:extent cx="2743200" cy="914400"/>
                <wp:docPr id="1" name="Text Box 1"/>
                
                <w:txbxContent>
                    <w:p>
                        <w:r>
                            <w:t>Content Inside</w:t>
                        </w:r>
                    </w:p>
                </w:txbxContent>
            </wp:anchor>
        </w:drawing>
        """
        elem = etree.fromstring(xml)
        
        span = _parse_drawing(elem, rels_map={}, styles_map={})
        
        assert span is not None
        assert span.floating_element is not None
        assert isinstance(span.floating_element, TextboxBlock)
        
        tb = span.floating_element
        assert tb.x == 1.0
        assert tb.y == 2.0
        assert tb.width == 3.0
        assert tb.height == 1.0
        
        # Check content blocks
        # _parse_drawing recursively calls _parse_paragraph, which calls _parse_runs, which calls _parse_single_run
        # The inner paragraph should have the text "Content Inside"
        assert len(tb.content_blocks) == 1
        p_block = tb.content_blocks[0]
        # p_block content joins to "Content Inside"
        assert p_block.content[0].text == "Content Inside"

    def test_fallback_inline(self):
        """Test fallback for inline drawing (extract text)."""
        xml = """
        <w:drawing xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
                   xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">
            <wp:inline>
                <w:txbxContent>
                    <w:p>
                        <w:r>
                            <w:t>Fallback Text</w:t>
                        </w:r>
                    </w:p>
                </w:txbxContent>
            </wp:inline>
        </w:drawing>
        """
        elem = etree.fromstring(xml)
        
        span = _parse_drawing(elem, rels_map={}, styles_map={})
        
        assert span is not None
        assert span.floating_element is None
        # It should just extract text
        assert "Fallback Text" in span.text

