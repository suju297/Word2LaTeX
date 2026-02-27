"""Tests for SDT (Structured Document Tag) content extraction in parser."""

import pytest
from lxml import etree
from wordtolatex.docx_parser.document import _parse_runs, NAMESPACES

class TestSDTExtraction:
    """Tests for extracting content from w:sdt elements."""

    def test_sdt_text_extraction(self):
        """Test that text inside simple sdt/sdtContent/r/t is extracted."""
        # <w:p>
        #   <w:sdt>
        #     <w:sdtContent>
        #       <w:r><w:t>Extracted</w:t></w:r>
        #     </w:sdtContent>
        #   </w:sdt>
        # </w:p>
        xml = f"""
        <w:p xmlns:w="{NAMESPACES['w']}">
            <w:sdt>
                <w:sdtContent>
                    <w:r>
                        <w:t>Extracted</w:t>
                    </w:r>
                </w:sdtContent>
            </w:sdt>
        </w:p>
        """
        elem = etree.fromstring(xml)
        spans = _parse_runs(elem)
        
        assert len(spans) == 1
        assert spans[0].text == "Extracted"

    def test_sdt_with_nested_formatting(self):
        """Test that formatting inside SDT is preserved."""
        xml = f"""
        <w:p xmlns:w="{NAMESPACES['w']}">
            <w:sdt>
                <w:sdtContent>
                    <w:r>
                        <w:rPr>
                            <w:b/>
                        </w:rPr>
                        <w:t>BoldText</w:t>
                    </w:r>
                </w:sdtContent>
            </w:sdt>
        </w:p>
        """
        elem = etree.fromstring(xml)
        spans = _parse_runs(elem)
        
        assert len(spans) == 1
        assert spans[0].text == "BoldText"
        assert spans[0].bold is True

    def test_nested_sdt(self):
        """Test recursively nested SDTs."""
        xml = f"""
        <w:p xmlns:w="{NAMESPACES['w']}">
            <w:sdt>
                <w:sdtContent>
                    <w:sdt>
                        <w:sdtContent>
                            <w:r>
                                <w:t>Deep</w:t>
                            </w:r>
                        </w:sdtContent>
                    </w:sdt>
                </w:sdtContent>
            </w:sdt>
        </w:p>
        """
        elem = etree.fromstring(xml)
        spans = _parse_runs(elem)
        
        assert len(spans) == 1
        assert spans[0].text == "Deep"

    def test_mixed_content_sdt(self):
        """Test SDT mixed with regular runs."""
        xml = f"""
        <w:p xmlns:w="{NAMESPACES['w']}">
            <w:r><w:t>Prefix </w:t></w:r>
            <w:sdt>
                <w:sdtContent>
                    <w:r><w:t>Inside</w:t></w:r>
                </w:sdtContent>
            </w:sdt>
            <w:r><w:t> Suffix</w:t></w:r>
        </w:p>
        """
        elem = etree.fromstring(xml)
        spans = _parse_runs(elem)
        
        # Will likely be coalesced if formatting is same
        # _coalesce_spans merges adjacent spans with same formatting
        assert len(spans) == 1
        assert spans[0].text == "Prefix Inside Suffix"
