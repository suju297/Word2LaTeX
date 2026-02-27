import unittest
from lxml import etree
from wordtolatex.docx_parser.document import _is_layout_table, _flatten_layout_table, NAMESPACES

class TestLayoutFlattening(unittest.TestCase):
    def setUp(self):
        self.ns = NAMESPACES

    def create_table(self, borders=False, merged=False, heading=False):
        tbl = etree.Element(f"{{{self.ns['w']}}}tbl")
        tblPr = etree.SubElement(tbl, f"{{{self.ns['w']}}}tblPr")
        
        # Borders
        tblBorders = etree.SubElement(tblPr, f"{{{self.ns['w']}}}tblBorders")
        if borders:
            top = etree.SubElement(tblBorders, f"{{{self.ns['w']}}}top")
            top.set(f"{{{self.ns['w']}}}val", "single")
        else:
            # Explicit nil or just missing borders implies borderless for typical layout tables
            # But let's be explicit for the test
            top = etree.SubElement(tblBorders, f"{{{self.ns['w']}}}top")
            top.set(f"{{{self.ns['w']}}}val", "nil")

        # Row
        tr = etree.SubElement(tbl, f"{{{self.ns['w']}}}tr")
        
        # Cells
        for i in range(3):
            tc = etree.SubElement(tr, f"{{{self.ns['w']}}}tc")
            tcPr = etree.SubElement(tc, f"{{{self.ns['w']}}}tcPr")
            
            if merged and i == 0:
                gridSpan = etree.SubElement(tcPr, f"{{{self.ns['w']}}}gridSpan")
                gridSpan.set(f"{{{self.ns['w']}}}val", "2")
            
            p = etree.SubElement(tc, f"{{{self.ns['w']}}}p")
            if heading and i == 0:
                pPr = etree.SubElement(p, f"{{{self.ns['w']}}}pPr")
                pStyle = etree.SubElement(pPr, f"{{{self.ns['w']}}}pStyle")
                pStyle.set(f"{{{self.ns['w']}}}val", "Heading1")
                
            # Add text run
            r = etree.SubElement(p, f"{{{self.ns['w']}}}r")
            t = etree.SubElement(r, f"{{{self.ns['w']}}}t")
            t.text = f"Content {i}"

        return tbl

    def test_is_layout_table_detection(self):
        # Case 1: Data table (Has borders) -> Should be False
        t1 = self.create_table(borders=True)
        self.assertFalse(_is_layout_table(t1), "Data table with borders should NOT be a layout table")

        # Case 2: Layout table (No borders, merged cells, headings) -> Should be True
        t2 = self.create_table(borders=False, merged=True, heading=True)
        self.assertTrue(_is_layout_table(t2), "Borderless table with merges/headings SHOULD be a layout table")

        # Case 3: Plain borderless table (No merges/headings) -> Should be False (default to data)
        # Note: Our heuristic requires >20% merge or >10% heading
        t3 = self.create_table(borders=False, merged=False, heading=False)
        self.assertFalse(_is_layout_table(t3), "Plain borderless table without signals should default to data table")

    def test_flatten_layout_table(self):
        # Create a layout table with 3 cells
        t = self.create_table(borders=False, merged=True, heading=True)
        
        # Flatten it
        # We need to mock styles_map and rels_map
        styles = {}
        rels = {}
        
        # Note: _flatten_layout_table calls _parse_paragraph which uses styles_map
        # We need to make sure _parse_paragraph works. 
        # Since we can't easily mock the internal _parse_paragraph call without patching,
        # we rely on it working with empty maps.
        
        blocks = _flatten_layout_table(t, 0, styles, rels)
        
        # Expect 3 blocks (one per cell paragraph)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0].content[0].text, "Content 0")
        self.assertEqual(blocks[1].content[0].text, "Content 1")
        self.assertEqual(blocks[2].content[0].text, "Content 2")
        
        # First block should be HeadingBlock (Heading1)
        self.assertEqual(blocks[0].block_type, "heading")
        self.assertEqual(blocks[0].level, 1)

if __name__ == '__main__':
    unittest.main()
