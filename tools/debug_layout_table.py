import sys
from lxml import etree
# Mock the module imports to avoid full dependency loading if possible, 
# but we need the actual functions from document.py. 
# We'll rely on the project structure.

try:
    from wordtolatex.docx_parser.document import _is_layout_table, NAMESPACES
    print("Successfully imported _is_layout_table")
except ImportError as e:
    print(f"Import failed: {e}")
    # Fallback to local path import if needed (but uv run should handle it)
    sys.exit(1)

def create_mock_table(borders=False, merged=False, heading=False):
    tbl = etree.Element(f"{{{NAMESPACES['w']}}}tbl")
    tblPr = etree.SubElement(tbl, f"{{{NAMESPACES['w']}}}tblPr")
    
    if borders:
        # Add borders
        tblBorders = etree.SubElement(tblPr, f"{{{NAMESPACES['w']}}}tblBorders")
        top = etree.SubElement(tblBorders, f"{{{NAMESPACES['w']}}}top")
        top.set(f"{{{NAMESPACES['w']}}}val", "single")
    else:
        # Explicit nil borders or just missing (missing is default borderless for layout tables usually?)
        # Actually usually they have borders=nil. Let's add nil borders.
        tblBorders = etree.SubElement(tblPr, f"{{{NAMESPACES['w']}}}tblBorders")
        top = etree.SubElement(tblBorders, f"{{{NAMESPACES['w']}}}top")
        top.set(f"{{{NAMESPACES['w']}}}val", "nil")

    # Create a row
    tr = etree.SubElement(tbl, f"{{{NAMESPACES['w']}}}tr")
    
    # Create 3 cells
    for i in range(3):
        tc = etree.SubElement(tr, f"{{{NAMESPACES['w']}}}tc")
        tcPr = etree.SubElement(tc, f"{{{NAMESPACES['w']}}}tcPr")
        
        if merged and i == 0:
            gridSpan = etree.SubElement(tcPr, f"{{{NAMESPACES['w']}}}gridSpan")
            gridSpan.set(f"{{{NAMESPACES['w']}}}val", "2")
            
        p = etree.SubElement(tc, f"{{{NAMESPACES['w']}}}p")
        if heading and i == 0:
            pPr = etree.SubElement(p, f"{{{NAMESPACES['w']}}}pPr")
            pStyle = etree.SubElement(pPr, f"{{{NAMESPACES['w']}}}pStyle")
            pStyle.set(f"{{{NAMESPACES['w']}}}val", "Heading1")

    return tbl

print("\n--- Test 1: Data Table (Borders) ---")
t1 = create_mock_table(borders=True)
is_layout_1 = _is_layout_table(t1)
print(f"Is Layout Table? {is_layout_1} (Expected: False)")

print("\n--- Test 2: Layout Table (No Borders, Merged, Heading) ---")
t2 = create_mock_table(borders=False, merged=True, heading=True)
is_layout_2 = _is_layout_table(t2)
print(f"Is Layout Table? {is_layout_2} (Expected: True)")

print("\n--- Test 3: Simple Table (No Borders, No Merges, No Heading) ---")
t3 = create_mock_table(borders=False, merged=False, heading=False)
is_layout_3 = _is_layout_table(t3)
print(f"Is Layout Table? {is_layout_3} (Expected: False - default)")
