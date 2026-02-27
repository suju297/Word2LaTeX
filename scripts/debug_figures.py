
import zipfile
import tempfile
import os
from lxml import etree

NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "v": "urn:schemas-microsoft-com:vml",
}

def debug_figures(docx_path):
    print(f"Inspecting {docx_path}...")
    
    with zipfile.ZipFile(docx_path, 'r') as z:
        xml_content = z.read('word/document.xml')
        
    tree = etree.fromstring(xml_content)
    
    print("\n--- Searching for w:drawing ---")
    drawings = tree.findall(".//w:drawing", NAMESPACES)
    print(f"Found {len(drawings)} w:drawing elements.")
    for i, d in enumerate(drawings):
        print(f"\nDrawing #{i+1}:")
        # Check for inline vs anchor
        inline = d.find(".//wp:inline", NAMESPACES)
        anchor = d.find(".//wp:anchor", NAMESPACES)
        if inline is not None:
            print("  Type: Inline")
            extent = inline.find("wp:extent", NAMESPACES)
            if extent is not None:
                print(f"    Extent: cx={extent.get('cx')}, cy={extent.get('cy')}")
        elif anchor is not None:
            print("  Type: Anchor (Floating)")
            simple_pos = anchor.get("simplePos")
            print(f"    SimplePos: {simple_pos}")
        else:
            print("  Type: Unknown (neither inline nor anchor)")
            
        # Check for blip (image)
        blip = d.find(".//a:blip", NAMESPACES)
        if blip is not None:
            embed = blip.get(f"{{{NAMESPACES['r']}}}embed") 
            print(f"  Image Embed ID: {embed}")
        else:
            print("  No a:blip found directly.")
            # Check for graphicData content
            graphicData = d.find(".//a:graphicData", NAMESPACES)
            if graphicData is not None:
                uri = graphicData.get("uri")
                print(f"  GraphicData URI: {uri}")

    print("\n--- Searching for w:pict (VML) ---")
    picts = tree.findall(".//w:pict", NAMESPACES)
    print(f"Found {len(picts)} w:pict elements.")
    for i, p in enumerate(picts):
        print(f"\nPict #{i+1}:")
        shape = p.find(".//v:shape", NAMESPACES)
        if shape is not None:
             print(f"  Shape ID: {shape.get('id')}")
             imagedata = shape.find(".//v:imagedata", NAMESPACES)
             if imagedata is not None:
                 rid = imagedata.get(f"{{{NAMESPACES['r']}}}id")
                 print(f"  VML Image ID: {rid}")

    print("\n--- Searching for mc:AlternateContent (Fallbacks) ---")
    alts = tree.findall(".//mc:AlternateContent", NAMESPACES)
    print(f"Found {len(alts)} AlternateContent blocks.")

if __name__ == "__main__":
    # Add 'r' namespace dynamically since it might vary but usually standard
    NAMESPACES['r'] = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    
    import sys
    # Hardcoded for now if arg not provided
    path = "tests/corpus/IEEE_TEMPLATE/src.docx"
    debug_figures(path)
