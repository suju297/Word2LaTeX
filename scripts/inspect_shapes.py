from wordtolatex.docx_parser.document import parse_docx, NAMESPACES
from pathlib import Path
from lxml import etree
import json
import zipfile

DOC_PATH = Path("tests/corpus/Resume_1/src.docx")

def inspect_docx(path):
    print(f"Inspecting {path}...")
    with zipfile.ZipFile(path, "r") as zf:
        xml = zf.read("word/document.xml")
        tree = etree.fromstring(xml)
        
        # Find all drawings
        drawings = tree.xpath("//w:drawing", namespaces=NAMESPACES)
        print(f"Found {len(drawings)} w:drawing elements.")
        
        for i, dr in enumerate(drawings):
            # Check for anchor vs inline
            anchor = dr.xpath(".//wp:anchor", namespaces=NAMESPACES)
            inline = dr.xpath(".//wp:inline", namespaces=NAMESPACES)
            pos_type = "anchor" if anchor else ("inline" if inline else "unknown")
            
            # Check for textbox
            txbx = dr.xpath(".//w:txbxContent", namespaces=NAMESPACES)
            
            print(f"  Drawing #{i}: {pos_type}")
            if txbx:
                print("    - Contains Textbox")
                # Dump text content
                texts = dr.xpath(".//w:t/text()", namespaces=NAMESPACES)
                print(f"    - Text sample: {texts[:3]}")
            
            pic = dr.xpath(".//pic:pic", namespaces=NAMESPACES)
            if pic:
                print("    - Contains Picture")
            
            if i == 0:
                print("    Children Tags:")
                for child in dr.iter():
                     # Simplify tag by removing namespace for readability
                     tag =  child.tag.split("}")[-1] if "}" in child.tag else child.tag
                     print(f"      {tag}")
            
            pic = dr.xpath(".//pic:pic", namespaces=NAMESPACES)
            if pic:
                print("    - Contains Picture")

        # Find VML shapes
        shapes = tree.xpath("//v:shape", namespaces=NAMESPACES)
        print(f"Found {len(shapes)} v:shape elements.")
        for i, sh in enumerate(shapes):
            style = sh.get("style", "")
            print(f"  VML Shape #{i}: style='{style}'")
            # Check for textpath or textbox
            textbox = sh.xpath(".//v:textbox", namespaces=NAMESPACES)
            if textbox:
                 print("    - Contains VML Textbox")
            texts = sh.xpath(".//w:t/text()", namespaces=NAMESPACES)
            if texts:
                print(f"    - Text: {texts[:3]}")

        # Find WPS shapes
        wps_shapes = tree.xpath("//wps:wsp", namespaces=NAMESPACES)
        print(f"Found {len(wps_shapes)} wps:wsp elements.")
        for i, sh in enumerate(wps_shapes):
            print(f"  WPS Shape #{i}")
            # Check for textbox
            txbx = sh.xpath(".//wps:txbx", namespaces=NAMESPACES)
            if txbx:
                print("    - Contains WPS Textbox")
            texts = sh.xpath(".//w:t/text()", namespaces=NAMESPACES)
            if texts:
                print(f"    - Text: {texts[:3]}")

if __name__ == "__main__":
    inspect_docx(DOC_PATH)
