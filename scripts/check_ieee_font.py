import wordtolatex.docx_parser.styles
print(f"Loading styles from: {wordtolatex.docx_parser.styles.__file__}")
from wordtolatex.docx_parser import parse_docx

doc = parse_docx("tests/corpus/IEEE_TEMPLATE/src.docx")
print(f"Detected Body Font: {doc.layout_hints.body_font_family}")
print(f"Detected Line Spacing: {doc.layout_hints.line_spacing}")
print("--- Debug Styles ---")
if "Normal" in doc.styles:
    print(f"Normal: {doc.styles['Normal']}")
if "Default Paragraph Font" in doc.styles:
    print(f"Default Paragraph: {doc.styles['Default Paragraph Font']}")

print(f"All Style Keys: {list(doc.styles.keys())}")
if "_docDefaults" in doc.styles:
    print(f"_docDefaults: {doc.styles['_docDefaults']}")


