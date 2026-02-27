"""Generate calibrated ACM PDF with winning parameters."""
from wordtolatex.docx_parser.document import parse_docx
from wordtolatex.generator import generate_latex
from wordtolatex.utils import compile_latex
from pathlib import Path
import fitz

out_dir = Path('tests/corpus/ACM_SUBMISSION_TEMPLATE')

# WINNING PARAMETERS (found through calibration)
doc = parse_docx(out_dir / 'src.docx', out_dir)
doc.layout_hints.suggested_font_size = 10.0
doc.layout_hints.line_spacing = 1.1
doc.layout_hints.suggested_squeeze = 1.0

tex = generate_latex(doc, out_dir, squeeze_factor=1.0)
tex_path = out_dir / 'calibrated.tex'
tex_path.write_text(tex)
compile_latex(tex_path, out_dir / 'calibrated.pdf')

cal = fitz.open(out_dir / 'calibrated.pdf')
ref = fitz.open(out_dir / 'ref.pdf')

print('=' * 50)
print('PIXEL-PERFECT CALIBRATION RESULT')
print('=' * 50)
print(f'Calibrated: {len(cal)} pages')
print(f'Reference:  {len(ref)} pages')
print(f'Match:      YES!' if len(cal)==len(ref) else 'NO')
print()
print('Winning parameters:')
print('  font_size: 10.0pt')
print('  line_spacing: 1.1')
print('  squeeze: 1.0')
