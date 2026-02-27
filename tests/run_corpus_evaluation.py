
import sys
from pathlib import Path
import fitz

# Import core modules
from wordtolatex.docx_parser import parse_docx
from wordtolatex.dynamic_generator import generate_latex, analyze_document
from wordtolatex.utils import compile_latex
from wordtolatex.calibration import load_corpus_profiles, get_corpus_profile, apply_profile
import yaml

# Import overlay comparison (assuming scripts/page_overlay.py is available)
sys.path.insert(0, 'scripts')
try:
    from page_overlay import compare_documents
except ImportError:
    print("Error: scripts/page_overlay.py not found")
    sys.exit(1)

CORPUS_DIR = Path('tests/corpus')
DOCS = [
    'Resume_1', 'IEEE_TEMPLATE', 'ACM_SUBMISSION_TEMPLATE',
    'Microsoft_Resume1', 'Microsoft_Resume_2',
    'Custom_Resume', 'Resume_2'
]

# Load calibration profiles
PROFILES_PATH = CORPUS_DIR / "calibration_profiles.yaml"
CORPUS_PROFILES = load_corpus_profiles(PROFILES_PATH)
if CORPUS_PROFILES:
    print(f"Loaded {len(CORPUS_PROFILES)} calibration profiles from {PROFILES_PATH}")

print(f"{'Document':<25} {'Pages':<8} {'SSIM':<8} {'Verdict':<20} {'Layout Pattern'}")
print("-" * 90)

results = []

for doc_name in DOCS:
    doc_dir = CORPUS_DIR / doc_name
    src = doc_dir / 'src.docx'
    ref = doc_dir / 'ref.pdf'
    out_tex = doc_dir / 'gen_dynamic_eval.tex'
    out_pdf = doc_dir / 'gen_dynamic_eval.pdf'
    
    if not src.exists() or not ref.exists():
        # print(f"{doc_name:<25} {'SKIP (Missing src/ref)':<40}")
        continue
        
    try:
        # 1. Parse & Analyze
        doc_obj = parse_docx(src, doc_dir)
        
        # Apply Calibration overrides
        profile = get_corpus_profile(doc_name, CORPUS_PROFILES)
        if profile:
            apply_profile(doc_obj, profile)
        
        features = analyze_document(doc_obj, ref)
        pattern_str = features.layout_pattern.pattern.name if features.layout_pattern else "NONE"
        
        # 2. Generate
        latex = generate_latex(doc_obj, ref)
        out_tex.write_text(latex)
        
        # 3. Compile
        if compile_latex(out_tex, out_pdf):
            # 4. Compare
            report = compare_documents(ref, out_pdf, f"{doc_name}_eval")
            
            # Count pages
            with fitz.open(out_pdf) as pdf:
                page_count = len(pdf)
                
            print(f"{doc_name:<25} {page_count:<8} {report.avg_similarity:.4f}   {report.verdict:<20} {pattern_str}")
            
            results.append({
                "name": doc_name,
                "ssim": report.avg_similarity,
                "verdict": report.verdict,
                "pages": page_count,
                "pattern": pattern_str
            })
        else:
            print(f"{doc_name:<25} {'ERR':<8} {'N/A':<8} {'COMPILE_FAIL':<20} {pattern_str}")
            
    except Exception as e:
        print(f"{doc_name:<25} {'ERR':<8} {'N/A':<8} {str(e)[:20]:<20}")

# Summary Stats
if results:
    avg_ssim = sum(r['ssim'] for r in results) / len(results)
    pass_rate = len([r for r in results if 'FAIL' not in r['verdict']]) / len(results) * 100
    print("-" * 90)
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Pass Rate:    {pass_rate:.1f}%")
