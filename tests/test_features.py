"""Feature tests using generated samples and snapshots."""
import pytest
from pathlib import Path
from wordtolatex.docx_parser import parse_docx
from wordtolatex.generator import generate_latex
from wordtolatex.policy import decide_policy

SAMPLES_DIR = Path(__file__).parent / "samples"

@pytest.fixture
def samples_dir():
    return SAMPLES_DIR

def convert_to_latex(docx_path):
    """Helper to run the full pipeline without I/O."""
    doc = parse_docx(docx_path)
    for block in doc.blocks:
        block.policy = decide_policy(block)
    return generate_latex(doc)

def test_feature_text_formatting(samples_dir, snapshot):
    """Test bold, italic, underline."""
    input_path = samples_dir / "feature_text.docx"
    assert input_path.exists(), "Run tests/generators.py first"
    
    latex = convert_to_latex(input_path)
    snapshot.assert_match(latex, "feature_text.tex")

def test_feature_lists(samples_dir, snapshot):
    """Test ordered and unordered lists."""
    input_path = samples_dir / "feature_lists.docx"
    latex = convert_to_latex(input_path)
    snapshot.assert_match(latex, "feature_lists.tex")

def test_feature_tables(samples_dir, snapshot):
    """Test tables."""
    input_path = samples_dir / "feature_tables.docx"
    latex = convert_to_latex(input_path)
    snapshot.assert_match(latex, "feature_tables.tex")
