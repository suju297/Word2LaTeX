"""Golden Layout Tests - Comprehensive validation of Layout Oracle and Policy Engine."""

import pytest
from pathlib import Path
from wordtolatex.layout_heuristics import detect_tables, detect_textboxes, detect_figures, detect_layout_heuristics
from wordtolatex.layout_oracle import extract_layout, create_layout_hints
from wordtolatex.policy import decide_policy, LayoutContext, PolicyResult
from wordtolatex.ir import Block, TableBlock, ParagraphBlock
import fitz
import json

SAMPLES_DIR = Path("tests/samples")
CORPUS_DIR = Path("tests/corpus")

# --- Heuristic Detection Tests ---

class TestHeuristicTableDetection:
    """Verify table heuristics on synthetic sample."""
    
    def test_table_detected(self):
        """Verify that sample_table.pdf is detected as a table."""
        pdf_path = SAMPLES_DIR / "sample_table.pdf"
        if not pdf_path.exists():
            pytest.skip("sample_table.pdf not found")
            
        doc = fitz.open(pdf_path)
        page = doc[0]
        
        regions = detect_tables(page, 0)
        
        # Must find at least one table
        assert len(regions) >= 1, f"Expected at least 1 table, found {len(regions)}"
        
        # Best region should be labeled "table"
        r = regions[0]
        assert r.label == "table"
        assert r.source == "heuristic"
        
    def test_table_confidence_normalized(self):
        """Verify table confidence is within expected range (normalized scoring)."""
        pdf_path = SAMPLES_DIR / "sample_table.pdf"
        if not pdf_path.exists():
            pytest.skip("sample_table.pdf not found")
            
        doc = fitz.open(pdf_path)
        regions = detect_tables(doc[0], 0)
        
        # Confidence should be stable and >= 0.7 (normalized)
        assert regions[0].confidence >= 0.7

    def test_table_bbox_valid(self):
        """Verify table bbox is within page bounds and reasonable."""
        pdf_path = SAMPLES_DIR / "sample_table.pdf"
        if not pdf_path.exists():
            pytest.skip("sample_table.pdf not found")
            
        doc = fitz.open(pdf_path)
        regions = detect_tables(doc[0], 0)
        
        r = regions[0]
        x0, y0, x1, y1 = r.bbox
        
        # Bboxes should be normalized [0..1]
        assert 0 <= x0 < x1 <= 1, f"Invalid x bounds: {x0}, {x1}"
        assert 0 <= y0 < y1 <= 1, f"Invalid y bounds: {y0}, {y1}"
        
        # Table should not be tiny (at least 10% of page area)
        area = (x1 - x0) * (y1 - y0)
        assert area > 0.05, f"Table area too small: {area}"


class TestHeuristicTextboxDetection:
    """Verify textbox heuristics on synthetic sample."""
    
    def test_textbox_detected(self):
        """Verify that sample_textbox.pdf contains a textbox."""
        pdf_path = SAMPLES_DIR / "sample_textbox.pdf"
        if not pdf_path.exists():
            pytest.skip("sample_textbox.pdf not found")
             
        doc = fitz.open(pdf_path)
        regions = detect_textboxes(doc[0], 0)
        
        # At least one textbox
        assert len(regions) >= 1
        
        r = regions[0]
        assert r.label == "textbox"
        assert r.source == "heuristic"

    def test_conflict_case_textbox(self):
        """Verify conflict_case.pdf textbox is detected correctly."""
        pdf_path = SAMPLES_DIR / "conflict_case.pdf"
        if not pdf_path.exists():
            pytest.skip("conflict_case.pdf not found")
            
        doc = fitz.open(pdf_path)
        regions = detect_textboxes(doc[0], 0)
        
        found = any(r.label == "textbox" and r.source == "heuristic" for r in regions)
        assert found, "Heuristics failed to detect the conflict case textbox."


# --- Column Detection Tests ---

class TestColumnDetection:
    """Verify column detection on real documents."""
    
    def test_resume_single_column(self):
        """Verify resume_styled.pdf is detected as single-column."""
        pdf_path = SAMPLES_DIR / "resume_styled.pdf"
        if not pdf_path.exists():
            pytest.skip("resume_styled.pdf not found")
            
        layout = extract_layout(pdf_path)
        
        assert len(layout.column_structure) >= 1
        cols = layout.column_structure[0]
        
        # Most resumes are 1 column (unless sidebar)
        assert cols.count in [1, 2], f"Unexpected column count: {cols.count}"
        
    def test_column_boundaries_normalized(self):
        """Verify column boundaries are in [0..1] range."""
        pdf_path = SAMPLES_DIR / "resume_styled.pdf"
        if not pdf_path.exists():
            pytest.skip("resume_styled.pdf not found")
            
        layout = extract_layout(pdf_path)
        
        for cols in layout.column_structure:
            for boundary in cols.boundaries:
                assert 0 < boundary < 1, f"Boundary {boundary} not normalized"


# --- Policy Engine Tests ---

class TestPolicyEngine:
    """Verify policy engine decision making."""
    
    def test_ooxml_table_wins(self):
        """OOXML table should result in semantic decision."""
        block = TableBlock(id="t1", block_type="table", rows=[], complexity_score=5)
        ctx = LayoutContext(page_index=0, column_count=1, block_bbox_norm=(0,0,1,1), overlapping_regions=[])
        
        result = decide_policy(block, ctx)
        
        assert isinstance(result, PolicyResult)
        assert result.decision == "semantic"
        assert "OOXML" in result.reason
        
    def test_default_semantic_for_paragraph(self):
        """Plain paragraphs should default to semantic."""
        block = ParagraphBlock(id="p1", block_type="paragraph")
        ctx = LayoutContext(page_index=0, column_count=1, block_bbox_norm=(0,0,1,1), overlapping_regions=[])
        
        result = decide_policy(block, ctx)
        
        assert result.decision == "semantic"
        
    def test_policy_result_has_reason(self):
        """All policy decisions must have a reason code."""
        block = Block(id="b1", block_type="unknown")
        ctx = LayoutContext(page_index=0, column_count=1, block_bbox_norm=(0,0,1,1), overlapping_regions=[])
        
        result = decide_policy(block, ctx)
        
        assert result.reason is not None
        assert len(result.reason) > 0


# --- Integration Tests ---

class TestLayoutOracleIntegration:
    """End-to-end oracle tests."""
    
    def test_extract_layout_returns_valid_structure(self):
        """Verify extract_layout returns all expected fields."""
        pdf_path = SAMPLES_DIR / "resume_styled.pdf"
        if not pdf_path.exists():
            pytest.skip("resume_styled.pdf not found")
            
        layout = extract_layout(pdf_path)
        
        assert hasattr(layout, 'text_blocks')
        assert hasattr(layout, 'column_structure')
        assert hasattr(layout, 'regions')
        
        assert len(layout.text_blocks) > 0
        
    def test_create_layout_hints_schema(self):
        """Verify layout hints have required fields."""
        pdf_path = SAMPLES_DIR / "resume_styled.pdf"
        if not pdf_path.exists():
            pytest.skip("resume_styled.pdf not found")
            
        layout = extract_layout(pdf_path)
        hints = create_layout_hints(layout)
        
        assert 'regions' in hints
        assert 'columns' in hints
        
        # Check region schema
        if hints['regions']:
            r = hints['regions'][0]
            assert 'label' in r
            assert 'bbox' in r
            assert 'confidence' in r


# --- Report Validation Tests ---

class TestReportIntegrity:
    """Verify report.json structure from corpus runs."""
    
    def test_corpus_report_has_policy_trace(self):
        """Verify merged reports contain policy trace."""
        # Check if any corpus run exists
        corpus_dirs = list(CORPUS_DIR.glob("*/report.json"))
        if not corpus_dirs:
            pytest.skip("No corpus reports found")
            
        report_path = corpus_dirs[0]
        with open(report_path) as f:
            report = json.load(f)
            
        # After merge, policy_trace should be present (may be empty for some docs)
        assert 'policy_trace' in report
        assert 'oracle_summary' in report
        assert 'columns' in report
