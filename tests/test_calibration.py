"""Tests for the calibration module."""

import pytest
from pathlib import Path

from wordtolatex.calibration import (
    CalibrationProfile,
    PROFILES,
    detect_document_type,
    get_profile,
    apply_profile,
    load_corpus_profiles,
    get_corpus_profile,
)
from wordtolatex.ir import Document, Section, HeadingBlock, ParagraphBlock, LayoutGridBlock, LayoutHints


class TestCalibrationProfile:
    """Tests for CalibrationProfile dataclass."""
    
    def test_default_values(self):
        """Test default profile values."""
        profile = CalibrationProfile(name="test")
        assert profile.font_size == 10.0
        assert profile.line_spacing == 1.15
        assert profile.squeeze_factor == 1.0
        assert profile.section_spacing == 1.0
    
    def test_custom_values(self):
        """Test custom profile values."""
        profile = CalibrationProfile(
            name="custom",
            font_size=9.0,
            line_spacing=0.9,
            squeeze_factor=0.8,
            section_spacing=0.7,
        )
        assert profile.font_size == 9.0
        assert profile.line_spacing == 0.9
    
    def test_to_dict(self):
        """Test profile serialization."""
        profile = CalibrationProfile(name="test", font_size=11.0)
        d = profile.to_dict()
        assert d["name"] == "test"
        assert d["font_size"] == 11.0


class TestBuiltinProfiles:
    """Tests for built-in calibration profiles."""
    
    def test_profiles_exist(self):
        """Verify all expected profiles exist."""
        assert "academic_twocol" in PROFILES
        assert "academic_singlecol" in PROFILES
        assert "resume" in PROFILES
        assert "default" in PROFILES
    
    def test_academic_twocol_tighter(self):
        """Academic two-column should have tighter spacing than default."""
        academic = PROFILES["academic_twocol"]
        default = PROFILES["default"]
        assert academic.line_spacing <= default.line_spacing
        assert academic.section_spacing <= default.section_spacing
    
    def test_resume_compact(self):
        """Resume profile should have compact section spacing."""
        resume = PROFILES["resume"]
        assert resume.section_spacing < 1.0


class TestDocumentTypeDetection:
    """Tests for automatic document type detection."""
    
    def _create_doc_with_headings(self, headings: list, col_count: int = 1) -> Document:
        """Helper to create a document with specified headings."""
        blocks = []
        for i, (text, level) in enumerate(headings):
            block = HeadingBlock(
                id=f"h{i}",
                block_type="heading",
                level=level,
                content=[type("Span", (), {"text": text})()],
            )
            blocks.append(block)
        
        return Document(
            sections=[Section(col_count=col_count)],
            blocks=blocks,
            layout_hints=LayoutHints(),
        )
    
    def test_detect_academic_twocol(self):
        """Two-column layout with academic headings → academic_twocol."""
        doc = self._create_doc_with_headings(
            [("Introduction", 1), ("Methodology", 1), ("Conclusion", 1)],
            col_count=2,
        )
        assert detect_document_type(doc) == "academic_twocol"
    
    def test_detect_academic_singlecol(self):
        """Single-column with many academic headings → academic_singlecol."""
        doc = self._create_doc_with_headings(
            [("Abstract", 1), ("Introduction", 1), ("Related Work", 1), 
             ("Methodology", 1), ("Conclusion", 1)],
            col_count=1,
        )
        assert detect_document_type(doc) == "academic_singlecol"
    
    def test_detect_resume_by_headings(self):
        """Resume-style headings → resume."""
        doc = self._create_doc_with_headings(
            [("Experience", 1), ("Education", 1), ("Skills", 1)],
            col_count=1,
        )
        assert detect_document_type(doc) == "resume"
    
    def test_detect_resume_by_layout_grids(self):
        """Many layout grids → resume."""
        doc = Document(
            sections=[Section(col_count=1)],
            blocks=[
                LayoutGridBlock(id="lg1", block_type="layout_grid"),
                LayoutGridBlock(id="lg2", block_type="layout_grid"),
                LayoutGridBlock(id="lg3", block_type="layout_grid"),
            ],
            layout_hints=LayoutHints(),
        )
        assert detect_document_type(doc) == "resume"
    
    def test_detect_default_fallback(self):
        """No strong signals → default."""
        doc = Document(
            sections=[Section(col_count=1)],
            blocks=[ParagraphBlock(id="p1", block_type="paragraph")],
            layout_hints=LayoutHints(),
        )
        assert detect_document_type(doc) == "default"


class TestGetProfile:
    """Tests for get_profile function."""
    
    def test_get_known_profile(self):
        """Get a known profile by name."""
        profile = get_profile("academic_twocol")
        assert profile.name == "academic_twocol"
    
    def test_get_profile_alias(self):
        """'academic' should alias to 'academic_singlecol'."""
        profile = get_profile("academic")
        assert profile.name == "academic_singlecol"
    
    def test_get_unknown_returns_default(self):
        """Unknown profile name returns default."""
        profile = get_profile("nonexistent_profile")
        assert profile.name == "default"
    
    def test_case_insensitive(self):
        """Profile lookup should be case-insensitive."""
        profile = get_profile("RESUME")
        assert profile.name == "resume"


class TestApplyProfile:
    """Tests for apply_profile function."""
    
    def test_applies_font_size(self):
        """Profile font_size applied to suggested_font_size."""
        doc = Document(layout_hints=LayoutHints())
        profile = CalibrationProfile(name="test", font_size=9.5)
        apply_profile(doc, profile)
        assert doc.layout_hints.suggested_font_size == 9.5
    
    def test_applies_line_spacing(self):
        """Profile line_spacing applied."""
        doc = Document(layout_hints=LayoutHints())
        profile = CalibrationProfile(name="test", line_spacing=1.2)
        apply_profile(doc, profile)
        assert doc.layout_hints.line_spacing == 1.2
    
    def test_does_not_override_existing_font_size(self):
        """If suggested_font_size already set, don't override."""
        doc = Document(layout_hints=LayoutHints(suggested_font_size=8.0))
        profile = CalibrationProfile(name="test", font_size=10.0)
        apply_profile(doc, profile)
        assert doc.layout_hints.suggested_font_size == 8.0  # Not overridden


class TestCorpusProfiles:
    """Tests for corpus-specific profile loading."""
    
    def test_load_missing_file(self):
        """Missing YAML file returns empty dict."""
        result = load_corpus_profiles(Path("/nonexistent/path.yaml"))
        assert result == {}
    
    def test_get_corpus_profile_missing(self):
        """Missing document returns None."""
        result = get_corpus_profile("UnknownDoc", {})
        assert result is None
    
    def test_get_corpus_profile_with_override(self):
        """Corpus profile with overrides creates custom profile."""
        corpus = {
            "TestDoc": {
                "profile": "academic_twocol",
                "overrides": {"font_size": 8.5}
            }
        }
        profile = get_corpus_profile("TestDoc", corpus)
        assert profile is not None
        assert profile.font_size == 8.5
        assert "calibrated" in profile.name
