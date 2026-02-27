"""Calibration Module - Document type detection and layout calibration profiles.

This module provides automatic detection of document types (academic, resume, etc.)
and applies appropriate calibration profiles to achieve consistent layout fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
import yaml

from wordtolatex.ir import Document, HeadingBlock, LayoutGridBlock


@dataclass
class CalibrationProfile:
    """Layout calibration parameters for a document type."""
    
    name: str
    font_size: float = 10.0        # Target base font size (pt)
    line_spacing: float = 1.15     # Line spacing multiplier
    squeeze_factor: float = 1.0    # Global spacing reduction factor
    section_spacing: float = 1.0   # Section title spacing multiplier
    
    def to_dict(self) -> Dict:
        """Convert profile to dictionary for serialization."""
        return {
            "name": self.name,
            "font_size": self.font_size,
            "line_spacing": self.line_spacing,
            "squeeze_factor": self.squeeze_factor,
            "section_spacing": self.section_spacing,
        }


# Built-in calibration profiles based on corpus analysis
PROFILES: Dict[str, CalibrationProfile] = {
    # Academic two-column papers (ACM, IEEE style)
    # Tight spacing, smaller fonts for dense content
    "academic_twocol": CalibrationProfile(
        name="academic_twocol",
        font_size=10.0,      # ACM calibration: 10pt matched 13-page ref
        line_spacing=1.1,    # Tighter than default
        squeeze_factor=1.0,
        section_spacing=0.8,
    ),
    
    # Academic single-column papers
    "academic_singlecol": CalibrationProfile(
        name="academic_singlecol",
        font_size=11.0,
        line_spacing=1.15,
        squeeze_factor=1.0,
        section_spacing=1.0,
    ),
    
    # Resume/CV documents
    # Compact layout, often uses layout grids
    "resume": CalibrationProfile(
        name="resume",
        font_size=10.5,
        line_spacing=1.0,
        squeeze_factor=1.0,
        section_spacing=0.7,
    ),
    
    # Default fallback profile
    "default": CalibrationProfile(
        name="default",
        font_size=11.0,
        line_spacing=1.15,
        squeeze_factor=1.0,
        section_spacing=1.0,
    ),
}


# Document type detection patterns
ACADEMIC_HEADING_PATTERNS = [
    "introduction", "conclusion", "abstract", "appendix", "references",
    "acknowledgment", "acknowledgement", "methodology", "results",
    "discussion", "related work", "background", "evaluation", "experiments",
    "nomenclature", "paper preparation", "template", "literature review",
    "future work", "proposed method", "experimental setup",
]

RESUME_HEADING_PATTERNS = [
    "education", "experience", "skills", "projects", "summary", "objective",
    "certifications", "awards", "publications", "references", "contact",
    "technical skills", "work experience", "academic projects", 
    "professional experience", "employment", "qualifications",
]


def detect_document_type(doc: Document) -> str:
    """Detect document type based on structural signals.
    
    Detection signals:
    1. Column count (2-col → academic)
    2. Heading patterns (academic vs resume keywords)
    3. Layout grid presence (many → resume)
    4. Heading count and structure
    
    Returns:
        str: One of "academic_twocol", "academic_singlecol", "resume", "default"
    """
    override = doc.metadata.get("slm_doc_type") if hasattr(doc, "metadata") else None
    if override in PROFILES:
        return override

    # Signal 1: Column count
    is_two_column = False
    if doc.sections:
        is_two_column = doc.sections[0].col_count > 1
    
    # Signal 2: Count heading patterns
    academic_score = 0
    resume_score = 0
    heading_count = 0
    
    for block in doc.blocks:
        if isinstance(block, HeadingBlock):
            heading_count += 1
            # Extract text from heading
            text = "".join(span.text for span in block.content).strip().lower()
            
            for pattern in ACADEMIC_HEADING_PATTERNS:
                if pattern in text:
                    academic_score += 2 if block.level == 1 else 1
                    break
            
            for pattern in RESUME_HEADING_PATTERNS:
                if pattern in text:
                    resume_score += 2 if block.level == 1 else 1
                    break
    
    # Signal 3: Layout grid count
    layout_grid_count = sum(
        1 for block in doc.blocks 
        if isinstance(block, LayoutGridBlock)
    )
    
    # Decision logic
    if is_two_column and academic_score >= 2:
        return "academic_twocol"
    
    if academic_score >= 4:  # Strong academic signal even without two-column
        return "academic_singlecol"
    
    if resume_score >= 3 or layout_grid_count >= 2:
        return "resume"
    
    if academic_score > resume_score:
        return "academic_singlecol"
    
    return "default"


def get_profile(name: str) -> CalibrationProfile:
    """Get a calibration profile by name.
    
    Args:
        name: Profile name (academic_twocol, resume, default, etc.)
        
    Returns:
        CalibrationProfile for the requested type
    """
    # Normalize name (handle aliases)
    name = name.lower().strip()
    if name == "academic":
        name = "academic_singlecol"
    
    return PROFILES.get(name, PROFILES["default"])


def apply_profile(doc: Document, profile: CalibrationProfile) -> None:
    """Apply calibration profile to document layout hints.
    
    Updates doc.layout_hints with values from the calibration profile.
    
    Args:
        doc: Document IR to update
        profile: Calibration profile to apply
    """
    hints = doc.layout_hints
    
    # Apply font size (only if not already set by PDF metrics)
    if hints.suggested_font_size is None:
        hints.suggested_font_size = profile.font_size
    
    # Apply line spacing
    hints.line_spacing = profile.line_spacing
    
    # Apply squeeze factor
    hints.suggested_squeeze = profile.squeeze_factor
    
    # Apply section spacing
    if hints.suggested_section_spacing is None:
        hints.suggested_section_spacing = profile.section_spacing


def calibrate_from_pdf(ref_pdf: Path, doc: Document) -> CalibrationProfile:
    """Create a calibration profile by analyzing a reference PDF.
    
    Uses pdf_metrics to extract font size, line height from the reference,
    then computes correction factors to match the reference.
    
    Args:
        ref_pdf: Path to reference PDF
        doc: Document IR (for current layout hints)
        
    Returns:
        CalibrationProfile tuned to match the reference PDF
    """
    from wordtolatex.pdf_metrics import extract_metrics
    
    ref_metrics = extract_metrics(ref_pdf)
    
    # Start with detected document type as base
    doc_type = detect_document_type(doc)
    base_profile = get_profile(doc_type)
    
    # Create new profile with measured values
    return CalibrationProfile(
        name=f"calibrated_{doc_type}",
        font_size=ref_metrics.avg_font_size,  # Use actual measured font size
        line_spacing=base_profile.line_spacing,  # Keep base line spacing
        squeeze_factor=base_profile.squeeze_factor,
        section_spacing=base_profile.section_spacing,
    )


def load_corpus_profiles(yaml_path: Path) -> Dict[str, Dict]:
    """Load per-document calibration profiles from YAML file.
    
    Args:
        yaml_path: Path to calibration_profiles.yaml
        
    Returns:
        Dict mapping document names to their calibration configs
    """
    if not yaml_path.exists():
        return {}
    
    with open(yaml_path) as f:
        return yaml.safe_load(f) or {}


def get_corpus_profile(doc_name: str, corpus_profiles: Dict) -> Optional[CalibrationProfile]:
    """Get calibration profile for a specific corpus document.
    
    Args:
        doc_name: Name of the document (e.g., "ACM_SUBMISSION_TEMPLATE")
        corpus_profiles: Loaded corpus profiles dict
        
    Returns:
        CalibrationProfile if found, None otherwise
    """
    config = corpus_profiles.get(doc_name)
    if not config:
        return None
    
    # Get base profile
    base_name = config.get("profile", "default")
    profile = get_profile(base_name)
    
    # Apply overrides
    overrides = config.get("overrides", {})
    if overrides:
        return CalibrationProfile(
            name=f"{doc_name}_calibrated",
            font_size=overrides.get("font_size", profile.font_size),
            line_spacing=overrides.get("line_spacing", profile.line_spacing),
            squeeze_factor=overrides.get("squeeze_factor", profile.squeeze_factor),
            section_spacing=overrides.get("section_spacing", profile.section_spacing),
        )
    
    return profile
