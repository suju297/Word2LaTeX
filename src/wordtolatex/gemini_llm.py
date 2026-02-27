"""Gemini LLM Integration for Advanced Document Analysis.

This module uses Google's Gemini 2.5 Flash model for:
1. Complex paragraph classification (addresses, signatures, etc.)
2. Style inference for documents without explicit Word styles
3. Table decision-making (semantic vs image fallback)
"""

from __future__ import annotations

import os
import json
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path
import PIL.Image

import warnings
warnings.simplefilter("ignore")  # Suppress generic warnings including FutureWarning

# Load .env file if present
try:
    from dotenv import load_dotenv
    # Look for .env in project root (3 levels up from here: src/wordtolatex/gemini_llm.py -> src/wordtolatex -> src -> root)
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

# Lazy import for google-generativeai
_genai = None
_model = None


@dataclass
class DocumentAnalysis:
    """Result of Gemini document analysis."""
    heading_suggestions: List[Dict[str, Any]] = None
    paragraph_types: List[Dict[str, str]] = None
    table_decisions: List[Dict[str, str]] = None
    style_recommendations: Dict[str, Any] = None


def _init_gemini():
    """Initialize Gemini client lazily."""
    global _genai, _model
    if _model is not None:
        return _model
    
    import google.generativeai as genai
    _genai = genai
    
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
        )
    
    genai.configure(api_key=api_key)
    _model = genai.GenerativeModel("gemini-2.0-flash")
    return _model


def analyze_document_structure(text_content: str, context: str = "resume") -> DocumentAnalysis:
    """Use Gemini to analyze document structure and suggest improvements.
    
    Args:
        text_content: Plain text content of the document
        context: Document type hint (resume, report, article, etc.)
        
    Returns:
        DocumentAnalysis with suggestions
    """
    model = _init_gemini()
    
    prompt = f"""Analyze this {context} document and provide structured suggestions.

DOCUMENT CONTENT:
{text_content[:4000]}  # Limit to avoid token limits

TASK: Return a JSON object with:
1. "heading_suggestions": List of text segments that should be headings with suggested levels (1-3)
2. "paragraph_types": List of special paragraph types detected (e.g., "address", "date", "signature")
3. "style_recommendations": Object with font_size, line_spacing, margin suggestions

Example output:
{{
    "heading_suggestions": [
        {{"text": "EDUCATION", "level": 1}},
        {{"text": "EXPERIENCE", "level": 1}}
    ],
    "paragraph_types": [
        {{"text": "Boston, MA | (857)...", "type": "contact_info"}}
    ],
    "style_recommendations": {{
        "font_size": 10,
        "line_spacing": 1.15,
        "margins": 0.5
    }}
}}

Return ONLY valid JSON, no markdown code blocks."""

    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # Parse JSON response
        if result_text.startswith("```"):
            # Strip markdown code blocks if present
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        data = json.loads(result_text)
        
        return DocumentAnalysis(
            heading_suggestions=data.get("heading_suggestions", []),
            paragraph_types=data.get("paragraph_types", []),
            style_recommendations=data.get("style_recommendations", {}),
        )
    except Exception as e:
        print(f"Warning: Gemini analysis failed: {e}")
        return DocumentAnalysis()


def classify_paragraph(text: str, surrounding_context: str = "") -> str:
    """Classify a single paragraph's semantic type.
    
    Args:
        text: The paragraph text
        surrounding_context: Text before/after for context
        
    Returns:
        Classification string: "heading", "body", "contact", "date", "bullet", etc.
    """
    model = _init_gemini()
    
    prompt = f"""Classify this paragraph from a resume/document.

PARAGRAPH: "{text}"
CONTEXT: "{surrounding_context[:200]}"

Respond with ONE word from: heading, body, contact, date, bullet, signature, address, title
Just the word, nothing else."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip().lower()
    except Exception:
        return "body"  # Safe default


def suggest_table_policy(table_description: str, complexity_score: int) -> str:
    """Use Gemini to decide if a table should be semantic or image fallback.
    
    Args:
        table_description: Description of table structure (rows, cols, merged cells)
        complexity_score: Pre-calculated complexity score
        
    Returns:
        "semantic" or "fallback"
    """
    if complexity_score > 20:
        return "fallback"  # Skip LLM call for clearly complex tables
    
    model = _init_gemini()
    
    prompt = f"""Should this table be rendered as semantic LaTeX or as an image?

TABLE: {table_description}
COMPLEXITY SCORE: {complexity_score}

Consider:
- Tables with merged cells, nested content, or >8 columns often need image fallback
- Simple data tables work well as semantic LaTeX

Respond with ONE word: semantic OR fallback"""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip().lower()
        return "fallback" if "fallback" in result else "semantic"
    except Exception:
        return "semantic" if complexity_score < 10 else "fallback"


def analyze_layout_with_vision(image_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Use Gemini Vision to detect layout elements (tables, figures).
    
    Args:
        image_path: Path to the image file (png/jpg) of the page.
        
    Returns:
        List of detected regions with normalized bounding boxes [0..1].
        Example: [{"label": "table", "bbox": [0.1, 0.2, 0.9, 0.5]}, ...]
    """
    model = _init_gemini()
    
    img = PIL.Image.open(image_path)
    
    prompt = """Analyze the layout of this document page.
Identify the bounding boxes for all:
1. Tables
2. Figures (charts, diagrams, images)
3. Main Semantic Headings (Level 1 only, e.g. "Introduction", "Methods")

Return a JSON object with a "regions" list.
Each region must have:
- "label": "table", "figure", or "heading"
- "bbox": [ymin, xmin, ymax, xmax] coordinates normalized to 0-1000 scale (integer).
  (Note: Gemini sometimes uses 0-1000, sometimes 0-1. Normalized 0-1000 is preferred standard for vision tasks).

Example:
{
  "regions": [
    {"label": "table", "bbox": [150, 100, 450, 900]},
    {"label": "heading", "bbox": [50, 100, 80, 500]}
  ]
}
"""
    
    try:
        response = model.generate_content([prompt, img])
        result_text = response.text.strip()
        
        # Clean markdown
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        
        data = json.loads(result_text)
        regions = data.get("regions", [])
        
        # Normalize to 0.0 - 1.0 float if in 0-1000 int
        normalized_regions = []
        for r in regions:
            bbox = r["bbox"]
            # Check if likely 1000 scale
            if any(c > 1.0 for c in bbox):
                bbox = [c / 1000.0 for c in bbox]
                
            # Gemini usually returns [ymin, xmin, ymax, xmax] for object detection
            # We want [x0, y0, x1, y1] (xmin, ymin, xmax, ymax)
            ymin, xmin, ymax, xmax = bbox
            
            normalized_regions.append({
                "label": r["label"],
                "bbox": [xmin, ymin, xmax, ymax],
                "confidence": 1.0  # Oracle assumption
            })
            
        return normalized_regions

    except Exception as e:
        print(f"Gemini Vision analysis failed: {e}")
        return []


if __name__ == "__main__":
    # Test the module
    test_text = """
    SUJENDRA JAYANT GHARAT
    Boston, MA | (857) 930-1933 | gharat.su@northeastern.edu
    
    EDUCATION
    Northeastern University, Boston, MA
    Master of Science in Information Systems
    
    EXPERIENCE
    Software Engineer Intern
    Developed FastAPI microservice with Redis lookups
    """
    
    print("Testing Gemini integration...")
    try:
        result = analyze_document_structure(test_text, "resume")
        print(f"Headings: {result.heading_suggestions}")
        print(f"Paragraph types: {result.paragraph_types}")
        print(f"Style recommendations: {result.style_recommendations}")
    except ValueError as e:
        print(f"API key not configured: {e}")
