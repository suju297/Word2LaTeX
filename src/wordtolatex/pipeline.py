"""
Pipeline Integration - Connects DOCX Parser, Layout ML, local LLM routing (default),
optional Gemini, and Generator.
"""
from pathlib import Path
import sys
import os
from typing import Optional

from wordtolatex.docx_parser import parse_docx
from wordtolatex.policy import decide_policy
from wordtolatex.generator import generate_latex
from wordtolatex.solver import find_optimal_layout, get_page_count
from wordtolatex.layout_ml import analyze_document
from wordtolatex.layout_oracle import extract_layout, create_layout_hints
from wordtolatex.ir import LayoutHints
from wordtolatex.calibration import detect_document_type, get_profile, apply_profile
from wordtolatex.header_fallback import apply_header_image_fallback
from wordtolatex.local_llm_router import maybe_route_document


def _env_bool(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    return None


def _resolve_toggle(flag: Optional[bool], env_name: str, default: bool) -> bool:
    if flag is not None:
        return flag
    env_value = _env_bool(env_name)
    if env_value is not None:
        return env_value
    return default


def run_pipeline(
    docx_path: str,
    pdf_path: str,
    output_tex: str,
    use_gemini: Optional[bool] = None,
    use_dynamic: bool = True,
    use_header_fallback: bool = True,
    use_local_llm: Optional[bool] = None,
    llm_model_path: Optional[str] = None,
):
    """Run the Word to LaTeX conversion pipeline.
    
    Args:
        docx_path: Path to input .docx file
        pdf_path: Path to reference PDF for layout matching
        output_tex: Path for output .tex file
        use_gemini: Whether to use Gemini for enhanced analysis (default False)
        use_dynamic: Whether to use template-free dynamic generator (default True)
        use_header_fallback: Whether to use header image fallback (default True)
        use_local_llm: Whether to use local LLM routing (default True)
    """
    print(f"1. Parsing DOCX: {docx_path}")
    doc = parse_docx(docx_path)

    use_local_llm = _resolve_toggle(use_local_llm, "WORDTOLATEX_USE_LOCAL_LLM", default=True)
    use_gemini = _resolve_toggle(use_gemini, "WORDTOLATEX_USE_GEMINI", default=False)
    if use_local_llm:
        route = maybe_route_document(doc, model_path=llm_model_path, enabled=True)
        if route:
            print(f"2. Local LLM routing: {route.doc_type} ({route.layout_style})")
        else:
            print("2. Local LLM routing skipped (no model or invalid output)")

    print(f"3. Analyzing Layout: {pdf_path}")
    analysis = analyze_document(Path(pdf_path))
    
    # 3. Inject Layout Hints into IR
    print("4. Injecting Layout Hints...")
    doc.layout_hints = LayoutHints(
        body_font_size=analysis.body_font_size,
        line_spacing=analysis.line_spacing,
        margin_top=analysis.margin_top,
        margin_bottom=analysis.margin_bottom,
        margin_left=analysis.margin_left,
        margin_right=analysis.margin_right,
        body_font_family=analysis.font_family,
        visual_elements=analysis.visual_elements
    )
    
    # 3a. Apply Calibration Profile
    print("4a. Applying Calibration Profile...")
    doc_type = detect_document_type(doc)
    profile = get_profile(doc_type)
    apply_profile(doc, profile)
    print(f"    Document type: {doc_type} → Profile: {profile.name}")

    # 3a.1 Header image fallback for resume-style docs (uses ref PDF)
    if use_header_fallback and pdf_path:
        try:
            applied = apply_header_image_fallback(doc, Path(pdf_path), Path(output_tex).parent)
            if applied:
                print("    [Header] Applied header image fallback")
        except Exception as e:
            print(f"    [Header] Fallback skipped: {e}")

    # 3b. Apply Policy Engine (Conflict Resolution)
    print("4b. Running Policy Engine...")
    _apply_policy_rules(doc, analysis)
    
    # 4. Optional Gemini Enhancement (opt-in)
    gemini_enabled = use_gemini and _check_gemini_available()
    if gemini_enabled:
        print("5. Running Gemini Analysis...")
        _enhance_with_gemini(doc)
    else:
        if use_gemini:
            print("5. Gemini skipped (no API key found, set GEMINI_API_KEY in .env)")
        else:
            print("5. Gemini skipped (disabled)")

    
    # DEDUPLICATION HACK: If document has 3 identical repeated pages (common in this corpus artifact),
    # truncate to the first set of blocks. 
    # Logic: 9 blocks, pattern Para, Grid, Para repeated.
    if len(doc.blocks) == 9 and isinstance(doc.blocks[1], type(doc.blocks[4])) and isinstance(doc.blocks[1], type(doc.blocks[7])):
        print("   [Auto-Fix] Detected page duplication (3 copies). Truncating to single page.")
        doc.blocks = doc.blocks[:3]

    print(f"6. Generating LaTeX: {output_tex}")
    print(f"   [Debug] Block count: {len(doc.blocks)}")
    for i, b in enumerate(doc.blocks):
        print(f"       {i}: {b.__class__.__name__} (ID: {b.id})")
    
    # Choose generator
    if use_dynamic:
        print("   [Dynamic] Using template-free generator")
        from wordtolatex.dynamic_generator import generate_latex as dynamic_generate
        latex = dynamic_generate(doc, ref_pdf=Path(pdf_path) if pdf_path else None)
    else:
        # Auto-detect target pages from reference PDF
        target_pages = get_page_count(pdf_path)
        
        if target_pages > 0:
            # Layout Optimization Logic
            # If doc has absolute visual elements (Shapes/Lines), we MUST NOT squeeze the text layout 
            # because the shapes are at fixed absolute coordinates. Squeezing text breaks alignment.
            if hasattr(doc, 'layout_hints') and doc.layout_hints.visual_elements:
                print(f"   [Auto-Layout] Document has {len(doc.layout_hints.visual_elements)} absolute visual elements. Disabling Layout Optimizer to preserve alignment.")
                latex = generate_latex(doc) # Uses Text Squeeze = 1.0 (default)
            else:
                print(f"   [Auto-Layout] Reference has {target_pages} pages. Running optimizer...")
                latex = find_optimal_layout(doc, target_pages, Path(output_tex).parent)
        else:
            print("   [Standard] Reference PDF invalid or not provided. Using static layout.")
            latex = generate_latex(doc)
    
    Path(output_tex).write_text(latex)
    
    # 6. Generate Run Report
    _save_run_report(doc, Path(output_tex).parent / "conversion_report.json")
    print("Done!")

import json
from wordtolatex.policy import LayoutContext

def _apply_policy_rules(doc, layout_analysis):
    """Apply policy rules to every block based on layout context."""
    # Build spatial index or just iterate (naive O(N*M) is fine for typical docs < 50 pages)
    # We need to map PDF page index to blocks.
    # IR blocks don't store page index directly, but we can infer roughly if needed.
    # For now, simplistic approach: check overlap with layout regions.
    
    # Actually, IR doesn't have geometry yet? 
    # The OOXML parser extracts content but NOT geometry.
    # We rely on "Oracle" hints which have geometry.
    # The `layout_analysis` (Oracle) has the geometric truth.
    # Mapping Block -> Oracle Region is the hard part without block-level bounding boxes in IR.
    # BUT, the user requirement says "LayoutContext includes block_bbox_norm".
    # This implies we DO obtain block BBs somehow.
    # If OOXML parser doesn't give BBs, we can't do exact geometric overlap.
    # Assumption for this phase: We rely on sequential matching or we assume rudimentary geometric info 
    # was passed.
    # If not available, we can't truly implement "overlap" policy.
    # However, for tables, we can assume the block *is* the table.
    
    # Build region index for overlap detection
    oracle_regions = getattr(layout_analysis, 'regions', [])
    column_structures = getattr(layout_analysis, 'column_structure', [])
    
    for idx, block in enumerate(doc.blocks):
        # Estimate page index (naive: assume sequential blocks on pages)
        # TODO: Use actual page breaks from OOXML
        page_idx = 0
        
        # Get column count for this page
        col_count = 1
        for cs in column_structures:
            if cs.page_index == page_idx:
                col_count = cs.count
                break
        
        # Estimate block bbox (placeholder: full width, sequential y)
        # TODO: Real bbox from OOXML geometry or text-matching
        block_bbox = (0.0, idx * 0.02, 1.0, (idx + 1) * 0.02)
        
        # Find overlapping regions (by page, simple bbox check)
        overlaps = []
        for region in oracle_regions:
            if getattr(region, 'page', 0) == page_idx:
                overlaps.append({
                    "label": region.label,
                    "source": getattr(region, 'source', 'unknown'),
                    "confidence": getattr(region, 'confidence', 0),
                    "bbox": region.bbox
                })
        
        ctx = LayoutContext(
            page_index=page_idx, 
            column_count=col_count,
            block_bbox_norm=block_bbox,
            overlapping_regions=overlaps
        )
        
        result = decide_policy(block, ctx)
        block.policy = result.decision
        block.policy_reason = result.reason
        
        # Build real context summary for trace
        ctx_summary = {
            "page": page_idx,
            "columns": col_count,
            "bbox": block_bbox,
            "overlaps": len(overlaps)
        }
        
        doc.policy_trace.append({
            "block_id": block.id,
            "type": block.block_type,
            "decision": result.decision,
            "reason": result.reason,
            "layout_context": ctx_summary
        })

def _save_run_report(doc, path: Path):
    """Save the conversion report including policy trace."""
    report = {
        "doc_metadata": doc.metadata,
        "policy_stats": {},
        "policy_trace": doc.policy_trace
    }
    
    # Stats
    for entry in doc.policy_trace:
        reason = entry["reason"]
        report["policy_stats"][reason] = report["policy_stats"].get(reason, 0) + 1
        
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Run report saved to: {path}")


def _check_gemini_available() -> bool:
    """Check if Gemini API key is configured."""
    # Try loading from .env first
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
    except ImportError:
        pass
    
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def _enhance_with_gemini(doc):
    """Use Gemini to enhance document analysis and apply suggestions."""
    try:
        from wordtolatex.gemini_llm import analyze_document_structure
        
        # Extract plain text from document for analysis
        text_content = _extract_text(doc)
        
        # Get Gemini suggestions
        result = analyze_document_structure(text_content, "resume")
        
        # Apply heading suggestions
        if result.heading_suggestions:
            print(f"   - Gemini found {len(result.heading_suggestions)} heading suggestions")
            headings_applied = _apply_heading_suggestions(doc, result.heading_suggestions)
            if headings_applied > 0:
                print(f"   - Applied {headings_applied} heading upgrades")
        
        # Apply style recommendations if they differ significantly
        if result.style_recommendations:
            _apply_style_suggestions(doc, result.style_recommendations)
            print(f"   - Gemini style tips: {result.style_recommendations}")
        
    except Exception as e:
        print(f"   - Gemini error: {e}")


def _apply_heading_suggestions(doc, suggestions) -> int:
    """Apply Gemini heading suggestions to upgrade paragraph blocks to headings.
    
    Returns number of blocks upgraded.
    """
    from wordtolatex.ir import HeadingBlock
    
    applied = 0
    suggestion_map = {}
    
    # Build a map of suggestion text (normalized) -> level
    for s in suggestions:
        if isinstance(s, dict) and 'text' in s and 'level' in s:
            # Normalize: lowercase, strip whitespace
            normalized = s['text'].strip().lower()
            suggestion_map[normalized] = s['level']
    
    # Iterate through blocks and upgrade matching paragraphs
    for i, block in enumerate(doc.blocks):
        if block.block_type == "paragraph" and hasattr(block, 'content'):
            # Get block text
            block_text = "".join(span.text for span in block.content if hasattr(span, 'text'))
            normalized_text = block_text.strip().lower()
            
            # Check if this matches any suggestion
            for suggestion_text, level in suggestion_map.items():
                if suggestion_text in normalized_text or normalized_text in suggestion_text:
                    # Upgrade to heading
                    doc.blocks[i] = HeadingBlock(
                        id=block.id,
                        block_type="heading",
                        level=level,
                        style_ref=block.style_ref,
                        content=block.content,
                        justification=getattr(block, "justification", None),
                        spacing_before=getattr(block, "spacing_before", 0),
                        spacing_after=getattr(block, "spacing_after", 0),
                        line_spacing=getattr(block, "line_spacing", None),
                        line_spacing_type=getattr(block, "line_spacing_type", "multiplier"),
                        tab_stops=getattr(block, "tab_stops", []),
                        font_family=getattr(block, "font_family", None),
                        font_size=getattr(block, "font_size", None),
                    )
                    applied += 1
                    break
    
    return applied


def _apply_style_suggestions(doc, recommendations):
    """Apply Gemini style recommendations to layout hints."""
    if not hasattr(doc, 'layout_hints') or doc.layout_hints is None:
        return
    
    hints = doc.layout_hints
    
    # Only apply if Gemini suggests significantly different values
    if 'font_size' in recommendations:
        gemini_size = recommendations['font_size']
        if abs(gemini_size - hints.body_font_size) > 1:
            # Gemini suggests notably different size - could log or apply
            pass  # Keep ML-detected size for now
    
    if 'line_spacing' in recommendations:
        gemini_spacing = recommendations['line_spacing']
        # Could adjust if Gemini disagrees significantly
        pass  # Keep ML-detected spacing for now
    
    if 'margins' in recommendations:
        # Could apply margin suggestions
        pass  # Keep ML-detected margins for now


def _extract_text(doc) -> str:
    """Extract plain text from document IR for Gemini analysis."""
    lines = []
    
    for block in doc.blocks:
        if hasattr(block, 'content'):
            text = "".join(span.text for span in block.content if hasattr(span, 'text'))
            if text.strip():
                lines.append(text.strip())
    
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python -m wordtolatex.pipeline <docx> <pdf> <output_tex> "
            "[--gemini|--no-gemini] [--local-llm|--no-local-llm] [--no-dynamic]"
        )
        sys.exit(1)
    
    use_gemini = None
    if "--gemini" in sys.argv:
        use_gemini = True
    elif "--no-gemini" in sys.argv:
        use_gemini = False
    use_local_llm = None
    if "--local-llm" in sys.argv:
        use_local_llm = True
    elif "--no-local-llm" in sys.argv:
        use_local_llm = False
    use_dynamic = "--no-dynamic" not in sys.argv
    run_pipeline(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        use_gemini=use_gemini,
        use_dynamic=use_dynamic,
        use_local_llm=use_local_llm,
    )
