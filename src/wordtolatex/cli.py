"""Word to LaTeX Converter - CLI Entry Point."""

import os
import sys
from pathlib import Path
from typing import Optional

import click

from wordtolatex.docx_parser import parse_docx
from wordtolatex.generator import generate_latex
from wordtolatex.policy import decide_policy
from wordtolatex.report import generate_report, save_report
from wordtolatex.calibration import (
    detect_document_type, 
    get_profile, 
    apply_profile, 
    calibrate_from_pdf,
)
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


@click.command()
@click.argument("input_docx", type=click.Path(exists=True, path_type=Path))
@click.argument("output_tex", type=click.Path(path_type=Path))
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--report", "report_path", type=click.Path(path_type=Path), help="Generate report.json")
@click.option("--feedback", is_flag=True, help="Enable vision-guided feedback loop (requires DocTR)")
@click.option("--ref-pdf", type=click.Path(exists=True, path_type=Path), help="Reference PDF for feedback loop or calibration")
@click.option("--calibrate", is_flag=True, help="Auto-calibrate using reference PDF metrics (requires --ref-pdf)")
@click.option("--profile", "profile_name", 
              type=click.Choice(["auto", "academic_twocol", "academic_singlecol", "resume", "default"]),
              default="auto", help="Calibration profile (default: auto-detect)")
@click.option("--dynamic/--no-dynamic", default=True, help="Use dynamic template-free generator (default: enabled)")
@click.option("--header-fallback/--no-header-fallback", default=True, help="Enable header image fallback for resume headers (requires --ref-pdf)")
@click.option(
    "--local-llm/--no-local-llm",
    default=None,
    show_default=False,
    help="Use local GGUF router for doc_type/layout_style (default: enabled)",
)
@click.option("--llm-model", type=click.Path(exists=True, path_type=Path), help="Path to local GGUF model")
def cli(input_docx: Path, output_tex: Path, verbose: bool, report_path: Path = None,
        feedback: bool = False, ref_pdf: Path = None, calibrate: bool = False,
        profile_name: str = "auto", dynamic: bool = True, header_fallback: bool = True,
        local_llm: Optional[bool] = None, llm_model: Path = None):
    """Convert a Word document to LaTeX.

    INPUT_DOCX: Path to the input .docx file.
    OUTPUT_TEX: Path where the output .tex file will be written.
    """
    if verbose:
        click.echo(f"Parsing: {input_docx}")

    output_dir = output_tex.parent
    
    # Vision-Guided Feedback Loop Mode
    if feedback:
        if not ref_pdf:
            click.echo("Error: --ref-pdf is required when --feedback is enabled.", err=True)
            sys.exit(1)
            
        from wordtolatex.feedback_loop import FeedbackController
        
        if verbose:
            click.echo(f"Starting feedback loop with reference: {ref_pdf}")
            
        controller = FeedbackController(output_dir)
        success = controller.run_loop(input_docx, ref_pdf)
        
        if success:
            click.echo("✓ Feedback loop converged successfully.")
        else:
            click.echo("⚠ Feedback loop finished without reaching target match rate.")
            
        # Continue to standard reporting (optional, but output is already generated)
        click.echo(f"✓ Converted {input_docx.name} → {output_tex.name}")
        return

    # Standard One-Shot Conversion
    # Stage A: Parse DOCX
    try:
        doc = parse_docx(input_docx, output_dir)
    except Exception as e:
        click.echo(f"Error parsing DOCX: {e}", err=True)
        sys.exit(1)

    use_local_llm = _resolve_toggle(local_llm, "WORDTOLATEX_USE_LOCAL_LLM", default=True)
    if use_local_llm:
        route = maybe_route_document(
            doc,
            model_path=str(llm_model) if llm_model else None,
            enabled=True,
        )
        if verbose and route:
            click.echo(f"  [LLM] Routed: {route.doc_type} ({route.layout_style})")
        elif verbose:
            click.echo("  [LLM] Skipped (no model or invalid output)")

    if verbose:
        click.echo(f"  Found {len(doc.blocks)} blocks")
        if doc.sections:
            click.echo(f"  Page size: {doc.sections[0].page_width:.2f}x{doc.sections[0].page_height:.2f} inches")
        
        # Count block types
        block_types = {}
        for block in doc.blocks:
            bt = block.block_type
            block_types[bt] = block_types.get(bt, 0) + 1
        click.echo(f"  Block types: {block_types}")

    # Stage B: Apply Calibration Profile
    if calibrate and ref_pdf:
        # PDF-based calibration: measure reference and derive parameters
        profile = calibrate_from_pdf(ref_pdf, doc)
        if verbose:
            click.echo(f"  [Calibration] PDF-based: font={profile.font_size:.1f}pt")
    elif profile_name == "auto":
        # Auto-detect document type and apply appropriate profile
        doc_type = detect_document_type(doc)
        profile = get_profile(doc_type)
        if verbose:
            click.echo(f"  [Calibration] Auto-detected: {doc_type}")
    else:
        # Manual profile selection
        profile = get_profile(profile_name)
        if verbose:
            click.echo(f"  [Calibration] Manual profile: {profile_name}")
    
    apply_profile(doc, profile)
    if verbose:
        click.echo(f"  [Calibration] Applied '{profile.name}' (font={profile.font_size}pt, line={profile.line_spacing})")

    # Header image fallback (resume branding) if reference PDF is available
    if ref_pdf and header_fallback:
        applied = apply_header_image_fallback(doc, ref_pdf, output_dir)
        if verbose and applied:
            click.echo("  [Header] Applied header image fallback")

    # Stage D: Apply policy decisions
    for block in doc.blocks:
        block.policy = decide_policy(block)

    # Stage E: Generate LaTeX
    if verbose:
        click.echo("Generating LaTeX...")

    try:
        if dynamic:
            # Use template-free dynamic generator
            from wordtolatex.dynamic_generator import generate_latex as dynamic_generate
            latex_content = dynamic_generate(doc, ref_pdf=ref_pdf)
            if verbose:
                click.echo("  [Dynamic] Using template-free generator")
        else:
            # Use template-based generator
            latex_content = generate_latex(doc, squeeze_factor=profile.squeeze_factor)
    except Exception as e:
        click.echo(f"Error generating LaTeX: {e}", err=True)
        sys.exit(1)

    # Write output
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text(latex_content, encoding="utf-8")

    if verbose:
        click.echo(f"Written to: {output_tex}")

    # Generate report if requested
    if report_path:
        report = generate_report(doc, input_docx, output_tex)
        save_report(report, report_path)
        if verbose:
            click.echo(f"Report saved to: {report_path}")
            click.echo(f"  Headings: {report.headings}, Paragraphs: {report.paragraphs}")
            click.echo(f"  Lists: {report.lists}, Tables: {report.tables}")
            click.echo(f"  Bold spans: {report.bold_spans}, Italic spans: {report.italic_spans}")

    click.echo(f"✓ Converted {input_docx.name} → {output_tex.name}")


if __name__ == "__main__":
    cli()
