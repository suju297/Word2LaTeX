"""Conversion service orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Optional

from server.config import settings
from server.conversion.constants import DEFAULT_TEX_NAME, DEFAULT_ZIP_NAME
from server.conversion.schemas import ConversionOptions
from server.conversion.utils import generate_reference_pdf, write_bytes, zip_directory

from wordtolatex.calibration import (
    apply_profile,
    calibrate_from_pdf,
    detect_document_type,
    get_profile,
)
from wordtolatex.docx_parser import parse_docx
from wordtolatex.generator import generate_latex
from wordtolatex.header_fallback import apply_header_image_fallback
from wordtolatex.ir import LayoutHints
from wordtolatex.layout_ml import analyze_document
from wordtolatex.local_llm_router import maybe_route_document
from wordtolatex.policy import decide_policy


@dataclass
class ConversionResult:
    latex: str
    doc_type: Optional[str]
    layout_style: Optional[str]
    metadata: dict
    has_assets: bool
    tex_path: Path
    zip_path: Path
    work_dir: Path


def _resolve_local_llm(options: ConversionOptions) -> bool:
    if options.local_llm is not None:
        return options.local_llm
    return settings.default_use_local_llm


def _resolve_gemini(options: ConversionOptions) -> bool:
    if options.use_gemini is not None:
        return options.use_gemini
    return settings.default_use_gemini


def _apply_gemini(doc) -> None:
    try:
        from wordtolatex.pipeline import _check_gemini_available, _enhance_with_gemini
    except Exception:
        return

    if _check_gemini_available():
        _enhance_with_gemini(doc)


def convert_document(
    docx_bytes: bytes,
    ref_pdf_bytes: Optional[bytes],
    options: ConversionOptions,
) -> ConversionResult:
    work_dir = Path(tempfile.mkdtemp(prefix=settings.temp_dir_prefix))
    input_docx = work_dir / "input.docx"
    write_bytes(input_docx, docx_bytes)

    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = parse_docx(input_docx, output_dir)

    if _resolve_local_llm(options):
        maybe_route_document(
            doc,
            model_path=options.llm_model or settings.default_llm_model,
            enabled=True,
        )

    ref_pdf_path = None
    if ref_pdf_bytes:
        ref_pdf_path = work_dir / "ref.pdf"
        write_bytes(ref_pdf_path, ref_pdf_bytes)
    else:
        ref_pdf_path = generate_reference_pdf(input_docx, work_dir)

    if ref_pdf_path:
        analysis = analyze_document(ref_pdf_path)
        doc.layout_hints = LayoutHints(
            body_font_size=analysis.body_font_size,
            line_spacing=analysis.line_spacing,
            margin_top=analysis.margin_top,
            margin_bottom=analysis.margin_bottom,
            margin_left=analysis.margin_left,
            margin_right=analysis.margin_right,
            body_font_family=analysis.font_family,
            visual_elements=analysis.visual_elements,
        )

    if options.calibrate and ref_pdf_path:
        profile = calibrate_from_pdf(ref_pdf_path, doc)
    elif options.profile == "auto":
        doc_type = detect_document_type(doc)
        profile = get_profile(doc_type)
    else:
        profile = get_profile(options.profile)

    apply_profile(doc, profile)

    if options.header_fallback and ref_pdf_path:
        try:
            apply_header_image_fallback(doc, ref_pdf_path, output_dir)
        except Exception:
            pass

    for block in doc.blocks:
        block.policy = decide_policy(block)

    if _resolve_gemini(options):
        _apply_gemini(doc)

    if options.dynamic:
        from wordtolatex.dynamic_generator import generate_latex as dynamic_generate
        latex = dynamic_generate(doc, ref_pdf=ref_pdf_path)
    else:
        latex = generate_latex(doc, squeeze_factor=profile.squeeze_factor)

    output_tex = output_dir / DEFAULT_TEX_NAME
    output_tex.write_text(latex, encoding="utf-8")

    assets_dir = output_dir / "media"
    has_assets = assets_dir.exists() and any(p.is_file() for p in assets_dir.rglob("*"))

    zip_path = work_dir / DEFAULT_ZIP_NAME
    zip_directory(output_dir, zip_path)

    doc_type = doc.metadata.get("slm_doc_type") or detect_document_type(doc)
    layout_style = doc.metadata.get("slm_layout_style")

    return ConversionResult(
        latex=latex,
        doc_type=doc_type,
        layout_style=layout_style,
        metadata=dict(doc.metadata),
        has_assets=has_assets,
        tex_path=output_tex,
        zip_path=zip_path,
        work_dir=work_dir,
    )
