"""Local LLM router for lightweight document classification.

This module uses a local GGUF model (llama-cpp-python) to classify:
1) doc_type: resume | academic | generic
2) layout_style: resume_grid | two_column | single_column
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from wordtolatex.ir import Document, LayoutGridBlock, TableBlock, TextboxBlock
from wordtolatex.calibration import detect_document_type


DOC_TYPE_CHOICES = {"resume", "academic", "generic"}
LAYOUT_STYLE_CHOICES = {"resume_grid", "two_column", "single_column"}

PROMPT_TEMPLATE = (
    "You are a strict JSON generator.\n"
    "Return JSON with keys doc_type and layout_style only.\n"
    f"doc_type must be one of {sorted(DOC_TYPE_CHOICES)}.\n"
    f"layout_style must be one of {sorted(LAYOUT_STYLE_CHOICES)}.\n"
    "Return only JSON, no extra text.\n"
    "\n"
    "Document excerpt:\n"
    "{text}\n"
    "\n"
    "JSON:\n"
)


@dataclass
class LLMRouteResult:
    """Normalized routing result from a local LLM."""

    doc_type: str
    layout_style: str
    raw: Dict[str, str]
    model_path: str


_LLM = None
_LLM_PATH: Optional[Path] = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _resolve_model_path(model_path: Optional[str]) -> Optional[Path]:
    if model_path:
        path = Path(model_path).expanduser()
        return path if path.exists() else None

    env_path = os.environ.get("WORDTOLATEX_LLM_MODEL")
    if env_path:
        path = Path(env_path).expanduser()
        return path if path.exists() else None

    repo_root = Path(__file__).resolve().parents[2]
    default_path = repo_root / "models" / "llm" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    return default_path if default_path.exists() else None


def _get_llm(model_path: Path, threads: int, ctx: int):
    global _LLM, _LLM_PATH
    if _LLM is None or _LLM_PATH != model_path:
        from llama_cpp import Llama
        verbose = _env_flag("WORDTOLATEX_LLM_VERBOSE", default=False)
        _LLM = Llama(
            model_path=str(model_path),
            n_ctx=ctx,
            n_threads=threads,
            n_gpu_layers=0,
            verbose=verbose,
        )
        _LLM_PATH = model_path
    return _LLM


def _extract_json(text: str) -> Optional[Dict]:
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start == -1:
            return None
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            idx = start + 1


def _normalize_label(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if not value:
        return None
    return value.replace("-", "_").replace(" ", "_")


def _resolve_doc_type(doc_type: str, layout_style: str) -> str:
    if doc_type == "academic":
        return "academic_twocol" if layout_style == "two_column" else "academic_singlecol"
    if doc_type == "resume":
        return "resume"
    return "default"


def _infer_layout_style(doc: Document, doc_type: str) -> str:
    layout_grids = sum(1 for block in doc.blocks if isinstance(block, LayoutGridBlock))
    if doc_type == "resume" and layout_grids >= 1:
        return "resume_grid"

    if doc.sections and doc.sections[0].col_count > 1:
        return "two_column"

    return "single_column"


def _extract_text(doc: Document, max_chars: int) -> str:
    lines: List[str] = []
    total = 0

    def push_text(text: str) -> None:
        nonlocal total
        text = " ".join(text.split())
        if not text:
            return
        lines.append(text)
        total += len(text) + 1

    def walk_blocks(blocks) -> None:
        nonlocal total
        for block in blocks:
            if total >= max_chars:
                return
            if isinstance(block, LayoutGridBlock):
                for row in block.rows:
                    for cell in row.cells:
                        walk_blocks(cell.blocks)
                continue
            if isinstance(block, TextboxBlock):
                walk_blocks(block.content_blocks)
                continue
            if isinstance(block, TableBlock):
                for row in block.rows:
                    for cell in row.cells:
                        text = "".join(span.text for span in cell.content if hasattr(span, "text"))
                        push_text(text)
                continue
            content = getattr(block, "content", None)
            if not content:
                continue
            text = "".join(span.text for span in content if hasattr(span, "text"))
            push_text(text)

    walk_blocks(doc.blocks)
    return "\n".join(lines)[:max_chars]


def classify_document(
    doc: Document,
    model_path: Optional[str] = None,
    max_chars: int = 1400,
    threads: int = 4,
    ctx: int = 2048,
    temperature: float = 0.0,
) -> Optional[LLMRouteResult]:
    path = _resolve_model_path(model_path)
    if not path:
        return None

    try:
        llm = _get_llm(path, threads=threads, ctx=ctx)
    except Exception:
        return None

    prompt = PROMPT_TEMPLATE.format(text=_extract_text(doc, max_chars))
    resp = llm(
        prompt,
        max_tokens=128,
        temperature=temperature,
        stop=["\n\n"],
    )
    raw = resp["choices"][0]["text"].strip()
    parsed = _extract_json(raw)
    if not parsed:
        return None

    doc_type = _normalize_label(parsed.get("doc_type"))
    layout_style = _normalize_label(parsed.get("layout_style"))
    if doc_type not in DOC_TYPE_CHOICES or layout_style not in LAYOUT_STYLE_CHOICES:
        return None

    llm_doc_type = doc_type
    llm_layout_style = layout_style
    heuristic_doc_type = detect_document_type(doc)
    if heuristic_doc_type == "resume" and doc_type != "resume":
        doc_type = "resume"

    heuristic_layout = _infer_layout_style(doc, doc_type)
    layout_style = heuristic_layout or layout_style
    internal_doc_type = _resolve_doc_type(doc_type, layout_style)
    return LLMRouteResult(
        doc_type=internal_doc_type,
        layout_style=layout_style,
        raw={
            "doc_type": llm_doc_type,
            "layout_style": llm_layout_style,
        },
        model_path=str(path),
    )


def maybe_route_document(
    doc: Document,
    model_path: Optional[str] = None,
    enabled: Optional[bool] = None,
) -> Optional[LLMRouteResult]:
    if enabled is None:
        enabled = _env_flag("WORDTOLATEX_USE_LOCAL_LLM", default=False)
    if not enabled:
        return None

    result = classify_document(doc, model_path=model_path)
    if not result:
        return None

    doc.metadata["slm_doc_type"] = result.doc_type
    doc.metadata["slm_layout_style"] = result.layout_style
    doc.metadata["slm_model_path"] = result.model_path
    doc.metadata["slm_raw_doc_type"] = result.raw.get("doc_type")
    doc.metadata["slm_raw_layout_style"] = result.raw.get("layout_style")

    if result.layout_style == "two_column" and doc.sections:
        if doc.sections[0].col_count < 2:
            doc.sections[0].col_count = 2
            if doc.sections[0].col_spacing < 1:
                doc.sections[0].col_spacing = 24.0
        doc.layout_hints.suggested_columns = 2

    return result
