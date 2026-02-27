#!/usr/bin/env python3
"""Diagnose pagination issues by comparing ref vs generated PDFs.

Example:
  python tools/diagnose_pagination.py --ref examples/Sujendra_Jayant_Gharat_Resume.pdf \
    --tex examples/wordtolatex_output.tex --docx examples/Sujendra_Jayant_Gharat_Resume.docx \
    --out pagination_report.json
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from wordtolatex.docx_parser.document import parse_docx
from wordtolatex.utils import compile_latex


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _line_similarity(a: str, b: str) -> float:
    a_norm = _normalize_text(a).lower()
    b_norm = _normalize_text(b).lower()
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    return difflib.SequenceMatcher(None, a_norm, b_norm).ratio()


def _extract_page_lines(page: fitz.Page) -> List[Dict[str, Any]]:
    data = page.get_text("dict")
    lines: List[Dict[str, Any]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans)
            text = _normalize_text(text)
            if not text:
                continue
            bbox = line.get("bbox")
            if not bbox:
                continue
            lines.append({"text": text, "bbox": bbox})
    lines.sort(key=lambda item: (round(item["bbox"][1], 1), item["bbox"][0]))
    return lines


def _extract_page_images(page: fitz.Page) -> List[Dict[str, Any]]:
    data = page.get_text("dict")
    images: List[Dict[str, Any]] = []
    for block in data.get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = block.get("bbox")
        if bbox:
            images.append({"bbox": bbox})
    return images


def _normalize_bbox(page: fitz.Page, bbox: Tuple[float, float, float, float]) -> List[float]:
    rect = page.rect
    w = rect.width
    h = rect.height
    if w <= 0 or h <= 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]


def _get_block_text(block: Any) -> str:
    content = getattr(block, "content", None)
    if isinstance(content, list) and content:
        return "".join(getattr(span, "text", "") for span in content).strip()

    items = getattr(block, "items", None)
    if isinstance(items, list) and items:
        return " ".join(_get_block_text(item) for item in items).strip()

    children = getattr(block, "children", None)
    if isinstance(children, list) and children:
        return " ".join(_get_block_text(child) for child in children).strip()

    rows = getattr(block, "rows", None)
    if rows is not None:
        return "TABLE"

    blocks = getattr(block, "blocks", None)
    if isinstance(blocks, list) and blocks:
        return " ".join(_get_block_text(child) for child in blocks).strip()

    return ""


def _block_similarity(a: str, b: str) -> float:
    a_norm = _normalize_text(a).lower()
    b_norm = _normalize_text(b).lower()
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return 0.95
    a_words = set(a_norm.split())
    b_words = set(b_norm.split())
    if not a_words or not b_words:
        return 0.0
    return len(a_words & b_words) / len(a_words | b_words)


def _collect_blocks(doc) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for scope, blocks in (
        ("header", getattr(doc, "header_blocks", [])),
        ("body", getattr(doc, "blocks", [])),
        ("footer", getattr(doc, "footer_blocks", [])),
    ):
        for block in blocks:
            text = _normalize_text(_get_block_text(block))
            if not text:
                continue
            entries.append({
                "scope": scope,
                "block": block,
                "text": text,
            })
    return entries


def _find_best_block_match(line_text: str, blocks: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], float]:
    best = None
    best_score = 0.0
    for entry in blocks:
        score = _block_similarity(line_text, entry["text"])
        if score > best_score:
            best_score = score
            best = entry
    return best, best_score


def _find_best_ref_line(
    line_text: str,
    ref_lines_by_page: Dict[int, List[Dict[str, Any]]],
) -> Tuple[Optional[int], Optional[str], float]:
    best_page = None
    best_text = None
    best_score = 0.0
    for page_idx, lines in ref_lines_by_page.items():
        for line in lines:
            score = _line_similarity(line_text, line["text"])
            if score > best_score:
                best_score = score
                best_text = line["text"]
                best_page = page_idx
    return best_page, best_text, best_score


def _compile_if_needed(tex_path: Path, gen_pdf: Optional[Path]) -> Path:
    tex_path = tex_path.resolve()
    if gen_pdf is None:
        gen_pdf = tex_path.with_suffix(".pdf")
    gen_pdf = gen_pdf.resolve()

    if not compile_latex(tex_path, output_pdf=gen_pdf):
        raise RuntimeError(f"Failed to compile LaTeX: {tex_path}")
    if not gen_pdf.exists():
        raise RuntimeError(f"Compiled PDF not found: {gen_pdf}")
    return gen_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose pagination mismatches between PDFs.")
    parser.add_argument("--ref", type=Path, required=True, help="Reference PDF path")
    parser.add_argument("--gen", type=Path, help="Generated PDF path")
    parser.add_argument("--tex", type=Path, help="Generated LaTeX (compile to PDF)")
    parser.add_argument("--docx", type=Path, help="Original DOCX for block mapping")
    parser.add_argument("--out", type=Path, help="Write JSON report to this path")
    parser.add_argument("--max-lines", type=int, default=80, help="Max lines printed per extra page")
    parser.add_argument("--block-threshold", type=float, default=0.55, help="Min similarity to map line to block")
    args = parser.parse_args()

    ref_pdf = args.ref
    if not ref_pdf.exists():
        raise SystemExit(f"Reference PDF not found: {ref_pdf}")

    gen_pdf = args.gen
    if args.tex:
        gen_pdf = _compile_if_needed(args.tex, gen_pdf)
    if gen_pdf is None:
        raise SystemExit("Provide --gen or --tex to locate the generated PDF.")
    if not gen_pdf.exists():
        raise SystemExit(f"Generated PDF not found: {gen_pdf}")

    doc = None
    block_index: List[Dict[str, Any]] = []
    if args.docx:
        if not args.docx.exists():
            raise SystemExit(f"DOCX not found: {args.docx}")
        doc = parse_docx(args.docx)
        block_index = _collect_blocks(doc)

    report: Dict[str, Any] = {
        "ref_pdf": str(ref_pdf),
        "gen_pdf": str(gen_pdf),
        "ref_pages": 0,
        "gen_pages": 0,
        "extra_pages": [],
        "extra_page_lines": [],
        "extra_page_images": [],
    }

    with fitz.open(ref_pdf) as ref_doc, fitz.open(gen_pdf) as gen_doc:
        ref_pages = len(ref_doc)
        gen_pages = len(gen_doc)
        report["ref_pages"] = ref_pages
        report["gen_pages"] = gen_pages

        if gen_pages <= ref_pages:
            print(f"No extra pages. Ref={ref_pages}, Gen={gen_pages}")
            if args.out:
                args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return

        extra_pages = list(range(ref_pages, gen_pages))
        report["extra_pages"] = [p + 1 for p in extra_pages]
        print(f"Pagination mismatch. Ref={ref_pages}, Gen={gen_pages}")
        print(f"Extra pages: {[p + 1 for p in extra_pages]}")

        ref_lines_by_page: Dict[int, List[Dict[str, Any]]] = {}
        for i in range(ref_pages):
            ref_lines_by_page[i] = _extract_page_lines(ref_doc[i])

        for page_idx in extra_pages:
            page = gen_doc[page_idx]
            lines = _extract_page_lines(page)
            images = _extract_page_images(page)

            if images:
                for img in images:
                    report["extra_page_images"].append({
                        "page": page_idx + 1,
                        "bbox": img["bbox"],
                        "bbox_norm": _normalize_bbox(page, img["bbox"]),
                    })

            if not lines:
                print(f"\nPage {page_idx + 1}: No text lines found. Images={len(images)}")
                continue

            print(f"\nPage {page_idx + 1}: {len(lines)} line(s)")
            for idx, line in enumerate(lines):
                best_page, best_text, best_score = _find_best_ref_line(
                    line["text"], ref_lines_by_page
                )
                entry: Dict[str, Any] = {
                    "page": page_idx + 1,
                    "text": line["text"],
                    "bbox": line["bbox"],
                    "bbox_norm": _normalize_bbox(page, line["bbox"]),
                    "best_ref_page": (best_page + 1) if best_page is not None else None,
                    "best_ref_text": best_text,
                    "best_ref_similarity": best_score,
                }

                block_match = None
                block_score = 0.0
                if block_index:
                    block_match, block_score = _find_best_block_match(
                        line["text"], block_index
                    )
                    if block_match and block_score >= args.block_threshold:
                        block = block_match["block"]
                        entry["block_match"] = {
                            "id": getattr(block, "id", None),
                            "type": getattr(block, "block_type", None),
                            "scope": block_match["scope"],
                            "score": block_score,
                        }

                report["extra_page_lines"].append(entry)

                spillover = ""
                if best_page is not None and best_score >= 0.75:
                    spillover = f" (matches ref page {best_page + 1}, score {best_score:.2f})"
                block_info = ""
                if entry.get("block_match"):
                    block = entry["block_match"]
                    block_info = f" -> block {block['id']} ({block['type']}, {block['scope']})"

                if idx < args.max_lines:
                    print(f"  - y={entry['bbox_norm'][1]:.3f}-{entry['bbox_norm'][3]:.3f} {line['text']}{spillover}{block_info}")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport saved: {args.out}")


if __name__ == "__main__":
    main()
