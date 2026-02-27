"""Generate a small LLM eval set from the local corpus.

This creates a JSONL file with document excerpts and expected labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from wordtolatex.docx_parser import parse_docx
from wordtolatex.ir import LayoutGridBlock, TableBlock, TextboxBlock


DOC_LABELS: Dict[str, str] = {
    "tests/corpus/Resume_1/src.docx": "resume",
    "tests/corpus/Resume_2/src.docx": "resume",
    "tests/corpus/Custom_Resume/src.docx": "resume",
    "tests/corpus/Microsoft_Resume1/src.docx": "resume",
    "tests/corpus/Microsoft_Resume_2/src.docx": "resume",
    "tests/corpus/IEEE_TEMPLATE/src.docx": "academic",
    "tests/corpus/ACM_SUBMISSION_TEMPLATE/src.docx": "academic",
}


def _infer_layout_style(doc) -> str:
    if any(isinstance(block, LayoutGridBlock) for block in doc.blocks):
        return "resume_grid"
    if doc.sections and doc.sections[0].col_count > 1:
        return "two_column"
    return "single_column"


def _extract_text(doc, max_chars: int) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate small LLM eval cases.")
    parser.add_argument(
        "--output",
        default="tests/llm_eval/cases.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1400,
        help="Max characters per document excerpt.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    for doc_path, doc_type in DOC_LABELS.items():
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing docx: {path}")
        doc = parse_docx(path, path.parent)
        layout_style = _infer_layout_style(doc)
        excerpt = _extract_text(doc, args.max_chars)
        case_id = path.parent.name
        cases.append(
            {
                "id": case_id,
                "doc_path": doc_path,
                "input_text": excerpt,
                "expected": {
                    "doc_type": doc_type,
                    "layout_style": layout_style,
                },
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=True) + "\n")

    print(f"Wrote {len(cases)} cases to {output_path}")


if __name__ == "__main__":
    main()
