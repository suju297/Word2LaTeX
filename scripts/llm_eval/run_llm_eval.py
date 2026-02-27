"""Run a tiny LLM eval on local GGUF models using llama-cpp-python."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional


DOC_TYPE_CHOICES = ["resume", "academic", "generic"]
LAYOUT_STYLE_CHOICES = ["resume_grid", "two_column", "single_column"]

PROMPT_TEMPLATE = (
    "You are a strict JSON generator.\n"
    "Return JSON with keys doc_type and layout_style only.\n"
    f"doc_type must be one of {DOC_TYPE_CHOICES}.\n"
    f"layout_style must be one of {LAYOUT_STYLE_CHOICES}.\n"
    "Return only JSON, no extra text.\n"
    "\n"
    "Document excerpt:\n"
    "{text}\n"
    "\n"
    "JSON:\n"
)


def _extract_json(text: str) -> Optional[Dict]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def _normalize_label(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if not value:
        return None
    return value.replace("-", "_").replace(" ", "_")


def _load_cases(path: Path) -> List[Dict]:
    cases = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local LLM eval (llama-cpp-python).")
    parser.add_argument("--model", required=True, help="Path to GGUF model.")
    parser.add_argument("--cases", default="tests/llm_eval/cases.jsonl", help="JSONL eval cases.")
    parser.add_argument("--out", default="tests/llm_eval/results.jsonl", help="JSONL results output.")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads.")
    parser.add_argument("--ctx", type=int, default=2048, help="Context length.")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    args = parser.parse_args()

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise SystemExit(
            "llama-cpp-python is not installed. Install with "
            "`uv pip install llama-cpp-python` or use a local llama.cpp binary."
        ) from exc

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    cases = _load_cases(Path(args.cases))

    llm = Llama(
        model_path=str(model_path),
        n_ctx=args.ctx,
        n_threads=args.threads,
        n_gpu_layers=0,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    json_ok = 0
    doc_type_ok = 0
    layout_ok = 0
    strict_ok = 0

    with out_path.open("w", encoding="utf-8") as f:
        for case in cases:
            total += 1
            prompt = PROMPT_TEMPLATE.format(text=case.get("input_text", ""))
            start = time.time()
            resp = llm(
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stop=["\n\n"],
            )
            elapsed = time.time() - start
            raw = resp["choices"][0]["text"].strip()
            parsed = _extract_json(raw)
            ok = parsed is not None
            if ok:
                json_ok += 1
            expected = case.get("expected", {})
            pred_type = _normalize_label(parsed.get("doc_type")) if parsed else None
            pred_layout = _normalize_label(parsed.get("layout_style")) if parsed else None
            if pred_type == expected.get("doc_type"):
                doc_type_ok += 1
            if pred_layout == expected.get("layout_style"):
                layout_ok += 1
            if (
                pred_type == expected.get("doc_type")
                and pred_layout == expected.get("layout_style")
            ):
                strict_ok += 1

            record = {
                "id": case.get("id"),
                "expected": expected,
                "predicted": {
                    "doc_type": pred_type,
                    "layout_style": pred_layout,
                },
                "raw": raw,
                "parsed": parsed,
                "json_valid": ok,
                "doc_type_ok": pred_type == expected.get("doc_type"),
                "layout_style_ok": pred_layout == expected.get("layout_style"),
                "strict_ok": (
                    pred_type == expected.get("doc_type")
                    and pred_layout == expected.get("layout_style")
                ),
                "latency_s": round(elapsed, 3),
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = {
        "cases": total,
        "json_valid_rate": round(json_ok / total, 3) if total else 0.0,
        "doc_type_accuracy": round(doc_type_ok / total, 3) if total else 0.0,
        "layout_style_accuracy": round(layout_ok / total, 3) if total else 0.0,
        "strict_accuracy": round(strict_ok / total, 3) if total else 0.0,
        "results_path": str(out_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
