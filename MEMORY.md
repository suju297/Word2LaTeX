# Project Memory (Word-to-LaTeX)

This file summarizes the key decisions, changes, and context so a new chat can resume work quickly.

## High-level goals
- Convert Word (.docx) to editable LaTeX while preserving layout as much as possible.
- Use ML layout detection (YOLO) + heuristics to improve layout hints.
- Use small local LLMs (SLMs) for lightweight routing and classification; keep Gemini optional.
- Provide a FastAPI server wrapper following best-practices layout.

## Pipeline overview (core library)
- Parse DOCX into IR (sections, blocks, spans, assets).
- Analyze layout from reference PDF with Layout Oracle (YOLO).
- Merge layout hints into IR.
- Apply calibration profile based on detected doc type.
- Optional header image fallback for resume-style headers (uses ref PDF crop).
- Policy engine decides semantic LaTeX vs image fallback for blocks.
- Generate LaTeX with template-free dynamic generator.

## Key model integrations
- Layout model: YOLOv11s V23 (mAP50 40.8%).
- Stored as `models/model_v5.pt` and loaded with highest priority.
- Layout inference uses `imgsz=1024` to match training.
- Labels include: Text, Title, List, Table, Figure, Caption, Footer, Header, Formula.
- Provenance tag updated to `ml_v5`.

## Local LLM routing (SLM)
- Purpose: classify `doc_type` and `layout_style` from extracted text.
- Default model: `models/llm/Qwen2.5-3B-Instruct-Q4_K_M.gguf`.
- Heuristic overrides:
  - If heuristic doc type is `resume`, override LLM doc_type to resume.
  - If layout grids detected, force `resume_grid`.
  - If section has 2+ columns, set `two_column`.
- Output stored in `doc.metadata`:
  - `slm_doc_type`, `slm_layout_style`, `slm_model_path`, `slm_raw_doc_type`, `slm_raw_layout_style`.

## Gemini usage
- Gemini is optional and disabled by default.
- If enabled, it can suggest headings and style hints.
- Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

## Header image fallback
- Resume headers with complex visuals are best handled via image fallback.
- Uses ref PDF crop (top ~20-25% or header region) to preserve branding.
- Designed as general fallback when ref PDF is available.

## Fixes and known limitations
- Images in DOCX inside `mc:AlternateContent` were not parsed; added handling.
- Visual styling (background shapes/colors) is not fully semantic; use image fallback.
- Font styling and layout fidelity may diverge for complex designs.

## Server repo structure (FastAPI)
- New repo created: `wordtolatex-server`.
- Structure follows FastAPI best-practices:
  - `src/wordtolatex/` core library
  - `src/server/` API layer
    - `conversion/` with `router.py`, `service.py`, `schemas.py`, `dependencies.py`
    - `main.py`, `config.py`, `exceptions.py`
- Endpoints:
  - `GET /health`
  - `POST /v1/convert` -> returns ZIP (assets + .tex)
  - `POST /v1/convert/json` -> returns LaTeX + metadata JSON

## Defaults and env flags
- Local LLM routing is default ON.
  - Disable: `WORDTOLATEX_USE_LOCAL_LLM=0` or `--no-local-llm`.
  - Override model: `WORDTOLATEX_LLM_MODEL=/path/to/model.gguf`.
- Gemini default OFF.
  - Enable: `WORDTOLATEX_USE_GEMINI=1` or `--gemini` in pipeline.
- Server config: `src/server/config.py`.

## Assets and repo hygiene
- Large models and datasets are excluded in `.gitignore`.
- Place YOLO weights under `models/` and GGUFs under `models/llm/` locally.

## Test highlights (manual)
- Resume_1, Resume_2, Microsoft_Resume1, Microsoft_Resume_2 routed as resume (grid/single).
- IEEE template routed as academic_twocol; ACM as academic_singlecol.
- Full pipeline test passed on Resume_1 with header image fallback (PDF produced).

## Notes on SLM selection
- Primary choice: Qwen2.5-3B-Instruct (Q4_K_M).
- Alternatives: Phi-4 Mini, Llama-3.2-3B Instruct.
- Use JSON-constrained decoding where possible for reliability.

## Files of interest
- Core pipeline: `src/wordtolatex/pipeline.py`
- Local LLM router: `src/wordtolatex/local_llm_router.py`
- Layout detector: `src/wordtolatex/layout_ml.py`
- Header fallback: `src/wordtolatex/header_fallback.py`
- Server entrypoint: `src/server/main.py`
