# Word → Editable LaTeX Converter

## Executive summary

This document describes a system that converts **Microsoft Word documents (.docx)** into **editable, semantic LaTeX (.tex)** while preserving **structure** (headings, paragraphs, lists, tables, figures, headers/footers) and **styling** (fonts, sizes, spacing, margins, numbering) as much as possible.

The key design choice is to treat **DOCX (OOXML)** as the **source of truth** for text and semantics, and to optionally use a **rendered PDF + layout understanding (ML/OCR geometry)** as a **layout oracle** to improve handling of complex layouts (columns, textboxes, difficult tables) without turning the output into fragile, absolute-positioned LaTeX.

This is **not** a universal “pixel-perfect” converter. The goal is  **maintainable LaTeX** . When a construct cannot be represented cleanly (e.g., SmartArt), the system uses **policy-driven fallbacks** (typically embedding a rendered image) and reports those decisions.

---

## Table of contents

1. Background and motivation
2. Goals, non-goals, and success criteria
3. Key concepts and terminology
4. Why Word → LaTeX is hard
5. High-level architecture
6. End-to-end pipeline
7. Intermediate Representation (IR)
8. Style mapping strategy
9. Handling hard cases (policies)
10. Optional layout oracle (PDF + ML/OCR geometry)
11. Validation, quality report, and evaluation metrics
12. MVP scope and roadmap
13. Implementation notes (tech stack options)
14. Risks and mitigations
15. Appendix: IR examples and scoring heuristics

---

## 1) Background and motivation

Many teams have years of content in Word, but need LaTeX for:

* publishing (academic templates, journals)
* consistent typography
* programmatic authoring
* version control and diffs

Existing converters often:

* preserve structure but lose styling detail
* generate LaTeX that’s hard to edit
* fail on Word-specific constructs (text boxes, SmartArt, complex tables)

We want a converter that is:

* **practical** for real documents
* **predictable** (clear fallbacks)
* **measurable** (quality metrics)
* **maintainable** (both code and output)

---

## 2) Goals, non-goals, and success criteria

### Goals

* Produce  **editable LaTeX** :
  * headings → `\section`, `\subsection`, etc.
  * lists → `itemize` / `enumerate`
  * tables → `tabularx` / `longtable` when feasible
  * figures → `figure` + `\includegraphics`
* Preserve styling in a controlled way:
  * page size, margins, orientation
  * paragraph spacing and line spacing
  * list indentation and labels
  * headers/footers
  * fonts when possible via XeLaTeX/LuaLaTeX
* Provide a  **conversion report** :
  * compile success/failure
  * coverage (semantic vs fallback)
  * warnings (overfull boxes, missing assets/fonts)

### Non-goals (v1)

* Guaranteed identical pagination/line breaks/kerning compared to Word
* Full editable conversion of SmartArt, complex drawings, or tight-wrap floating shapes
* “No manual cleanup ever” guarantee for arbitrary documents

### Success criteria

* ≥ 90% of a representative corpus compiles to PDF without manual edits
* High semantic coverage for common content:
  * headings/paragraphs/lists/figures: near 100%
  * tables: high for simple/medium, graceful fallback for complex
* Clear deterministic fallbacks + useful report

---

## 3) Key concepts and terminology

### DOCX / OOXML

A `.docx` file is a ZIP archive of XML parts (Office Open XML). Important parts:

* `word/document.xml` — main body content
* `word/styles.xml` — style definitions + inheritance
* `word/numbering.xml` — bullets/numbering definitions
* `word/header*.xml`, `word/footer*.xml` — headers/footers
* `word/footnotes.xml`, `word/endnotes.xml`
* section properties (`w:sectPr`) — page size, margins, columns, etc.

### Semantic LaTeX

LaTeX that represents meaning (sections, lists, tables) rather than absolute x/y placement. It remains editable and reflows when content changes.

### Layout oracle

A rendered view (PDF/page images) used to infer layout hints (columns, header/footer repetition, region classification). It is not the canonical source for text.

### Policy engine

A deterministic module that chooses how to represent difficult constructs:

* semantic LaTeX when feasible
* fallback (usually image embed) when not

---

## 4) Why Word → LaTeX is hard

Word and TeX are different layout engines.

Word computes many details at render-time:

* line breaks and page breaks
* justification spacing behavior
* anchoring/wrapping for floating objects

DOCX stores styles and structure, but generally does **not** store:

* exact final line breaks
* exact final page breaks (unless explicit)
* per-word x/y coordinates
* kerning/glyph-level placement

If we force per-line breaks or absolute positioning to match Word, LaTeX becomes fragile and not truly editable. This project prioritizes semantic output.

---

## 4.5) Matching Word PDF: what “same result” means

You can absolutely require that **(Word → PDF)** and **(LaTeX → PDF)** look identical, but you must choose which of these you care about:

* **Pixel-perfect (Replica mode):** the PDFs match extremely closely (even page breaks and line breaks), but the LaTeX source is mostly **positioned layout** and is  **not pleasant to edit** .
* **Editable (Semantic mode):** the LaTeX is clean and maintainable, but the PDF will be **close** rather than guaranteed pixel-identical for arbitrary Word documents.

This project supports both as  **separate modes** .

### Mode 1 — Semantic (Editable) mode (recommended default)

Goal: produce idiomatic LaTeX that stays editable.

* Use DOCX (OOXML) as truth for text + structure.
* Use a style layer (geometry/fontspec/setspace/enumitem/titlesec) to match Word defaults.
* Use the layout oracle only for hints (columns, table widths, textbox detection).
* Validate with PDF diff, but only tune **global knobs** (margins, line spacing, list indents).

Outcome: highly editable `.tex`, high semantic coverage, visually close.

### Mode 2 — Replica (Pixel-perfect) mode (optional)

Goal: produce a PDF that matches Word’s rendered PDF as closely as possible.

Two practical approaches:

**(A) “PDF as background + LaTeX overlay”**

1. Render DOCX → PDF (this is the target look).
2. Include each PDF page as a background.
3. Overlay content positioned at exact coordinates.

* If you only need identical visuals, you can stop at background inclusion.
* If you also want selectable/searchable text, overlay extracted text at precise x/y.

Common LaTeX building blocks:

* `pdfpages` to place the background pages
* `tikz` (overlay) or `textpos` / `eso-pic` to position text blocks

Example skeleton:

```tex
\includepdf[pages={1-},pagecommand={%
  \begin{tikzpicture}[remember picture,overlay]
    % place positioned text blocks here
  \end{tikzpicture}
}]{word_render.pdf}
```

**(B) “Extract geometry from the Word-PDF and re-emit”**

1. Render DOCX → PDF.
2. Parse the PDF to extract:
   * per-word or per-glyph bounding boxes
   * font family/size
   * images and their boxes
3. Generate LaTeX that places these elements with absolute positioning.

This can be very accurate because the PDF already contains the final layout decisions. However, the LaTeX becomes a *replay script* and is not semantically meaningful.

Outcome: near pixel-perfect output, low editability.

### What is realistically achievable?

* **For general docs:** guaranteed pixel-perfect *and* fully semantic editable LaTeX is not realistic.
* **For constrained templates (reports/forms):** you can get very close semantically, sometimes indistinguishable.
* If you truly need “same PDF every time,” prefer Replica mode and accept limited editability.

## 5) High-level architecture

### System diagram

```
User / Client
   |
   v
API Gateway / Web UI
   |
   v
Job Orchestrator / Queue  ------------------------------+
   |                                                   |
   +--> DOCX Parser (OOXML, deterministic)             |
   |        (truth: text + styles + structure)         |
   |                                                   |
   +--> Renderer (optional): DOCX -> PDF/images        |
   |        (layout oracle)                            |
   |          |                                        |
   |          +--> Layout ML (optional): region labels |
   |          +--> OCR (optional): geometry boxes      |
   |                                                   |
   +--> Asset Extractor (images/embeds/fonts)          |
   |                                                   |
   v                                                   |
IR Builder / Merger (canonical IR)  <-------------------+
   |
   v
Policy Engine (hard-case routing)
   |
   v
LaTeX Generator (semantic + style mapping)
   |
   v
LaTeX Compile + Validate
   |
   v
Packager + Storage
   |
   v
Outputs: output.tex + assets/ + output.pdf + report.json
```

### Responsibilities

* **API/UI** : upload, configure options, download outputs.
* **Orchestrator** : staged execution, retries, artifact tracking.
* **DOCX Parser** : canonical semantics + styles.
* **Renderer (optional)** : produce PDF/images to infer layout hints.
* **Layout ML/OCR (optional)** : region labels + geometry boxes.
* **IR Builder** : unify signals into one consistent representation.
* **Policy Engine** : semantic vs fallback decisions.
* **Generator** : emit LaTeX + assets.
* **Compiler/Validator** : compile PDF, extract warnings.
* **Report** : document what happened, where fallbacks occurred.

---

## 6) End-to-end pipeline (editable mode)

### Stage A — Deterministic DOCX extraction (required)

1. Unzip `.docx`.
2. Parse XML parts.
3. Build base IR:

* section setup (page size, margins, columns)
* ordered block stream (headings/paragraphs/lists/tables/figures)
* inline spans (run-level overrides)
* headers/footers per section

### Stage B — Optional rendering + layout hint extraction

1. Render DOCX → PDF.
2. Optionally render pages → images.
3. Run layout detection / OCR geometry:

* region labels (header/footer/table/figure/textbox)
* word/line bounding boxes (geometry only)

4. Derive hints:

* repeated header/footer zones
* column regions
* “object-like” blocks (textboxes/callouts)
* approximate table width/column proportions

### Stage C — IR enrichment

Merge deterministic structure with hints (deterministic text always wins). Hints are used to improve environment choices and spacing parameters.

### Stage D — Policy routing

For each difficult block, decide representation:

* semantic LaTeX
* fallback image

### Stage E — LaTeX generation

Emit:

* preamble (packages, fonts, lengths, macros)
* body (semantic blocks)
* assets directory

### Stage F — Compile + validate

Compile PDF, collect warnings, generate report.

---

## 7) Intermediate Representation (IR)

### IR design goals

* canonical and debuggable
* preserves style inheritance + overrides
* supports policy routing and fallbacks
* stable across implementations

### IR components

* **Document** : metadata + sections + ordered blocks + assets
* **Section** : page setup, columns, header/footer rules
* **Block** : typed structural element
* **InlineSpan** : text with inline style
* **Hints** : optional layout guidance

### Example block types

* `heading(level)`
* `paragraph`
* `list(ordered/unordered, nesting)`
* `table`
* `figure`
* `textbox`
* `page_break`
* `columns_region`

---

## 8) Style mapping strategy (Word styles → LaTeX)

### Why mapping matters

If we directly emit every inline style as raw LaTeX commands, output becomes unreadable. Instead:

* map paragraph styles to a controlled set of LaTeX macros
* apply inline overrides only when needed

### Strategy

1. Parse `styles.xml` + inheritance.
2. Choose semantic mapping:

* Heading styles → sections/subsections
* Normal → paragraph macro
* Quote → quote environment

3. Translate key layout parameters into LaTeX lengths:

* paragraph spacing before/after
* line spacing
* indentations
* list label widths and label separation

4. Use XeLaTeX/LuaLaTeX + `fontspec` to match fonts when needed.

---

## 9) Hard cases (policy playbook)

### 9.1 Text boxes / shapes / SmartArt

**Detection**

* OOXML: drawings (anchored/inline)
* layout hints: region labeled textbox/figure

**Policy**

* If it’s essentially text content in a box with simple placement:
  * emit editable framed box (`tcolorbox` or `mdframed`)
* Otherwise:
  * render to image and embed (`\includegraphics`)

**Reason**
SmartArt and complex shapes do not map cleanly to semantic LaTeX without major effort.

### 9.2 Tables

**Scoring inputs**

* number of merges (row/col)
* nested tables
* number of rows/columns
* border/shading variety
* fixed width constraints

**Policy**

* simple/medium → semantic table (`tabularx`, `multirow`, `multicolumn`)
* long → `longtable`/`ltablex`
* complex → image fallback

### 9.3 Multi-column layouts

**Detection**

* true columns: section properties (`sectPr/cols`)
* fake columns: tabs/spaces or textbox-based layout

**Policy**

* true columns → `multicol` or `paracol`
* fake columns → convert to semantic structures:
  * aligned columns → `tabularx`
  * label/value → description list
* unstable mixed layout → normalize to one column (editable) or fallback block

---

## 10) Optional layout oracle (PDF + ML/OCR geometry)

### Purpose

Improve decisions for:

* columns
* header/footer detection (repetition)
* detecting callouts/textboxes
* estimating table widths

### What it is used for

* labels and geometry (where blocks appear)
* confidence/warnings when DOCX authoring is inconsistent

### What it is not used for

* canonical text content
* exact kerning/glyph placement

---

## 11) Validation, report, and evaluation

### Validator outputs

* compile success/failure
* warnings: missing assets/fonts, overfull boxes, unresolved refs

### Report.json should include

* conversion summary
* semantic coverage (% semantic vs fallback)
* list of fallbacks (with reasons)
* warnings and manual review checklist

### Metrics to track

* compile success rate
* semantic coverage by type
* table success rate by complexity bucket
* textbox semantic conversion rate

Optional diagnostics:

* PDF visual similarity score (for guiding improvements, not for forcing per-line hacks).

---

## 12) MVP scope and roadmap

### v1 (shippable)

* paragraphs/headings with spacing
* lists with nesting
* images with captions
* basic headers/footers
* simple/medium tables
* policy fallbacks for SmartArt/textboxes/complex tables

### v2

* improved section-linked headers/footers
* better list label fidelity
* better table width estimation using layout hints

### v3

* more robust textbox handling
* improved multi-column region handling

---

## 13) Implementation notes (tech stack)

### Open-source friendly

* OOXML parse: unzip + XML parsing (any language)
* Python convenience: `python-docx` for traversal
* Rendering: LibreOffice headless
* PDF inspection: PyMuPDF/pdfplumber
* OCR geometry: PaddleOCR (optional)

### Commercial high-fidelity

* Aspose.Words for parsing + DOCX→PDF rendering

---

## 14) Risks and mitigations

* **Layout divergence** : tune global lengths; use layout oracle as hints.
* **Complex objects** : deterministic fallback + reporting.
* **Inconsistent Word authoring** : heuristics + warnings + optional layout hints.

---

## 15) Appendix

### A) Table complexity scoring (starter)

* merges: +2 per merged region
* nested table: +10 each
* columns > 8: +3
* rows > 30: +3
* mixed border styles: +5

Buckets:

* 0–8: simple → semantic table
* 9–18: medium → semantic table
* 19+: complex → fallback image

### B) Minimal IR example

```json
{
  "sections": [{
    "page": {"size": "LETTER", "margins": {"top": 72, "bottom": 72, "left": 72, "right": 72}},
    "columns": 1
  }],
  "blocks": [
    {"id": "b1", "type": "heading", "level": 1, "content": [{"text": "Title", "style": {"bold": true}}]},
    {"id": "b2", "type": "paragraph", "content": [{"text": "Body text...", "style": {}}]},
    {"id": "b3", "type": "table", "hints": {"complexity": 7}, "rows": 3, "cols": 3}
  ]
}
```

### C) Default policy summary

* textbox: framed if text-only; else image
* table: semantic if simple/medium; else image
* columns: support true OOXML columns; convert fake columns to tabular/description; normalize unstable

---

## How to use this document

If you’re new to this project:

1. Read sections 1–5 for the “what and why”.
2. Read sections 6–10 for the pipeline and optional layout oracle.
3. Read section 9 to understand hard-case policy behavior.
4. Use sections 11–12 for evaluation and MVP scope.




I’d do  **Layout Oracle first** , then expand the Policy Engine — because the oracle is what turns a bunch of “hard cases” from guesswork into  **deterministic decisions** .

Also: even if `scripts/compare_layout.py` is open, the best use of it is to make the oracle *measurable* (what changed, where, and why), not just “pixel diff says it’s different”.

Here’s the plan I’d propose.

---

## 1) Layout Oracle: what it should do (and what it must NOT do)

### What it should output (signals)

Per  **page** , produce a `LayoutHints` object:

* **Page geometry**
  * width/height, margins (from Word→PDF render if needed)
* **Regions (bboxes + labels)**
  * `header`, `footer`, `title`, `paragraph`, `list`, `table`, `figure`, `textbox/callout`
* **Reading order**
  * especially for **multi-column** pages
* **Column structure**
  * number of columns + column boundaries per region (not necessarily whole page)
* **Width hints**
  * table width, column proportions (rough), textbox width
* **Tabstop intent (optional but huge)**
  * detect “left text + right-aligned date/number” patterns on same line

### What it must NOT do

* It must not become the “truth” for text.
* It must not try to reproduce kerning/glyph-level placement.
* It must not force per-line breaks into semantic mode.

### How it informs conversion (examples)

* If oracle says `2 columns` for region X → wrap that region in `multicols/paracol`
* If oracle finds a right-aligned chunk near right margin on same baseline → use `tabular*` / `tabto` strategy, not `\hfill`
* If oracle says “textbox/callout” → policy routes to `tcolorbox` or image
* If oracle says “table region” but OOXML table is absent → policy routes to “reconstruct table” or fallback image

---

## 2) How `compare_layout.py` should fit in

Use it to answer  **two questions** :

1. *Where* do the PDFs diverge? (page number + bounding box)
2. *Which category* caused it? (table vs paragraph vs header)

A good workflow:

* Render both PDFs to images at the same DPI.
* Compute per-page diff metrics (SSIM / pixel error).
* Then run  **region-based diff** :
  * overlay oracle regions on both renders
  * attribute error to region types (“80% of error is in tables on page 2”)

That makes oracle + policy debugging surgical.

---

## 3) Implementation steps for Layout Oracle (practical)

### Step A — Render Word to PDF (ground truth)

* Word/LibreOffice/Aspose → `word.pdf`

### Step B — Convert PDF to page images

* fixed DPI (e.g., 150–200) for stable detection

### Step C — Run layout detection

Pick one:

* PubLayNet-style detector (fast and good)
* DiT/LayoutLMv3-style detector (stronger but heavier)

Output: list of `(bbox, label, score)`.

### Step D — Optional OCR geometry (boxes only)

* Run OCR to get word/line bounding boxes (ignore text if you already have OOXML text)
* Use it for:
  * baseline alignment (tabs)
  * column grouping

### Step E — Produce `LayoutHints` JSON

This is what the rest of the pipeline consumes.

---

## 4) Then expand the Policy Engine (because now it has better inputs)

### Tables

Policy should use  **both** :

* OOXML structure (rows/cols/merges)
* oracle width hints (table bbox + column proportions)

Decisions:

* If `complexity_score <= threshold` → emit LaTeX table
* else → fallback image + caption (and report)

### Textboxes / Callouts

Decisions:

* if textbox is “text-only simple” and fits cleanly → `tcolorbox`
* else → image snapshot

### Multi-column sections

* if OOXML says columns → trust that
* else if oracle sees stable two-column region → wrap only that region in `multicols`

---

## 5) What I’d respond to that message (the actual choice)

**Let’s discuss the Layout Oracle first.**
Because it directly improves the accuracy of the Policy Engine (tables/textboxes) and gives us measurable debugging via `compare_layout.py`. Once we agree on the oracle outputs (region labels, reading order, column boundaries, width hints), we can implement policy rules confidently and validate improvements region-by-region.


## The biggest correction: LayoutLMv3-base is the wrong default for “regions”

`microsoft/layoutlmv3-base` is primarily a **document encoder** for token/sequence tasks (it *can* be fine-tuned for layout, but out of the box it does **not** give you clean “table/figure/textbox region boxes” the way you want).

For a Layout Oracle that outputs  **regions + bboxes** , you want a  **vision layout detector** :

### Best default backend (what I’d ship)

* **Detectron2/Mask R-CNN** trained on **PubLayNet** (via LayoutParser)
  Outputs: `Text, Title, List, Table, Figure` regions. Very plug-and-play.

### Stronger but heavier

* **DiT + Cascade Mask R-CNN** fine-tuned for PubLayNet
  Higher quality regions, heavier stack.

### Where LayoutLMv3 actually fits (later)

Use it (or a smaller classifier) for  *semantic refinement* :

* caption vs body paragraph
* header/footer vs title
* “textbox/callout” classification for text regions that look like sidebars

**Action:** Update your plan so `layout_ml.py` uses a detector-first approach for regions, and keep LayoutLMv3 optional for refinement.

---

## Dependencies: your “ml” optional group idea is correct — but refine it

### Core deps (should be safe in main)

* `pymupdf` (for PDF rendering + extracting text blocks)
* `pillow`
* `numpy` (you’ll need it for metrics)

### Optional `ml` group (keep heavy stuff out of core)

* `torch`
* `detectron2` (or a pinned wheel strategy)
* `layoutparser` (nice wrapper, optional)
* (optional) `transformers` only if you use LayoutLM/Donut later

**Why this matters:** Detectron2 install can be painful on some environments. Keeping it optional avoids breaking your core install for users who just want “basic conversion”.

---

## Layout Oracle output contract: make it deterministic and easy to consume

Right now you say: “regions + column_structure + reading order.” Good — but  **define the coordinate system** .

### Strong recommendation: store bboxes as normalized coordinates

Per page:

* `page_w`, `page_h` in pixels for the render DPI you used
* bbox in normalized `[0..1]`:
  * `x0 = left / page_w`, `y0 = top / page_h`, `x1`, `y1`

This prevents DPI mismatch and makes region overlap robust.

### Suggested schema (practical)

* `regions[]`: `{label, score, bbox_norm, page_index}`
* `columns[]`: `{page_index, column_count, boundaries_norm[]}` (each boundary: x0/x1)
* `reading_order[]`: list of region IDs in reading order per page

---

## Column structure: don’t overcomplicate it

You can get good results with deterministic clustering on **text block x-centers** (from PyMuPDF) even without ML.

Algorithm (simple + reliable):

1. Extract text blocks from **ref PDF** with PyMuPDF.
2. Take each block’s x-center.
3. Run 1D clustering (k=1..3) and choose k by gap/variance heuristic.
4. Column boundaries = min/max of blocks per cluster.

This works shockingly well for “true” columns and avoids ML dependency for column detection.

---

## Region-based diff: compute *visual diffs by region* first (most robust)

Your plan says “text positional error per region type.” That sounds nice but is fragile because:

* PDF text extraction differs between Word-PDF and TeX-PDF
* hyphenation changes tokenization
* fonts cause different chunking

### Better v1: region-masked image diff

1. Render both PDFs to images at same DPI.
2. Run oracle on **reference** page images → regions.
3. For each region type:
   * crop both images to that bbox
   * compute region score: `SSIM` or normalized pixel diff
4. Report:
   * `Table: avg_diff=..., max_diff=...`
   * `Text: ...`
   * `Figure: ...`

This will reliably tell you “tables are causing 80% of divergence” without brittle text matching.

### v2 (optional): text-box shift metric

Only after v1:

* Extract spans from both PDFs (`page.get_text("dict")`)
* Within each region, match spans by (reading order index) or fuzzy string
* Compute average `(dx, dy)` drift

---

## `compare_layout.py` CLI upgrades (what I’d add)

Besides `--ref` and `--gen`, add:

* `--dpi 200`
* `--out report.json`
* `--save-overlays` (writes images with region boxes + diff heatmap)
* `--page N` (debug one page)

Output should include:

* per-page summary
* per-region-type aggregation
* top-k worst regions with bbox + page index

---

## Textboxes: don’t promise them from PubLayNet

PubLayNet doesn’t have a “textbox/callout” class.
So your plan line “Ensure it returns distinct regions for Textboxes” is risky.

**Practical approach:**

* Detect as `Text` region (from detector)
* Then classify “textbox/callout” using heuristics:
  * bounded by a rectangle border? (detect lines)
  * background shading? (image statistics)
  * unusually narrow width + offset?
  * anchored near figure/table?

If you later add LayoutLMv3 refinement, you can label some examples and train a small classifier.

---

## Verification plan: tighten it

Your current tests are mostly “doesn’t crash.” Add one real regression check:

### Add golden layout hints snapshot test

* Run oracle on a fixed PDF sample
* Assert that:
  * #tables detected == expected
  * regions exist on page X with roughly expected bbox ranges (± tolerance)
  * column_count correct on known multi-column sample

This makes “oracle quality” measurable across commits.

---

## What I would change in your plan (summary)

1. **Switch primary region model** to Detectron2 PubLayNet (or DiT Cascade RCNN).
2. Keep ML in optional `ml` dependency group (good call).
3. Make `LayoutHints` use  **normalized bbox coordinates** .
4. Make region-based diff  **image-based first** , text positional error optional later.
5. Treat “textbox” as a heuristic/refinement class, not a guaranteed ML label.
