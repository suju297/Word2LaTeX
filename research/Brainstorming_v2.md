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

### 10.1 Implementation update (v1)

A practical v1 Layout Oracle can be implemented with a **hybrid** approach:

* **Deterministic columns (no ML):**
  * Extract PDF text blocks (e.g., via PyMuPDF).
  * Compute each block’s center-x.
  * Identify 1–3 columns by finding stable vertical “gutters” (large x-gaps / histogram valleys).
  * Output column boundaries as normalized coordinates.
* **Vision region detection (optional ML):**
  * Use LayoutParser + Detectron2 (PubLayNet Mask R-CNN) to detect region bboxes for `Title`, `List`, `Table`, `Figure`, `Text`.
  * Make this an **optional dependency group** so the core tool stays lightweight and installs reliably.
  * At runtime, if the detector is unavailable, return empty regions (do not crash) and rely on deterministic + DOCX-derived structure.
* **Coordinate normalization:**
  * Store all bboxes normalized to `[0..1]` relative to page dimensions to avoid DPI coupling.

This split keeps the oracle useful even when ML deps are missing, while enabling richer region hints when installed.

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



This is a good update. You made the  **right architectural call** : *deterministic columns* +  *optional vision regions* . That’s exactly how you keep the core stable while still allowing “oracle upgrades” when ML is available.

Here’s what’s solid, what I’d tweak, and what to do next.

---

## What you did right

### 1) Column detection as deterministic

Your gutter-based approach is the right default because:

* it works without ML
* it’s fast
* it’s debuggable
* it avoids dependency hell

For resumes and many reports, this will be surprisingly reliable.

### 2) LayoutParser PubLayNet for regions (optional)

That’s the correct “regions backend” for v1. Also good choice to do runtime import checks and return empty regions rather than crashing.

### 3) Normalized coordinates

This is mandatory for long-term sanity. Good.

---

## The main gap (and how to fix it)

### Regions = 0 because Detectron2 isn’t installed

If the oracle produces no regions on most machines, the Policy Engine won’t get value from the “region” side.

So you need **at least one non-ML fallback** for “table/figure/textbox-ish” even when detectron2 is missing.

#### Minimal non-ML region fallback (worth doing)

Use PyMuPDF + OpenCV (you already added opencv-headless) to detect:

1. **Figures/images**

* PyMuPDF can enumerate images per page and give you their bounding boxes.
* That alone gives you `Figure` regions without detectron2.

2. **Tables**

* Do line-detection inside the page image crop:
  * binarize → detect horizontal + vertical rulings via morphology
  * if there’s a grid-like density → label region as `Table`
    This won’t catch all “borderless tables”, but it catches the common hard ones.

3. **Textboxes/callouts**

* Detect filled backgrounds or bordered rectangles:
  * look for rectangle border lines
  * or compute background variance/mean vs page background
    If present, label region `Textbox`.

This gives you “regions are usually non-empty” without ML.

---

## Column detection edge cases you should handle now

Your “gap > 15% page width” works, but add guardrails:

1. **Don’t treat margins as gutters**
   Ignore gaps near the left/right margins (e.g., within first/last 10% width).
2. **Use y-coverage**
   A real column gutter typically exists across a big vertical span.
   Require that “gap” is supported by blocks across, say, >40% of page height.
3. **Sidebars**
   A sidebar can look like a second “column”.
   Add a rule: if one cluster has very small total area (or very narrow width), classify as `sidebar` not `2-column`.

These make your column detector feel “enterprise-grade”.

---

## Dependency strategy: your optional `ml` group is correct

Two tweaks:

1. Keep `opencv-python-headless` in optional too unless you truly need it in core
   (OpenCV is heavy and often causes conflicts.)
2. Provide a single blessed setup path for ML users:

* **Docker image** (best, easiest)
* or **conda env file**
* or pinned “torch+detectron2” combos in docs

If someone can’t install detectron2 in 5 minutes, they’ll drop the tool.

---

## compare_layout.py: change what you measure (v1)

Right now you mention “text positional error per region type”.
That will be brittle.

### Better v1: region-masked image diff

* Render both PDFs at same DPI
* Use reference regions from oracle
* For each region: crop both images → compute pixel diff metric
* Report per-type drift and top worst regions

This will correctly blame “tables” even when text extraction differs.

Then you can add text-pos drift as v2.

---

## Next steps (high priority order)

### Step 1 — Make oracle useful without detectron2

Implement:

* figure regions via PyMuPDF image boxes
* table regions via line density heuristics (OpenCV)
* textbox regions via border/shading heuristics

### Step 2 — Wire Policy Engine using your hints

Start with deterministic, high-confidence rules:

* if `columns=2` for a page/section → `multicols` for that region
* if region label `Table` + doc has real OOXML table → emit LaTeX table with width hint
* if region label `Table` but OOXML has no table → fallback image
* if region label `Textbox` → `tcolorbox` when simple else fallback image
* if region label `Figure` → LaTeX figure + caption heuristic

### Step 3 — Make compare_layout actionable

* emit JSON report
* save overlay PNGs with bboxes + heatmap
* output “top 5 worst regions” with coords

### Step 4 — Add one “golden” regression test

* known PDF sample
* assert: column_count, at least one figure/table region, bbox ranges within tolerance

---

## One more thing: your writeup reads like a PR summary

That’s good. If you want this to be “sellable”, keep doing that.

I also folded your architecture update into our project doc so it stays consistent with the earlier design (deterministic columns + optional vision regions).

If you paste your `layout_hints.json` schema (or the dataclasses), I’ll tell you exactly what fields to add so the Policy Engine stays clean and future-proof (versioning, region IDs, page index, confidence, provenance).





Nice — this is a real milestone. You’ve now got an Oracle that (a) works without ML for columns and (b) can emit meaningful regions when ML is available. That’s exactly the “hybrid, deterministic-first” architecture you wanted.

That said, your update has **two red flags** you should fix before you wire this into the Policy Engine.

---

## 1) Red flag: `NUM_CLASSES=6` is likely wrong for PubLayNet

PubLayNet’s canonical label set is  **5 foreground classes** : `Text, Title, List, Table, Figure`. In Detectron2, `MODEL.ROI_HEADS.NUM_CLASSES` should be the  **number of foreground classes** , not “+ background”.

If your config uses `NUM_CLASSES=6`, you will often see **weird class behavior** (like “table detected in header” or over-predicting Title/List).

**What to do right now**

* Verify what your checkpoint expects:
  * If the checkpoint is actually from PubLayNet standard training, set `NUM_CLASSES=5`.
  * If it truly was trained with 6 labels, document the label map and make sure `layoutparser` is using that same map.

**Quick sanity check**
Run the detector on a known scientific page (tables/figures common). If you still only see Titles/Lists, your class config or label map is off.

---

## 2) Red flag: “1 Table detected via heuristic fallback (header section)”

That’s almost certainly a **false positive** (headers often have lines/shading that look like table rules).

Before the Policy Engine trusts a `Table` region, add a  **table confidence gate** :

### Table gate (simple + effective)

Only accept a region as `Table` if at least 2 of these are true:

* **Grid evidence** : enough horizontal AND vertical line segments
* **Line intersection density** : intersections above threshold
* **Text alignment** : multiple text blocks aligned into columns inside bbox
* **Aspect ratio** : table bbox not too thin (reject short banner-like regions)

And if ML says `Title/List` but heuristic says `Table`, prefer ML unless heuristic confidence is very high.

---

## 3) Clean up the Oracle output contract (so Policy Engine stays sane)

Now that ML is operational, you should make every region include:

* `label` (Title/List/Table/Figure/Textbox/etc.)
* `bbox_norm` in `[0..1]`
* `score` (ML confidence or heuristic confidence)
* `source` = `"ml"` or `"heuristic"`
* `page_index`
* stable `region_id`

This lets Policy do deterministic rules like:

* “Only treat as Table if `label=Table` and `score>=0.8` and `source=ml`”
* “If `source=heuristic`, require stronger validation”

---

## 4) Next step: wire Policy Engine using *safe* rules first

Start with rules that won’t backfire:

### Column policy (safe)

* If oracle says 2-column for a page/section → wrap that section in `multicols` (or convert as two independent flows).
* If it’s “sidebar-like” (one narrow cluster) → treat as textbox/callout, not true columns.

### Title/List policy (safe)

PubLayNet “Title” is not always a LaTeX `\section`. For resumes it often marks:

* the name line
* section headers
* sometimes bold-ish subheaders

So do:

* Top-most Title at top of page → header block (not `\section`)
* Titles below that with consistent y spacing → section headings
* Lists → `itemize` only if you see bullet/number markers in DOCX text; otherwise treat as paragraph group

### Table/Figure policy (guarded)

* If `Table` region exists **and** DOCX contains a real table intersecting that bbox → emit LaTeX table
* Else fallback to image (don’t attempt semantic table reconstruction yet)
* Figures: if PDF has an image bbox in that region → figure environment; else ignore

---

## 5) Upgrade `compare_layout.py` now that regions exist

Don’t do “text positional error” first. Do this first:

### v1: region-masked image diff

For each region type:

* crop ref and gen renders
* compute pixel-diff metric
* aggregate by type
* output top 5 worst regions with bboxes

This will make your diff report  *actually useful* .

---

## 6) What I’d do in your repo next (concrete checklist)

1. **Fix class count** (`NUM_CLASSES`) + verify label map
2. Add **table false-positive guard** (especially for heuristic tables)
3. Add `source/score/region_id` to every region
4. Implement **region-masked diff** in `compare_layout.py`
5. Add one “golden oracle” test asserting:
   * column count is correct
   * region labels exist and are within bbox tolerances


## What looks solid (and why it matters)

### Golden tests

* ✅ Having `sample_table.pdf` and `sample_textbox.pdf` is the right move. It prevents the “resume falsely detected as table” trap.
* ✅ Confidence thresholds in tests (`>0.8`) are good, as long as you keep them tied to stable signals (intersection density, not fragile OCR text).

### Table Gate with intersection counts

* ✅ This is the correct gating primitive.
* ⚠️ Make sure the intersection count is *normalized* (by area / expected cell size). Otherwise, DPI changes or scaling will change your score even with normalized bboxes.

### Textbox detection gated by “floating”

* ✅ This is crucial. Without it, shaded headers become “textboxes.”

### Policy Engine: LayoutContext + conflict resolution

* ✅ Typed `LayoutContext` prevents a lot of future bugs.
* ✅ “OOXML first” is the correct baseline.
* ✅ Allowing oracle to flag figures/textboxes is the right exception mechanism.

### compare_layout.py visuals

* ✅ Overlays are non-negotiable for debugging. Good that it’s a flag.

---

## The 5 upgrades I’d do next (high ROI)

### 1) Normalize Table Gate scores

Right now you say “confidence > 0.8”. Make that confidence derived from stable, scale-invariant features:

* `grid_score = f(line_density_norm, intersection_density_norm, orthogonality)`
* `intersection_density_norm = intersections / area`
* `line_density_norm = (h_lines + v_lines) / (sqrt(area))` (or similar)

This makes thresholds transferable across documents.

---

### 2) Add a “borderless table” pathway (common in Word)

Many Word tables have no ruling lines. Your line-based gate will miss them.

Add a second detector that triggers when:

* text blocks inside a region align into **consistent columns** across multiple rows
* multiple x-clusters repeat across y

This doesn’t need ML and will catch a lot of real tables.

---

### 3) Region provenance and confidence should influence policy decisions

Make sure Policy uses:

* `source = ml|heuristic|ooxml`
* `confidence`
* `region_type`

Example:

* heuristic `table` → prefer fallback image unless OOXML confirms table
* ml `table` with high confidence → attempt semantic table if OOXML matches, else fallback image

This prevents heuristics from overreaching.

---

### 4) Expand golden suite with one “conflict case”

Add a PDF where:

* OOXML indicates a normal paragraph
* oracle flags a textbox region (e.g., callout)

Your test should confirm the resolver:

* does not reclassify semantics incorrectly
* but triggers special handling (tcolorbox or image fallback)

This will validate your conflict rules actually work.

---

### 5) NUM_CLASSES validation should be automated, not “if/when”

You can validate without guessing by checking the checkpoint metadata (or inference sanity):

* assert label map length matches `NUM_CLASSES`
* run on a known PubLayNet-like page and ensure it can output **Table** and **Figure** sometimes (not only Title/List)
* log class distribution

If your model predicts only Title/List, it’s almost always config/label-map mismatch.

---

## Next Steps: your plan is correct, just sharpen it

### “Run on a larger dataset”

Yes, but do it with a goal:

* produce a report: false positive rate for tables/textboxes
* tune thresholds once, then lock them with regression tests

### “Verify NUM_CLASSES”

Do it as soon as you can. If ML is working, you want it dependable; if it’s flaky, you’ll waste time debugging downstream policy.

---

## What I want you to produce next (to keep momentum)

One artifact that makes this system credible:

**A single `report.json` per document run** containing:

* column count per page
* region counts by type + source
* top 5 drift regions from compare_layout
* policy fallbacks triggered (table->image, textbox->tcolorbox, etc.)

That is the “enterprise-grade” story.

---

If you paste the output of one run (the `layout_hints.json` schema + a sample policy decision trace), I’ll tell you exactly what fields are missing to make this fully traceable and debuggable end-to-end.



What looks good

* Column detection is working and even finds true 2-column pages in at least one run (page 3–4 show `count: 2`).
* Your oracle is producing region summaries (titles/lists/figures show up across runs).

## What’s **not** consistent yet (needs fixing before scaling)

### 1) `policy_trace` is empty in every `report.json`

All the reports you uploaded have `"policy_trace": []`.
So either:

* `compare_layout.py` is writing this `report.json` and isn’t merging the conversion pipeline’s `conversion_report.json`, or
* the pipeline isn’t actually emitting trace into the artifact you uploaded.

**Fix:** keep two artifacts (`conversion_report.json` + `layout_diff_report.json`) *or* explicitly merge them into one final report in a single place (recommended: a “runner” script that calls convert + compare and merges outputs).

---

### 2) Your oracle labels still look “half-normalized”

You’ve got keys like `"title (unknown)"`, `"list (unknown)"`, `"figure (unknown)"`.
That suggests label normalization / provenance formatting isn’t finalized (or you’re mixing multiple label sources).

**Fix:** enforce a canonical enum (`title/list/table/figure/textbox/text`) and always store `source` and `score` per region.

---

### 3) Column boundaries don’t look normalized

You said everything is normalized to `[0..1]`, but boundaries like `252.8456` look like pixels/points.

**Fix:** store both:

* `boundary_norm` (0..1) for logic
* `boundary_px` for debugging overlays

---

### 4) Drift metrics look noisy and “unknown-heavy”

One report shows `"unknown"` drift count 238 with very large average shift.
That often means the block matching in compare is weak (or the PDF text extraction differs a lot).

Also, one run shows `original_blocks: 250` but `generated_blocks: 1` which strongly hints extraction failed on the generated PDF.

**Fix (priority):**

* Add a hard check: if generated_blocks is “too small”, fail the run with an error (“PDF text extraction failed; cannot compute drift”).
* Prefer **region-masked image diff** as your primary drift metric (text-based drift can be v2).

---

## Go/No-Go

✅ **Go** for scaling the corpus, **after** you fix (1) policy trace integration and (3) coordinate normalization. Otherwise you’ll tune thresholds using unreliable signals.

If you want, upload (or paste) one `conversion_report.json` from the pipeline too, and I’ll tell you exactly how to merge it with this `report.json` into a single “enterprise-grade” final report (with clean reason codes + drift attribution).
