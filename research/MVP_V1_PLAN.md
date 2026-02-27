# Word → LaTeX Converter: MVP v1 Complete Guide

## 1. Overview

**Goal**: Convert `.docx` to editable semantic LaTeX (.tex)  
**Approach**: OOXML parsing → IR → LaTeX generation (no ML models in v1)

---

## 2. Tech Stack (Finalized)

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Rich XML/templating ecosystem |
| **Package Manager** | `uv` | Fast, modern |
| **XML Parsing** | `lxml` | XPath, speed, full control |
| **Templating** | `Jinja2` | Flexible LaTeX generation |
| **CLI** | `click` | Clean argument parsing |
| **IR Classes** | `dataclasses` | Built-in, lightweight |
| **Testing** | `pytest` | Standard |
| **Formatting** | `black` + `ruff` | Fast, consistent |

---

## 3. Dependencies

### `requirements.txt`
```text
lxml>=5.0
Jinja2>=3.1
click>=8.0
```

### `requirements-dev.txt`
```text
pytest>=8.0
pytest-cov>=4.0
black>=24.0
ruff>=0.1
```

---

## 4. Project Structure

```text
wordTOlatex/
├── src/
│   ├── __init__.py
│   ├── docx_parser/
│   │   ├── __init__.py
│   │   ├── document.py      # Parse document.xml → blocks
│   │   ├── styles.py        # Parse styles.xml → style map
│   │   ├── numbering.py     # Parse numbering.xml → lists
│   │   ├── media.py         # Extract images
│   │   └── tables.py        # Parse tables + scoring
│   ├── ir.py                # IR dataclasses
│   ├── policy.py            # Semantic vs fallback
│   ├── generator.py         # IR → LaTeX
│   └── templates/
│       ├── document.tex.j2
│       └── preamble.tex.j2
├── tests/
│   ├── samples/             # Test .docx files
│   └── test_*.py
├── main.py                  # CLI
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

---

## 5. MVP v1 Scope

### ✅ Supported
- Paragraphs with spacing
- Headings (H1-H6 → `\section`, `\subsection`, etc.)
- Bold, italic, underline inline styles
- Bullet lists (`itemize`)
- Numbered lists (`enumerate`)
- Nested lists (up to 3 levels)
- Simple tables (no merges) → `tabularx`
- Images → `\includegraphics`
- Page setup (margins, size) → `geometry`

### ⚠️ Fallback (Image)
- Complex tables (merges, nesting)
- Textboxes/shapes
- SmartArt

### ❌ Not in v1
- Headers/footers
- Footnotes/endnotes
- Multi-column layouts
- Layout Oracle (PDF hints)

---

## 6. IR Schema (Simplified)

```python
@dataclass
class Document:
    metadata: dict
    sections: list[Section]
    blocks: list[Block]
    assets: list[Asset]

@dataclass
class Section:
    page_width: float
    page_height: float
    margins: dict  # top, bottom, left, right

@dataclass
class Block:
    id: str
    type: str  # heading, paragraph, list, table, figure
    style_ref: str | None
    content: list[InlineSpan]
    children: list[Block] | None  # For lists/tables

@dataclass
class InlineSpan:
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
```

---

## 7. Implementation Phases

| Phase | Deliverable | Tasks |
| :--- | :--- | :--- |
| **1** | Scaffolding | Create structure, `ir.py`, `pyproject.toml` |
| **2** | Basic Parser | Parse `document.xml` → paragraphs |
| **3** | Styles | Parse `styles.xml`, detect headings |
| **4** | LaTeX Gen | Jinja2 templates, basic output |
| **5** | Lists | Parse `numbering.xml`, nested lists |
| **6** | Tables | Parse `w:tbl`, simple tables |
| **7** | Images | Extract `media/`, `\includegraphics` |
| **8** | Policy | Table scoring, fallback stubs |
| **9** | Testing | Sample docs, pytest suite |

---

## 8. Key OOXML Elements

| Word Element | XML Tag | LaTeX Output |
| :--- | :--- | :--- |
| Paragraph | `<w:p>` | Text or `\par` |
| Heading 1 | `<w:p>` + style `Heading1` | `\section{}` |
| Bold | `<w:b/>` | `\textbf{}` |
| Italic | `<w:i/>` | `\textit{}` |
| Bullet list | `<w:numPr>` + bullet numId | `\begin{itemize}` |
| Table | `<w:tbl>` | `\begin{tabularx}` |
| Image | `<w:drawing>` | `\includegraphics{}` |

---

## 9. Quick Start Commands

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Run
python main.py input.docx output.tex

# Test
pytest tests/ -v
```

---

## 10. Success Criteria

- [ ] Compiles without errors on 90%+ of test docs
- [ ] Headings map to correct `\section` level
- [ ] Lists render with proper nesting
- [ ] Simple tables render as `tabularx`
- [ ] Images extract and embed correctly
