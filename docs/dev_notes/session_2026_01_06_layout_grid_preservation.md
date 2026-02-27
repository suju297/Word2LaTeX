# Layout Grid Preservation - Session 2026-01-06

## Objective
Harden layout table detection to production-grade level. Key goal: preserve Word's 2-column layout geometry instead of flattening to semantic blocks.

## Key Discovery
**Root Cause of Title Misalignment**: Flattening layout tables to `\section*{}` blocks destroyed the 2-column geometry. Titles that should be in the right column (X~158px) were being placed at the left margin (X~40px).

### Proof (locate_headings.py output)
```
Title           | Old dx   | After LayoutGridBlock
----------------|----------|----------------------
PROFILE         | +117.5   | -9.4  (FIXED)
WORK EXPERIENCE | +116.1   | -10.7 (FIXED)
TYLER STEIN MD  | +115.7   | -11.2 (FIXED)
```

## Implementation

### 1. New IR Types (`ir.py`)
```python
@dataclass
class GridCell:
    blocks: List["Block"]
    col_span: int = 1
    row_span: int = 1
    padding_twips: Optional[Dict[str, int]] = None  # tcMar overrides

@dataclass
class GridRow:
    cells: List[GridCell]
    height_twips: Optional[int] = None  # from w:trHeight
    height_rule: Optional[str] = None   # "exact" | "atLeast"

@dataclass
class LayoutGridBlock(Block):
    col_widths_twips: Optional[List[int]] = None  # from w:tblGrid
    rows: List[GridRow]
    cell_margin_twips: Optional[Dict[str, int]] = None  # tblCellMar
```

### 2. Parser Changes (`document.py`)
- Added `_parse_layout_grid()` function that:
  - Extracts `col_widths_twips` from `w:tblGrid/w:gridCol`
  - Extracts `height_twips` and `height_rule` from `w:trPr/w:trHeight`
  - Extracts `padding_twips` from `w:tcPr/w:tcMar`
  - Extracts `cell_margin_twips` from `w:tblPr/w:tblCellMar`
  - Parses cell content as nested blocks (preserving paragraph structure)

- Changed policy from `LAYOUT_TABLE_FLATTENED` to `LAYOUT_GRID_PRESERVED`

### 3. Generator Changes (`generator.py`)
- Added `_format_layout_grid()` that renders as borderless `tabularx`:
  - Column widths as fractions from `col_widths_twips`
  - Row heights via struts (`\rule{0pt}{H}`)
  - Top-aligned cells via `minipage[t]`
  - Default cell margins (2pt) when not specified
  - Paragraph spacing inside cells

- Added `_format_cell_blocks()` with explicit `\vspace*` for spacing

## Results (Resume_1)

### Before vs After
| Metric | Before (Flattened) | After (Grid Preserved) |
|--------|-------------------|------------------------|
| dx offset | +117px | -10px |
| dy pattern | All negative, worsening | Mixed (+28 to -65) |
| blank_gen_count | 3 | 1 |
| hit_boundary_count | 4 | 1 |

### Remaining Issues
- **dy still mixed** (+28 to -65): Baseline alignment inside cells not matching Word
- **title_002 (CITY HOSPITAL)** still `blank_gen=True`: Content at dy=-65
- Negative SSIM on small crops (not a bug - SSIM can be <0)

## Files Changed
- `src/wordtolatex/ir.py`: Added GridCell, GridRow, LayoutGridBlock
- `src/wordtolatex/docx_parser/document.py`: Added _parse_layout_grid()
- `src/wordtolatex/generator.py`: Added _format_layout_grid(), _format_cell_blocks()
- `scripts/locate_headings.py`: New debug script for heading bbox comparison

## Next Steps (Not Yet Implemented)
1. **Cell vertical alignment** (`w:vAlign`): Word cells can be top/center/bottom
2. **Style-level line height**: Inherit from Normal style
3. **Table vertical anchoring**: `w:spacing before` on containing paragraph

## Key Learnings
1. Layout tables should be preserved as grids, not flattened to semantic blocks
2. Row heights are critical for preventing cumulative vertical collapse
3. Default cell margins matter even when not specified in OOXML
4. Top alignment (`minipage[t]`) helps but doesn't fully match Word's baseline positioning
