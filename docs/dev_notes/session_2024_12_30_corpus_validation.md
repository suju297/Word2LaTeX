# Layout Oracle & Policy Engine - Development Session Notes

**Date:** December 30, 2025  
**Session:** Corpus Validation & Report Quality Fixes

---

## Summary of Work Done

This session focused on validating the Layout Oracle and Policy Engine against a 7-document corpus, identifying report quality issues, and implementing fixes.

---

## Key Files Modified

### Core Pipeline
- `src/wordtolatex/pipeline.py` - Real policy context (not "simulated")
- `src/wordtolatex/policy.py` - Handle dict regions, cleaner code
- `src/wordtolatex/layout_oracle.py` - Source propagation (`ml`/`heuristic`), normalized boundaries
- `src/wordtolatex/layout_heuristics.py` - Table gate, textbox detection
- `src/wordtolatex/layout_ml.py` - NUM_CLASSES validation, source field

### Scripts
- `scripts/run_corpus.py` - Word PDF fallback, report merging
- `scripts/compare_layout.py` - Hard-fail extraction, canonical labels, proper doc_id

### Tests
- `tests/test_golden_layout.py` - 13 comprehensive tests (all pass)

---

## Fixes Implemented

### Fix 1: Hard-fail for Extraction Failures
**File:** `scripts/compare_layout.py`
```python
threshold = max(10, int(0.1 * orig_blocks))
if gen_blocks < threshold:
    report["diff_status"] = "INVALID_DIFF_EXTRACTION"
```

### Fix 2: Real Policy Context (not "simulated")
**File:** `src/wordtolatex/pipeline.py`
```python
ctx_summary = {
    "page": page_idx,
    "columns": col_count,
    "bbox": block_bbox,
    "overlaps": len(overlaps)
}
```

### Fix 3: Canonical Labels
**File:** `scripts/compare_layout.py`
```python
oracle_summary = {"by_label": {}, "by_source": {}, "regions": []}
CANONICAL_LABELS = {"title", "list", "table", "figure", "textbox", "text"}
```

### Fix 4: Proper doc_id
**File:** `scripts/compare_layout.py`
```python
doc_id = args.ref.parent.name if args.ref.parent.name != "." else args.ref.stem
```

### Fix 5: Source Propagation
**File:** `src/wordtolatex/layout_oracle.py`
```python
# When creating regions from ML
source="ml"

# When creating regions from heuristics
source="heuristic"

# In create_layout_hints
"source": r.source
```

---

## Report Schema v1.1

```json
{
  "report_version": "1.1",
  "doc_id": "Document_Name",
  "oracle_summary": {
    "by_label": { "title": 11, "list": 1 },
    "by_source": { "ml": 12 },
    "regions": [{ "label": "title", "source": "ml", "confidence": 0.93, "page": 0 }]
  },
  "columns": [{ "page": 0, "count": 1, "boundaries": [] }],
  "drift_summary": [{ "type": "title", "count": 6, "avg_shift": 29.0 }],
  "diff_status": "VALID" | "INVALID_DIFF_EXTRACTION" | "SKIPPED_NO_GEN",
  "diff_stats": { "original_blocks": 79, "generated_blocks": 61 },
  "policy_trace": [{
    "block_id": "b0",
    "type": "heading",
    "decision": "semantic",
    "reason": "DEFAULT_SEMANTIC",
    "layout_context": { "page": 0, "columns": 1, "bbox": [0,0,1,0.02], "overlaps": 10 }
  }],
  "policy_stats": { "DEFAULT_SEMANTIC": 29 }
}
```

---

## Corpus Validation Results

| Document | diff_status | by_source | columns |
|----------|-------------|-----------|---------|
| ACM_SUBMISSION_TEMPLATE | VALID | unknown:2 | 1 (13 pages) |
| Custom_Resume | VALID | unknown:8 | 1 (2 pages) |
| IEEE_TEMPLATE | VALID | unknown:8 | 1 (4 pages) |
| Microsoft_Resume1 | SKIPPED_NO_GEN | unknown:4 | 1 (3 pages) |
| Microsoft_Resume_2 | INVALID_DIFF_EXTRACTION | unknown:5 | 1-2 (5 pages) |
| Resume_1 | VALID | unknown:9 | 1 (1 page) |
| Resume_2 | SKIPPED_NO_GEN | ml:12 ✅ | 1 (1 page) |

**Note:** Most show `unknown` because cached data was used before the source fix. Only `Resume_2` was re-run with the fix.

---

## Current Status

### What's Working ✅
1. Hard-fail for extraction failures
2. Real policy context (not "simulated")
3. Canonical labels + structured oracle_summary
4. Proper doc_id from parent directory
5. 2-column detection (MS_Resume_2 pages 3-4)
6. 13/13 golden tests pass

### What's NOT Working ❌
1. Source provenance shows "unknown" for cached runs
2. LaTeX compilation failures for some docs (MS_Resume1, Resume_2)
3. High drift values (529pt for ACM) - likely matching issues

---

## Immediate Next Steps

1. **Clear cache and re-run corpus:**
   ```bash
   rm -rf tests/corpus/*/ref.pdf tests/corpus/*/gen.pdf tests/corpus/*/report.json tests/corpus/*/conversion_report.json
   uv run python scripts/run_corpus.py
   ```

2. **Fix LaTeX compilation for MS_Resume1:**
   - Check for missing image assets
   - Check for unsupported Unicode characters

3. **Add region-masked image diff** (short-term)
   - Current text-based drift is too noisy
   - Image diff would be more reliable KPI

---

## Key Commands

### Run corpus validation
```bash
uv run python scripts/run_corpus.py
```

### Run golden tests
```bash
uv run pytest tests/test_golden_layout.py -v
```

### Run single document pipeline
```bash
uv run python -m wordtolatex.pipeline <docx> <ref.pdf> <output.tex>
```

### Compare layouts
```bash
uv run python scripts/compare_layout.py --ref ref.pdf --gen gen.pdf --out report.json --hints hints.json --visuals
```

---

## Architecture Reference

```
DOCX Input → OOXML Parser → IR Builder
     ↓
PDF Render → Layout Oracle
                  ↓
           ML Regions (PubLayNet) + Heuristic Regions
                  ↓
           Column Detection (Gutter Analysis)
                  ↓
           Policy Engine → LaTeX Generator → output.tex
                  ↓
           conversion_report.json (policy trace)
```

---

## Test Suite Coverage

| Test Category | Count | Status |
|---------------|-------|--------|
| Table Heuristics | 3 | ✅ |
| Textbox Heuristics | 2 | ✅ |
| Column Detection | 2 | ✅ |
| Policy Engine | 3 | ✅ |
| Integration | 2 | ✅ |
| Report Integrity | 1 | ✅ |
| **Total** | **13** | **✅ All Pass** |

---

## Brainstorming Feedback Items (All Addressed)

From `research/Brainstorming_v2.md`:

1. ✅ Policy trace empty → Fixed (now merges conversion_report.json)
2. ✅ Oracle labels not normalized → Fixed (canonical labels + structured format)
3. ✅ Column boundaries not normalized → Fixed (stored as [0..1])
4. ✅ doc_id always "ref.pdf" → Fixed (uses parent directory name)
5. ✅ Hard-fail on extraction → Implemented (INVALID_DIFF_EXTRACTION)
6. ✅ Make policy context real → Implemented (page, columns, bbox, overlaps)

---

## Dependencies

- PyMuPDF (fitz) - PDF parsing
- LayoutParser + Detectron2 - ML region detection (optional)
- OpenCV - Heuristic table detection
- python-docx - OOXML parsing
- Jinja2 - LaTeX templating
- scikit-learn - Column clustering (KMeans)

---

## Environment

```bash
# Activate
cd /path/to/wordTOlatex
source .venv/bin/activate

# Or use uv
uv run python ...
```

---

*End of session notes. Continue from here in next chat.*

---

## Session Continuation - Priority 1-4 Completion

**Date:** December 30, 2025 (Evening)  
**Run ID:** 20251230_184129

### Implemented Changes

1. **run_corpus.py** - Added run_id freshness, build_meta.json, validate_corpus with 3 checks, --clean flag
2. **compare_layout.py** - Report v1.2 with diff_health/layout_quality separation, match_stats
3. **image_diff.py** - NEW script for SSIM/MAD visual similarity
4. **layout_oracle.py** - compare_layouts now returns matched_pairs

### Corpus Results (7 docs)

| Doc | gen.pdf | diff_health | by_source | match_rate | image_sim |
|-----|---------|-------------|-----------|------------|-----------|
| ACM | ✅ 127KB | VALID | ml:2 | 0.40 | 0.92 |
| Custom_Resume | ✅ 92KB | VALID | ml:8 | 0.72 | 0.88 |
| IEEE | ✅ 109KB | INVALID_MATCH_COVERAGE | ml:8 | 0.20 | 0.90 |
| MS_Resume1 | ❌ 0B | - | - | - | - |
| MS_Resume_2 | ❌ 0B | - | - | - | - |
| Resume_1 | ❌ 0B | - | - | - | - |
| Resume_2 | ❌ 0B | - | - | - | - |

### Key Wins
- **Provenance fixed**: All show `ml` source, not `unknown`
- **diff_health working**: IEEE correctly marked INVALID_MATCH_COVERAGE
- **Image diff integrated**: 0.88-0.92 similarity scores

### Remaining Issues
1. 4/7 docs have LaTeX compilation failures (Unicode, tabularx)
2. Match rate still low (20-72%) due to exact text matching
3. IEEE column detection showing 1-column (should be 2)

