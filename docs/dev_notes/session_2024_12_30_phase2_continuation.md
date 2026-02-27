# Session Continuation - Pipeline Reliability Phase 2

**Date:** December 30, 2025 (19:00 EST)  
**Run ID:** 20251230_185256

---

## Current State

### Corpus Results (4/7 passing)

| Document | gen.pdf | diff_health | match_rate | image_sim | Notes |
|----------|---------|-------------|------------|-----------|-------|
| ACM_SUBMISSION_TEMPLATE | ✅ 128KB | VALID | 0.36 | 0.92 | |
| Custom_Resume | ✅ 92KB | VALID | 0.60 | 0.88 | |
| IEEE_TEMPLATE | ✅ 109KB | VALID | 0.23 | 0.90 | Was INVALID, now passes threshold |
| Resume_2 | ✅ 28KB | VALID | 0.53 | 0.92 | **Fixed** - XeLaTeX handles Unicode |
| Microsoft_Resume1 | ❌ 0B | - | - | - | tabularx error |
| Microsoft_Resume_2 | ❌ 0B | - | - | - | tabularx error |
| Resume_1 | ❌ 0B | - | - | - | tabularx error |

---

## What Was Implemented This Session

### 1. LaTeX Engine Switch (run_corpus.py)
- **XeLaTeX is now default** - handles Unicode natively
- **pdfLaTeX fallback** with Unicode sanitizer if XeLaTeX unavailable
- **0-byte PDF = hard failure** - forces `latex_exit_code = 1` if output < 5KB

### 2. Match Coverage Improvements (layout_oracle.py)
- **Text normalization**: NFKC, soft hyphen removal, whitespace collapse
- **Duplicate handling**: `text -> list[blocks]` instead of `text -> block`
- **Reading order matching**: sort by (page, y, x) before pairing

### 3. Threshold Fix (compare_layout.py)
- Changed `< 0.2` to `<= 0.2` for `INVALID_MATCH_COVERAGE`
- Logs unrounded `match_rate` for clarity

---

## What Still Needs Fixing

### Priority 1: Fix tabularx failures (3 docs)

The remaining 3 failures all have `tabularx` errors. Check:

```bash
grep -E "^!|error:" tests/corpus/Microsoft_Resume_2/latex.log | head -10
grep -E "^!|error:" tests/corpus/Resume_1/latex.log | head -10
```

Typical causes:
- Missing `{}` around column spec
- Unescaped `& % _ #` in cells
- Mismatched row `\\`
- Wrong width (`\linewidth` vs `\textwidth`)

**Location to fix:** `src/wordtolatex/generator.py` - table generation code

### Priority 2: Add compile-based unit test

Create a test that:
1. Generates minimal table LaTeX
2. Compiles with XeLaTeX
3. Asserts non-zero PDF size

### Priority 3: Further match coverage improvements

Current rates (0.23-0.60) could be higher. Consider:
- Fuzzy matching (Levenshtein)
- Ignore punctuation differences
- Handle line-break artifacts

### Priority 4: Region-masked image diff

Current `image_diff` is whole-page. Next step:
- Crop by oracle regions
- Score per region type (tables, titles, lists)
- Identify localized failures

---

## Key Files Modified

| File | Changes |
|------|---------|
| `scripts/run_corpus.py` | XeLaTeX default, Unicode sanitizer, 0-byte hard fail |
| `scripts/compare_layout.py` | Threshold fix (`<= 0.2`), unrounded match_rate logging |
| `src/wordtolatex/layout_oracle.py` | Text normalization, duplicate handling, reading order |
| `scripts/image_diff.py` | NEW - SSIM/MAD visual similarity |

---

## Commands to Continue

```bash
# Clean and re-run corpus
cd /Users/sujendragharat/Library/CloudStorage/GoogleDrive-sgharat298@gmail.com/My\ Drive/MacExternalCloud/Documents/Projects/wordTOlatex
rm -rf tests/corpus/*
uv run python scripts/run_corpus.py

# Validate only
uv run python scripts/run_corpus.py --validate-only

# Run tests
uv run pytest tests/test_golden_layout.py -v

# Check specific LaTeX error
cat tests/corpus/Microsoft_Resume_2/latex.log | grep -A5 "^!"
```

---

## Success Criteria for Next Session

1. **7/7 gen.pdf** - All docs compile successfully
2. **Match rates > 0.5** for all docs
3. **No tabularx errors** - Fix table generation code
4. **Add compile unit test** - Prevent table regressions
