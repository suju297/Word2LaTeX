## What I would change (small but important)

### 1) Don’t call `ltablex` a “reliability” win without clarifying tradeoff

Right now `Microsoft_Resume_2` shows  **+5 pages** , which is a **fidelity regression** even if it avoids empty pages. You already labeled it “Fidelity Issue,” good — but add one sentence:

* “We prefer overflow-safe output over blank pages in v1; Phase 3 will reduce page inflation.”

That prevents someone from saying “your converter explodes page count.”

### 2) Your “No Empty Pages” check is good, but phrase it as a heuristic

Mean intensity thresholds can false-positive on mostly-white pages (cover pages, spacing-heavy docs). Say:

* “No empty pages (heuristic: page render intensity + text block count).”

### 3) Add one KPI for Phase 3 besides SSIM

SSIM alone can look good while small text shifts are unacceptable. Add:

* “Page count delta within ±10% for non-table-heavy docs”

  or
* “Median text baseline shift < 10pt for Titles/Lists”

Keep it simple.

### 4) Clarify corpus composition in one line

You mention categories. Add the actual count split:

* “Corpus: 4 resumes, 2 academic templates, 1 custom resume” (or whatever is accurate).

This makes the claim concrete.

---

## A cleaned-up version (ready to paste)

Use this as your final “public” report text:

**Word to LaTeX Converter — v1 Status: Reliability Green**

Date: 2025-12-30 | Status: ✅ 7/7 corpus documents passing

**Summary**

v1 reliably converts DOCX into compilable, non-empty LaTeX/PDF with full audit traces. Phase 3 focuses on region-level visual fidelity.

**Reliability Milestone Achieved**

The converter now processes all corpus documents (resumes, templates, academic-style papers) without crashing or producing empty output.

### Key Achievements

**1) Parser robustness**

* Extracts hidden text from `w:sdt` (Structured Document Tags) and `w:drawing` textboxes (layout-heavy templates).
* Preserves intra-run line breaks via `w:br`.
* Unit tests added: `tests/test_parser_sdt.py`, `tests/test_parser_drawing.py`.

**2) Generator stability**

* Fixed `tabularx` logical column counting for `\multicolumn` cells (resolved “Extra alignment tab”).
* Complex table fallback now emits output with warnings instead of producing nothing.
* Large tables: uses `ltablex` in overflow-prone cases to prevent blank-page output (may increase page count; treated as a fidelity issue to fix in Phase 3).

**3) Validation hardening (prevents false positives)**

* PDF size > 5KB
* Content check: `generated_blocks >= 20`
* No empty pages (heuristic: page render intensity + no-text detection)
* Full JSON audit trace per run

### Corpus Results

| Document           | Status | PDF Size | Blocks | Page Count Diff | Notes                                         |
| ------------------ | ------ | -------- | ------ | --------------- | --------------------------------------------- |
| Microsoft_Resume1  | ✅     | 19KB     | 48     | Match           | Fixed via SDT extraction                      |
| Resume_1           | ✅     | 15KB     | High   | Match           | Fixed `tabularx`                            |
| Microsoft_Resume_2 | ✅     | 35KB     | High   | +5              | Overflow-safe via `ltablex`(fidelity issue) |
| ACM_Submission     | ✅     | >5KB     | 611    | -2              | Layout variance                               |
| Custom_Resume      | ✅     | >5KB     | High   | -1              | Layout variance                               |
| IEEE_Template      | ✅     | >5KB     | High   | +1              | Layout variance                               |
| Resume_2           | ✅     | >5KB     | High   | Match           | —                                            |

---

## Phase 3: Fidelity Priorities

**Priority A: Region-masked image diff (main driver)**

* Goal: tight feedback loop by region type (tables, figures, titles).
* Metric targets: Avg region SSIM > 0.85 (titles/lists), > 0.75 (tables/figures).
* Artifact: “Top 5 worst regions” report + overlays.

**Priority B: Text matching improvements (make drift meaningful)**

* Unicode normalization (NFKC), remove soft hyphens, normalize ligatures.
* Reading-order matching (page, y, x) to handle duplicates and multi-column drift.

**Priority C: Layout normalization (reduce systematic mismatch)**

* Align page size + margins (Letter vs A4).
* Map Word line spacing to LaTeX lengths.
* Standardize on XeLaTeX + fontspec for font fidelity.

---

## One direct piece of advice

When you share this with others, lead with: **“reliability green, fidelity next”** and show 1 screenshot of overlays + 1 sample audit trace snippet. That proves it’s real.




## Two important implementation choices

### 1) Convert normalized bbox →  **page points** , then render/crop

Instead of converting to “pixels using rect.width/height” directly, do:

* `bbox_norm` → **bbox in PDF points** using `page.rect.width/height`
* then either:
  * render the full page at `dpi` and crop pixels, **or**
  * render the region directly with `page.get_pixmap(clip=rect, dpi=dpi)`

Both are valid. The full-page render is faster if you have many regions on a page. The `clip=` approach is simpler and avoids pixel math bugs.

### 2) Blank-region detection should use **mean + variance**

Mean alone can misclassify lightly populated regions as “blank”. Use both:

* `mean > 250` **and** `std < 2` → blank (tune later)
* “blank both” → skip
* “blank one side” → keep and score (this is a meaningful failure)

## Extra guardrails worth adding

* **Clamp bbox** to image bounds after conversion (avoid negative/out-of-range crops).
* **Status coverage** should be reported per label (titles/lists/tables), not just overall.
* Add `run_id` to outputs (keeps reports comparable across runs).
* SSIM `win_size` handling: if region is small, fall back to MAD rather than trying to force SSIM.

---

# Suggested output files

Per corpus doc:

* `region_stats.json` (per-region records + per-label summary + coverage)
* `worst_regions/` (top N crops: `ref.png`, `gen.png`, `diff.png`)

Global:

* `tests/corpus/region_diff_global.json` (top 20 across corpus + global averages)

---

## Code skeleton for `scripts/region_diff.py`

This is a complete “drop-in” script structure you can adapt. It assumes `layout_hints.json` contains a top-level `"regions": [...]` with items like:

`{"page": 0, "label": "title", "bbox_norm": [x0,y0,x1,y1], "source": "ml", "confidence": 0.93}`

<pre class="overflow-visible! px-0!" data-start="1993" data-end="15586"><div class="contain-inline-size rounded-2xl corner-superellipse/1.1 relative bg-token-sidebar-surface-primary"><div class="sticky top-[calc(--spacing(9)+var(--header-height))] @w-xl/main:top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span>#!/usr/bin/env python3</span><span>
</span><span>from</span><span> __future__ </span><span>import</span><span> annotations

</span><span>import</span><span> argparse
</span><span>import</span><span> json
</span><span>import</span><span> math
</span><span>from</span><span> pathlib </span><span>import</span><span> Path
</span><span>from</span><span> dataclasses </span><span>import</span><span> dataclass
</span><span>from</span><span> typing </span><span>import</span><span></span><span>Any</span><span>, </span><span>Dict</span><span>, </span><span>List</span><span>, </span><span>Optional</span><span>, </span><span>Tuple</span><span>

</span><span>import</span><span> fitz  </span><span># pymupdf</span><span>
</span><span>import</span><span> numpy </span><span>as</span><span> np
</span><span>from</span><span> PIL </span><span>import</span><span> Image

</span><span>try</span><span>:
    </span><span>from</span><span> skimage.metrics </span><span>import</span><span> structural_similarity </span><span>as</span><span> ssim
    HAS_SSIM = </span><span>True</span><span>
</span><span>except</span><span> Exception:
    HAS_SSIM = </span><span>False</span><span>


</span><span># ---------- Helpers ----------</span><span>

</span><span>def</span><span></span><span>load_json</span><span>(</span><span>path: Path</span><span>) -> </span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>Any</span><span>]:
    </span><span>return</span><span> json.loads(path.read_text(encoding=</span><span>"utf-8"</span><span>))

</span><span>def</span><span></span><span>save_json</span><span>(</span><span>path: Path, data: Any</span><span>) -> </span><span>None</span><span>:
    path.write_text(json.dumps(data, indent=</span><span>2</span><span>, ensure_ascii=</span><span>False</span><span>), encoding=</span><span>"utf-8"</span><span>)

</span><span>def</span><span></span><span>ensure_dir</span><span>(</span><span>p: Path</span><span>) -> </span><span>None</span><span>:
    p.mkdir(parents=</span><span>True</span><span>, exist_ok=</span><span>True</span><span>)

</span><span>def</span><span></span><span>to_grayscale_uint8</span><span>(</span><span>arr_rgb: np.ndarray</span><span>) -> np.ndarray:
    </span><span># arr_rgb: HxWx3 uint8</span><span>
    </span><span># luminance weights reduce noise vs simple mean</span><span>
    r = arr_rgb[..., </span><span>0</span><span>].astype(np.float32)
    g = arr_rgb[..., </span><span>1</span><span>].astype(np.float32)
    b = arr_rgb[..., </span><span>2</span><span>].astype(np.float32)
    gray = </span><span>0.299</span><span> * r + </span><span>0.587</span><span> * g + </span><span>0.114</span><span> * b
    </span><span>return</span><span> np.clip(gray, </span><span>0</span><span>, </span><span>255</span><span>).astype(np.uint8)

</span><span>def</span><span></span><span>is_blank</span><span>(</span><span>gray: np.ndarray, mean_thresh: float</span><span> = </span><span>250.0</span><span>, std_thresh: </span><span>float</span><span> = </span><span>2.0</span><span>) -> </span><span>bool</span><span>:
    </span><span>if</span><span> gray.size == </span><span>0</span><span>:
        </span><span>return</span><span></span><span>True</span><span>
    m = </span><span>float</span><span>(gray.mean())
    s = </span><span>float</span><span>(gray.std())
    </span><span>return</span><span> (m >= mean_thresh) </span><span>and</span><span> (s <= std_thresh)

</span><span>def</span><span></span><span>dynamic_win_size</span><span>(</span><span>h: int</span><span>, w: </span><span>int</span><span>, preferred: </span><span>int</span><span> = </span><span>11</span><span>) -> </span><span>Optional</span><span>[</span><span>int</span><span>]:
    m = </span><span>min</span><span>(h, w)
    </span><span>if</span><span> m < </span><span>7</span><span>:
        </span><span>return</span><span></span><span>None</span><span>
    win = </span><span>min</span><span>(preferred, m)
    </span><span># SSIM needs odd win_size</span><span>
    </span><span>if</span><span> win % </span><span>2</span><span> == </span><span>0</span><span>:
        win -= </span><span>1</span><span>
    </span><span>if</span><span> win < </span><span>7</span><span>:
        </span><span>return</span><span></span><span>None</span><span>
    </span><span>return</span><span> win

</span><span>def</span><span></span><span>mad_similarity</span><span>(</span><span>a: np.ndarray, b: np.ndarray</span><span>) -> </span><span>float</span><span>:
    </span><span># 1 - normalized mean absolute difference</span><span>
    diff = np.</span><span>abs</span><span>(a.astype(np.float32) - b.astype(np.float32))
    </span><span>return</span><span></span><span>float</span><span>(</span><span>max</span><span>(</span><span>0.0</span><span>, </span><span>1.0</span><span> - (diff.mean() / </span><span>255.0</span><span>)))

</span><span>def</span><span></span><span>absdiff_heatmap</span><span>(</span><span>a: np.ndarray, b: np.ndarray</span><span>) -> np.ndarray:
    </span><span># returns grayscale diff image uint8</span><span>
    diff = np.</span><span>abs</span><span>(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    </span><span>return</span><span> diff

</span><span>def</span><span></span><span>bbox_norm_to_points</span><span>(</span><span>page: fitz.Page, bbox_norm: List</span><span>[</span><span>float</span><span>]) -> fitz.Rect:
    x0, y0, x1, y1 = bbox_norm
    w = page.rect.width
    h = page.rect.height
    </span><span># convert normalized to points</span><span>
    </span><span>return</span><span> fitz.Rect(x0 * w, y0 * h, x1 * w, y1 * h)

</span><span>def</span><span></span><span>clamp_rect_to_page</span><span>(</span><span>page: fitz.Page, rect: fitz.Rect</span><span>) -> fitz.Rect:
    r = fitz.Rect(rect)
    </span><span># clamp to page rect</span><span>
    pr = page.rect
    r.x0 = </span><span>max</span><span>(pr.x0, </span><span>min</span><span>(r.x0, pr.x1))
    r.x1 = </span><span>max</span><span>(pr.x0, </span><span>min</span><span>(r.x1, pr.x1))
    r.y0 = </span><span>max</span><span>(pr.y0, </span><span>min</span><span>(r.y0, pr.y1))
    r.y1 = </span><span>max</span><span>(pr.y0, </span><span>min</span><span>(r.y1, pr.y1))
    </span><span># ensure proper ordering</span><span>
    </span><span>if</span><span> r.x1 < r.x0:
        r.x0, r.x1 = r.x1, r.x0
    </span><span>if</span><span> r.y1 < r.y0:
        r.y0, r.y1 = r.y1, r.y0
    </span><span>return</span><span> r

</span><span>def</span><span></span><span>render_clip_gray</span><span>(</span><span>page: fitz.Page, rect: fitz.Rect, dpi: int</span><span>) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, clip=rect, alpha=</span><span>False</span><span>)
    </span><span># pix.n should be 3 (RGB)</span><span>
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    </span><span>if</span><span> arr.shape[</span><span>2</span><span>] >= </span><span>3</span><span>:
        arr = arr[:, :, :</span><span>3</span><span>]
    </span><span>return</span><span> to_grayscale_uint8(arr)

</span><span>def</span><span></span><span>save_gray_png</span><span>(</span><span>path: Path, gray: np.ndarray</span><span>) -> </span><span>None</span><span>:
    Image.fromarray(gray).save(path)

</span><span># ---------- Core ----------</span><span>

</span><span>def</span><span></span><span>compute_region_score</span><span>(</span><span>ref_gray: np.ndarray, gen_gray: np.ndarray</span><span>) -> </span><span>Tuple</span><span>[</span><span>str</span><span>, </span><span>float</span><span>, </span><span>str</span><span>]:
    """
    Returns (method_used, score, status_detail)
    """
    </span><span># Ensure same shape: strict baseline uses same coords, but images can differ due to page boxes</span><span>
    </span><span># Use center-crop to common size (no resizing warps geometry)</span><span>
    h = </span><span>min</span><span>(ref_gray.shape[</span><span>0</span><span>], gen_gray.shape[</span><span>0</span><span>])
    w = </span><span>min</span><span>(ref_gray.shape[</span><span>1</span><span>], gen_gray.shape[</span><span>1</span><span>])
    ref_c = ref_gray[:h, :w]
    gen_c = gen_gray[:h, :w]

    </span><span>if</span><span> HAS_SSIM:
        win = dynamic_win_size(h, w, preferred=</span><span>11</span><span>)
        </span><span>if</span><span> win </span><span>is</span><span></span><span>not</span><span></span><span>None</span><span>:
            score = </span><span>float</span><span>(ssim(ref_c, gen_c, data_range=</span><span>255</span><span>, win_size=win))
            </span><span>return</span><span> (</span><span>"ssim"</span><span>, score, </span><span>f"win_size={win}</span><span>")
        </span><span># fallback if too small for SSIM</span><span>
    score = mad_similarity(ref_c, gen_c)
    </span><span>return</span><span> (</span><span>"mad"</span><span>, score, </span><span>"fallback"</span><span>)

</span><span>def</span><span></span><span>process_doc</span><span>(</span><span>case_dir: Path, dpi: int</span><span>, min_region_px: </span><span>int</span><span>) -> </span><span>Tuple</span><span>[</span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>Any</span><span>], </span><span>List</span><span>[</span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>Any</span><span>]]]:
    hints_path = case_dir / </span><span>"layout_hints.json"</span><span>
    ref_pdf = case_dir / </span><span>"ref.pdf"</span><span>
    gen_pdf = case_dir / </span><span>"gen.pdf"</span><span>

    out_stats_path = case_dir / </span><span>"region_stats.json"</span><span>
    worst_dir = case_dir / </span><span>"worst_regions"</span><span>
    ensure_dir(worst_dir)

    </span><span>if</span><span></span><span>not</span><span> (hints_path.exists() </span><span>and</span><span> ref_pdf.exists() </span><span>and</span><span> gen_pdf.exists()):
        </span><span>return</span><span> ({</span><span>"doc_id"</span><span>: case_dir.name, </span><span>"error"</span><span>: </span><span>"missing_inputs"</span><span>}, [])

    hints = load_json(hints_path)
    regions = hints.get(</span><span>"regions"</span><span>, [])
    doc_id = hints.get(</span><span>"doc_id"</span><span>, case_dir.name)

    ref = fitz.</span><span>open</span><span>(ref_pdf)
    gen = fitz.</span><span>open</span><span>(gen_pdf)

    per_region: </span><span>List</span><span>[</span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>Any</span><span>]] = []
    scored_regions: </span><span>List</span><span>[</span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>Any</span><span>]] = []

    ref_pages = </span><span>len</span><span>(ref)
    gen_pages = </span><span>len</span><span>(gen)
    page_count_match = (ref_pages == gen_pages)

    </span><span># optional: log page sizes</span><span>
    page_boxes = []
    </span><span>for</span><span> i </span><span>in</span><span></span><span>range</span><span>(</span><span>min</span><span>(ref_pages, gen_pages)):
        rp = ref[i].rect
        gp = gen[i].rect
        page_boxes.append({
            </span><span>"page"</span><span>: i,
            </span><span>"ref_pts"</span><span>: [rp.width, rp.height],
            </span><span>"gen_pts"</span><span>: [gp.width, gp.height],
        })

    counts = {
        </span><span>"total"</span><span>: </span><span>0</span><span>,
        </span><span>"scored"</span><span>: </span><span>0</span><span>,
        </span><span>"missing_page"</span><span>: </span><span>0</span><span>,
        </span><span>"tiny"</span><span>: </span><span>0</span><span>,
        </span><span>"blank_both"</span><span>: </span><span>0</span><span>,
    }

    by_label: </span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>List</span><span>[</span><span>float</span><span>]] = {}

    </span><span>for</span><span> idx, r </span><span>in</span><span></span><span>enumerate</span><span>(regions):
        counts[</span><span>"total"</span><span>] += </span><span>1</span><span>
        page = </span><span>int</span><span>(r.get(</span><span>"page"</span><span>, </span><span>0</span><span>))
        label = </span><span>str</span><span>(r.get(</span><span>"label"</span><span>, </span><span>"unknown"</span><span>))
        source = </span><span>str</span><span>(r.get(</span><span>"source"</span><span>, </span><span>"unknown"</span><span>))
        conf = r.get(</span><span>"confidence"</span><span>, </span><span>None</span><span>)

        bbox_norm = r.get(</span><span>"bbox_norm"</span><span>) </span><span>or</span><span> r.get(</span><span>"bbox"</span><span>)  </span><span># be forgiving</span><span>
        region_id = r.get(</span><span>"id"</span><span>) </span><span>or</span><span></span><span>f"p{page}</span><span>_</span><span>{label}</span><span>_</span><span>{idx:03d}</span><span>"

        record = {
            </span><span>"region_id"</span><span>: region_id,
            </span><span>"doc_id"</span><span>: doc_id,
            </span><span>"page"</span><span>: page,
            </span><span>"label"</span><span>: label,
            </span><span>"source"</span><span>: source,
            </span><span>"confidence"</span><span>: conf,
            </span><span>"bbox_norm"</span><span>: bbox_norm,
            </span><span>"status"</span><span>: </span><span>None</span><span>,
            </span><span>"score"</span><span>: </span><span>None</span><span>,
            </span><span>"method"</span><span>: </span><span>None</span><span>,
            </span><span>"detail"</span><span>: </span><span>None</span><span>,
            </span><span>"blank_ref"</span><span>: </span><span>None</span><span>,
            </span><span>"blank_gen"</span><span>: </span><span>None</span><span>,
        }

        </span><span>if</span><span> bbox_norm </span><span>is</span><span></span><span>None</span><span></span><span>or</span><span></span><span>len</span><span>(bbox_norm) != </span><span>4</span><span>:
            record[</span><span>"status"</span><span>] = </span><span>"INVALID_BBOX"</span><span>
            per_region.append(record)
            </span><span>continue</span><span>

        </span><span>if</span><span> page >= ref_pages </span><span>or</span><span> page >= gen_pages:
            record[</span><span>"status"</span><span>] = </span><span>"MISSING_PAGE"</span><span>
            counts[</span><span>"missing_page"</span><span>] += </span><span>1</span><span>
            per_region.append(record)
            </span><span>continue</span><span>

        ref_page = ref[page]
        gen_page = gen[page]

        </span><span># normalized -> points rect; clamp</span><span>
        rect_ref = clamp_rect_to_page(ref_page, bbox_norm_to_points(ref_page, bbox_norm))
        rect_gen = clamp_rect_to_page(gen_page, bbox_norm_to_points(gen_page, bbox_norm))

        </span><span># render clipped crops</span><span>
        ref_gray = render_clip_gray(ref_page, rect_ref, dpi=dpi)
        gen_gray = render_clip_gray(gen_page, rect_gen, dpi=dpi)

        </span><span># tiny region skip</span><span>
        </span><span>if</span><span> ref_gray.shape[</span><span>0</span><span>] < min_region_px </span><span>or</span><span> ref_gray.shape[</span><span>1</span><span>] < min_region_px:
            record[</span><span>"status"</span><span>] = </span><span>"SKIPPED_TINY_REGION"</span><span>
            counts[</span><span>"tiny"</span><span>] += </span><span>1</span><span>
            per_region.append(record)
            </span><span>continue</span><span>

        blank_ref = is_blank(ref_gray)
        blank_gen = is_blank(gen_gray)
        record[</span><span>"blank_ref"</span><span>] = blank_ref
        record[</span><span>"blank_gen"</span><span>] = blank_gen

        </span><span>if</span><span> blank_ref </span><span>and</span><span> blank_gen:
            record[</span><span>"status"</span><span>] = </span><span>"SKIPPED_BLANK_REGION"</span><span>
            counts[</span><span>"blank_both"</span><span>] += </span><span>1</span><span>
            per_region.append(record)
            </span><span>continue</span><span>

        method, score, detail = compute_region_score(ref_gray, gen_gray)
        record[</span><span>"status"</span><span>] = </span><span>"SCORED"</span><span>
        record[</span><span>"score"</span><span>] = score
        record[</span><span>"method"</span><span>] = method
        record[</span><span>"detail"</span><span>] = detail

        counts[</span><span>"scored"</span><span>] += </span><span>1</span><span>
        per_region.append(record)
        scored_regions.append(record)

        by_label.setdefault(label, []).append(score)

    </span><span># per-label summary stats</span><span>
    </span><span>def</span><span></span><span>summarize</span><span>(</span><span>scores: List</span><span>[</span><span>float</span><span>]) -> </span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>Any</span><span>]:
        arr = np.array(scores, dtype=np.float32)
        </span><span>return</span><span> {
            </span><span>"count_scored"</span><span>: </span><span>int</span><span>(arr.size),
            </span><span>"avg"</span><span>: </span><span>float</span><span>(arr.mean()) </span><span>if</span><span> arr.size </span><span>else</span><span></span><span>None</span><span>,
            </span><span>"p50"</span><span>: </span><span>float</span><span>(np.percentile(arr, </span><span>50</span><span>)) </span><span>if</span><span> arr.size </span><span>else</span><span></span><span>None</span><span>,
            </span><span>"p10"</span><span>: </span><span>float</span><span>(np.percentile(arr, </span><span>10</span><span>)) </span><span>if</span><span> arr.size </span><span>else</span><span></span><span>None</span><span>,
            </span><span>"min"</span><span>: </span><span>float</span><span>(arr.</span><span>min</span><span>()) </span><span>if</span><span> arr.size </span><span>else</span><span></span><span>None</span><span>,
        }

    by_label_summary = {k: summarize(v) </span><span>for</span><span> k, v </span><span>in</span><span> by_label.items()}

    doc_stats = {
        </span><span>"doc_id"</span><span>: doc_id,
        </span><span>"case_dir"</span><span>: case_dir.name,
        </span><span>"ref_pages"</span><span>: ref_pages,
        </span><span>"gen_pages"</span><span>: gen_pages,
        </span><span>"page_count_match"</span><span>: page_count_match,
        </span><span>"page_boxes"</span><span>: page_boxes,
        </span><span>"coverage"</span><span>: counts,
        </span><span>"by_label"</span><span>: by_label_summary,
        </span><span>"regions"</span><span>: per_region,
    }

    save_json(out_stats_path, doc_stats)

    ref.close()
    gen.close()

    </span><span>return</span><span> (doc_stats, scored_regions)

</span><span>def</span><span></span><span>save_worst_regions</span><span>(</span><span>worst: List</span><span>[</span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>Any</span><span>]], corpus_root: Path, dpi: </span><span>int</span><span>, out_dir: Path) -> </span><span>None</span><span>:
    ensure_dir(out_dir)
    </span><span># Create images for worst regions (ref/gen/diff)</span><span>
    </span><span>for</span><span> i, rec </span><span>in</span><span></span><span>enumerate</span><span>(worst):
        case_dir = corpus_root / rec[</span><span>"case_dir"</span><span>]
        ref_pdf = case_dir / </span><span>"ref.pdf"</span><span>
        gen_pdf = case_dir / </span><span>"gen.pdf"</span><span>
        hints_path = case_dir / </span><span>"layout_hints.json"</span><span>
        </span><span>if</span><span></span><span>not</span><span> (ref_pdf.exists() </span><span>and</span><span> gen_pdf.exists() </span><span>and</span><span> hints_path.exists()):
            </span><span>continue</span><span>

        hints = load_json(hints_path)
        </span><span># locate region by id</span><span>
        regions = hints.get(</span><span>"regions"</span><span>, [])
        region_id = rec[</span><span>"region_id"</span><span>]
        target = </span><span>None</span><span>
        </span><span>for</span><span> r </span><span>in</span><span> regions:
            rid = r.get(</span><span>"id"</span><span>) </span><span>or</span><span></span><span>None</span><span>
            </span><span>if</span><span> rid </span><span>is</span><span></span><span>None</span><span>:
                </span><span># fallback match by computed id format is hard; just skip if no ids exist</span><span>
                </span><span>continue</span><span>
            </span><span>if</span><span> rid == region_id:
                target = r
                </span><span>break</span><span>
        </span><span>if</span><span> target </span><span>is</span><span></span><span>None</span><span>:
            </span><span>continue</span><span>

        page = </span><span>int</span><span>(target.get(</span><span>"page"</span><span>, </span><span>0</span><span>))
        bbox_norm = target.get(</span><span>"bbox_norm"</span><span>) </span><span>or</span><span> target.get(</span><span>"bbox"</span><span>)
        </span><span>if</span><span> bbox_norm </span><span>is</span><span></span><span>None</span><span>:
            </span><span>continue</span><span>

        ref = fitz.</span><span>open</span><span>(ref_pdf)
        gen = fitz.</span><span>open</span><span>(gen_pdf)
        </span><span>if</span><span> page >= </span><span>len</span><span>(ref) </span><span>or</span><span> page >= </span><span>len</span><span>(gen):
            ref.close()
            gen.close()
            </span><span>continue</span><span>

        ref_page = ref[page]
        gen_page = gen[page]
        rect_ref = clamp_rect_to_page(ref_page, bbox_norm_to_points(ref_page, bbox_norm))
        rect_gen = clamp_rect_to_page(gen_page, bbox_norm_to_points(gen_page, bbox_norm))

        ref_gray = render_clip_gray(ref_page, rect_ref, dpi=dpi)
        gen_gray = render_clip_gray(gen_page, rect_gen, dpi=dpi)

        </span><span># center-crop to common size</span><span>
        h = </span><span>min</span><span>(ref_gray.shape[</span><span>0</span><span>], gen_gray.shape[</span><span>0</span><span>])
        w = </span><span>min</span><span>(ref_gray.shape[</span><span>1</span><span>], gen_gray.shape[</span><span>1</span><span>])
        ref_c = ref_gray[:h, :w]
        gen_c = gen_gray[:h, :w]
        diff = absdiff_heatmap(ref_c, gen_c)

        prefix = </span><span>f"{i:02d}</span><span>_</span><span>{rec['case_dir'</span><span>]}_</span><span>{rec['label'</span><span>]}_p</span><span>{page}</span><span>"
        save_gray_png(out_dir / </span><span>f"{prefix}</span><span>_ref.png", ref_c)
        save_gray_png(out_dir / </span><span>f"{prefix}</span><span>_gen.png", gen_c)
        save_gray_png(out_dir / </span><span>f"{prefix}</span><span>_diff.png", diff)

        ref.close()
        gen.close()


</span><span>def</span><span></span><span>main</span><span>() -> </span><span>None</span><span>:
    ap = argparse.ArgumentParser()
    ap.add_argument(</span><span>"--corpus-root"</span><span>, </span><span>type</span><span>=</span><span>str</span><span>, default=</span><span>"tests/corpus"</span><span>)
    ap.add_argument(</span><span>"--dpi"</span><span>, </span><span>type</span><span>=</span><span>int</span><span>, default=</span><span>150</span><span>)
    ap.add_argument(</span><span>"--min-region-px"</span><span>, </span><span>type</span><span>=</span><span>int</span><span>, default=</span><span>32</span><span>)
    ap.add_argument(</span><span>"--max-worst"</span><span>, </span><span>type</span><span>=</span><span>int</span><span>, default=</span><span>20</span><span>)
    args = ap.parse_args()

    corpus_root = Path(args.corpus_root)
    </span><span>if</span><span></span><span>not</span><span> corpus_root.exists():
        </span><span>raise</span><span> SystemExit(</span><span>f"Corpus root not found: {corpus_root}</span><span>")

    all_scored: </span><span>List</span><span>[</span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>Any</span><span>]] = []
    global_docs: </span><span>List</span><span>[</span><span>Dict</span><span>[</span><span>str</span><span>, </span><span>Any</span><span>]] = []

    </span><span>for</span><span> case_dir </span><span>in</span><span></span><span>sorted</span><span>([p </span><span>for</span><span> p </span><span>in</span><span> corpus_root.iterdir() </span><span>if</span><span> p.is_dir()]):
        stats, scored = process_doc(case_dir, dpi=args.dpi, min_region_px=args.min_region_px)
        </span><span>if</span><span></span><span>"error"</span><span></span><span>in</span><span> stats:
            </span><span>continue</span><span>
        </span><span># attach case_dir for global worst tracking</span><span>
        </span><span>for</span><span> r </span><span>in</span><span> scored:
            r2 = </span><span>dict</span><span>(r)
            r2[</span><span>"case_dir"</span><span>] = stats[</span><span>"case_dir"</span><span>]
            all_scored.append(r2)
        global_docs.append(stats)

    </span><span># global worst (lowest scores)</span><span>
    scored_only = [r </span><span>for</span><span> r </span><span>in</span><span> all_scored </span><span>if</span><span> r.get(</span><span>"status"</span><span>) == </span><span>"SCORED"</span><span></span><span>and</span><span> r.get(</span><span>"score"</span><span>) </span><span>is</span><span></span><span>not</span><span></span><span>None</span><span>]
    scored_only.sort(key=</span><span>lambda</span><span> x: </span><span>float</span><span>(x[</span><span>"score"</span><span>]))
    worst = scored_only[: args.max_worst]

    global_summary = {
        </span><span>"corpus_root"</span><span>: </span><span>str</span><span>(corpus_root),
        </span><span>"docs_processed"</span><span>: </span><span>len</span><span>(global_docs),
        </span><span>"regions_scored"</span><span>: </span><span>len</span><span>(scored_only),
        </span><span>"worst_regions"</span><span>: [
            {
                </span><span>"case_dir"</span><span>: w[</span><span>"case_dir"</span><span>],
                </span><span>"doc_id"</span><span>: w.get(</span><span>"doc_id"</span><span>),
                </span><span>"region_id"</span><span>: w[</span><span>"region_id"</span><span>],
                </span><span>"label"</span><span>: w[</span><span>"label"</span><span>],
                </span><span>"page"</span><span>: w[</span><span>"page"</span><span>],
                </span><span>"score"</span><span>: w[</span><span>"score"</span><span>],
                </span><span>"method"</span><span>: w.get(</span><span>"method"</span><span>),
            }
            </span><span>for</span><span> w </span><span>in</span><span> worst
        ],
    }

    save_json(corpus_root / </span><span>"region_diff_global.json"</span><span>, global_summary)
    save_worst_regions(worst, corpus_root=corpus_root, dpi=args.dpi, out_dir=corpus_root / </span><span>"worst_regions"</span><span>)

    </span><span># console report</span><span>
    </span><span>if</span><span> worst:
        </span><span>print</span><span>(</span><span>"\nTop worst regions (strict baseline):"</span><span>)
        </span><span>for</span><span> w </span><span>in</span><span> global_summary[</span><span>"worst_regions"</span><span>]:
            </span><span>print</span><span>(</span><span>f"- {w['case_dir'</span><span>]} p</span><span>{w['page'</span><span>]} </span><span>{w['label'</span><span>]} score=</span><span>{w['score'</span><span>]:</span><span>.3</span><span>f} (</span><span>{w['method'</span><span>]})")
    </span><span>print</span><span>(</span><span>f"\nWrote: {corpus_root / 'region_diff_global.json'</span><span>}")
    </span><span>print</span><span>(</span><span>f"Worst images: {corpus_root / 'worst_regions'</span><span>}")

</span><span>if</span><span> __name__ == </span><span>"__main__"</span><span>:
    main()
</span></span></code></div></div></pre>

## One key note about your “coverage > 0.7 for titles/figures”

Make that check per label like:

* `coverage_titles = scored_titles / total_titles`
* and exclude `MISSING_PAGE` from the denominator if you want a fairness variant

For strict baseline, I’d keep missing pages in the denominator (it correctly punishes reflow), but report both:

* `coverage_strict`
* `coverage_excluding_missing`
