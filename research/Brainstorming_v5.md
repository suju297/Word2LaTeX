# Phase 3 Exit: Corpus Gating Policy & CI Artifacts

## Goal

Lock the evaluation gates into a repeatable CI-enforceable contract.

## Proposed Changes

### 1. Create Gating Config

#### [NEW]

corpus_gates.yaml

Define per-label thresholds:

<pre><div node="[object Object]" class="relative whitespace-pre-wrap word-break-all p-3 my-2 rounded-sm bg-list-hover-subtle"><div class="w-full h-full text-xs cursor-text"><div class="code-block"><div class="code-line" data-line-number="1" data-line-start="1" data-line-end="1"><div class="line-content"><span class="mtk5">gates</span><span class="mtk1">:</span></div></div><div class="code-line" data-line-number="2" data-line-start="2" data-line-end="2"><div class="line-content"><span class="mtk1"></span><span class="mtk5">coverage</span><span class="mtk1">:</span></div></div><div class="code-line" data-line-number="3" data-line-start="3" data-line-end="3"><div class="line-content"><span class="mtk1"></span><span class="mtk5">min_effective</span><span class="mtk1">: </span><span class="mtk6">0.70</span></div></div><div class="code-line" data-line-number="4" data-line-start="4" data-line-end="4"><div class="line-content"><span class="mtk1"></span><span class="mtk5">structure</span><span class="mtk1">:</span></div></div><div class="code-line" data-line-number="5" data-line-start="5" data-line-end="5"><div class="line-content"><span class="mtk1"></span><span class="mtk5">hit_boundary_max</span><span class="mtk1">: </span><span class="mtk6">0</span></div></div><div class="code-line" data-line-number="6" data-line-start="6" data-line-end="6"><div class="line-content"><span class="mtk1"></span><span class="mtk5">blank_gen_max</span><span class="mtk1">: </span><span class="mtk6">0</span></div></div><div class="code-line" data-line-number="7" data-line-start="7" data-line-end="7"><div class="line-content"><span class="mtk1"></span><span class="mtk5">geometry</span><span class="mtk1">:</span></div></div><div class="code-line" data-line-number="8" data-line-start="8" data-line-end="8"><div class="line-content"><span class="mtk1"></span><span class="mtk5">title</span><span class="mtk1">:</span></div></div><div class="code-line" data-line-number="9" data-line-start="9" data-line-end="9"><div class="line-content"><span class="mtk1"></span><span class="mtk5">median_dy_max</span><span class="mtk1">: </span><span class="mtk6">15</span></div></div><div class="code-line" data-line-number="10" data-line-start="10" data-line-end="10"><div class="line-content"><span class="mtk1"></span><span class="mtk5">median_dx_max</span><span class="mtk1">: </span><span class="mtk6">15</span></div></div><div class="code-line" data-line-number="11" data-line-start="11" data-line-end="11"><div class="line-content"><span class="mtk1"></span><span class="mtk5">list</span><span class="mtk1">:</span></div></div><div class="code-line" data-line-number="12" data-line-start="12" data-line-end="12"><div class="line-content"><span class="mtk1"></span><span class="mtk5">median_dy_max</span><span class="mtk1">: </span><span class="mtk6">20</span></div></div><div class="code-line" data-line-number="13" data-line-start="13" data-line-end="13"><div class="line-content"><span class="mtk1"></span><span class="mtk5">table</span><span class="mtk1">:</span></div></div><div class="code-line" data-line-number="14" data-line-start="14" data-line-end="14"><div class="line-content"><span class="mtk1"></span><span class="mtk5">median_dy_max</span><span class="mtk1">: </span><span class="mtk6">25</span></div></div></div></div></div></pre>

### 2. Update

region_diff.py

* Load `corpus_gates.yaml`.
* Emit `PASS/FAIL` status per doc/label based on thresholds.
* Exit with non-zero code if any gate fails (CI integration).

### 3. CI Artifact: Top Offenders

* Already generating

  worst_regions/ with ref/gen/diff triplets.
* Add a `top_offenders.md` artifact summarizing top 20 globally with links.

## Phase 3 Exit Criteria

* [ ] All corpus docs pass **structure** gates (hit_boundary=0, blank_gen=0).
* [ ] All corpus docs pass **geometry** gates (median |dy| < threshold).
* [ ]

  worst_regions/ report generated and stable.
* [ ] CI script (`scripts/ci_corpus_check.py`) exists and returns 0.
