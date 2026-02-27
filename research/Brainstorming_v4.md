
## What to prioritize next (highest impact → lowest)

### Priority 0 (must): Bundle metric-compatible fonts

Your current line “Calibri → Arial due to CI constraints” guarantees pagination drift.

**Do this first:**

* Bundle `Carlito` (Calibri metrics) and `TeX Gyre Termes` (Times-ish) into `fonts/`.
* Make font loading deterministic via `fontspec` `Path=fonts/`.

**Why:** It reduces cumulative reflow across the entire document. It will lift title SSIM *and* reduce page count drift.

---

## Phase 4 is the real “Word-equivalent” move

### Priority 1: Tab stops (resumes, headers, right-aligned dates)

Your plan here is correct and should be  **immediate** .

**Implementation reality check:**

* `tabto` works if you can consistently measure positions against the same left margin.
* Tabs in Word are absolute within the paragraph’s text area and depend on:
  * section margins
  * paragraph indent (`w:ind left/right/firstLine/hanging`)
  * default tab stop (Word uses a default interval when no explicit tab stop exists)

**Solution pattern**

1. **Compute text area origin** (page left margin + paragraph indent)
2. Convert each Word tab stop `pos` (twips) to TeX points
3. Emit `\tabto{<absolute-from-left-margin>}` or `\tabto*{...}` depending on alignment
4. Collapse repeated tabs into a single stop jump
5. For right-aligned tab (dates): use `\tabto{X}\hfill` only inside a bounded box, or better: use `\makebox[\linewidth][r]{...}` after a `\tabto{...}`

**Why this works:** you stop “guessing” with `\hfill` and reproduce Word’s positional intent.

---

### Priority 2: Absolute-positioned textboxes (floating shapes)

Also correct. This is where “semantic LaTeX only” loses.

**Policy**

* If textbox is a simple rectangle + no complex wrapping:
  * render as **tcolorbox** placed by coordinates
* If it’s SmartArt / complex wrap / nested drawing:
  * fallback to **image** of that region (still track it)

**Implementation**

* Use `textpos` (`\begin{textblock*}{w}(x,y) ...`) or TikZ overlay.
* Coordinates should come from:
  * Oracle PDF region bbox OR OOXML anchor offsets (if reliable)
* Keep it editable by putting text inside the box when possible.

**Why:** Word’s floating objects are not “flow layout.” They are geometry constraints.

---

### Priority 3: Fixed-width tables using tblGrid + gridSpan

Your wording is right, but here’s the key: **never let LaTeX decide widths** when Word already decided them.

**Do this**

* If `tblGrid` exists, compute each grid column width.
* For each cell:
  * width = sum(grid widths for spanned columns)
  * emit `p{<exact width>}` columns (or `m{}` via array) instead of `X`
* Only use `tabularx/ltablex` for overflow tables *without* grid widths.

**Why:** Table width decisions are a major source of drift and page inflation.

---

## How these changes connect to your SSIM metrics

Right now your SSIM is a strict baseline. Good. Use it as a driver:

* **If aligned_ssim >> strict_ssim:** fix geometry/fonts/spacing (global drift)
* **If strict+aligned low in titles:** usually font mismatch or wrong crop
* **If tables low:** enforce grid widths
* **If “date alignment” low:** tabs

Your Phase 4 items directly target the regions that SSIM punishes most.

---

## Updated acceptance criteria (more realistic)

Your “Title strict SSIM > 0.6” is fine  *after fonts are bundled* .

Add these two, because they reflect real Word equivalence:

* **Right-aligned tab regions** : SSIM > 0.75 after tab-stop engine
* **Textbox regions** : SSIM > 0.75 for “simple boxes rendered as textpos+tcolorbox”, else explicitly counted as “image fallback”

Also track:

* % of paragraphs using explicit tab stops successfully converted (coverage)

---

## One correction to your plan

This line should be updated once you bundle fonts:

> “Fallback used due to CI constraints: Calibri → Arial…”

Replace with:

* **Calibri → Carlito (bundled)**
* fallback to Arial only if font load fails

---

## If you want the shortest path to “Word-equivalent”

Do these in order:

1. Bundle fonts (Carlito + Termes) and enforce fontspec loading from repo
2. Implement tab stops with absolute positions (kills resume alignment issues)
3. Implement absolute textboxes policy (kills Word-only layouts)
4. Enforce tblGrid widths (kills table drift and pagination drift)

That’s the practical, deterministic hybrid engine that actually mimics Word.




## 1) Phase 3 is good — but fix the font story (it’s still the #1 driver)

You still say:

> “Fallback used due to CI constraints: Calibri → Arial…”

If Phase 4 tabs are “completed,” you should  **also complete reproducible fonts** , otherwise your SSIM + pagination targets won’t stabilize.

### Concrete fix

* Bundle **Carlito** and **TeX Gyre Termes** into `fonts/`
* Make fontspec load from repo path
* Fail loudly (or warn + mark run non-reproducible) if fonts are missing

**Why:** Without metric-compatible fonts, your geometry and spacing work will still drift due to line breaks, especially in academic templates.

---

## 2) Tabs: your current generator choice is slightly wrong

You wrote:

> “Use tabto package (\tabto{in}). Implemented \llap for right alignment…”

That will work sometimes, but **it will break whenever paragraph indents exist** (and in Word they often do). Tab stops are measured from the paragraph’s text origin, not from the physical page.

### Correct approach (deterministic)

When you see a tab stop at `pos_twips`, compute:

**absolute_tab_pos = (left_margin + paragraph_left_indent + pos_twips) in page coordinates**

Then implement tabs using one of these two patterns:

#### Best pattern (stable for right-aligned dates)

Use a fixed-width line container and compute the tab position inside it:

* `\makebox[\linewidth][l]{ <left text> \hspace{X} \makebox[0pt][r]{<right text>} }`

Where **X is computed** from the tab stop position relative to the paragraph origin.

#### If you keep tabto

Use `\tabto{<pos from paragraph origin>}`  **not inches** , but points computed from twips, and ensure the paragraph origin is consistent (indent + margins).

**Quick rule:** never mix “tab stop inches” with page-level coordinates unless you incorporate indent/margins.

✅ Your unit tests passing is good, but this is exactly the kind of thing unit tests miss unless you have tab+indent fixtures.

---

## 3) Textboxes: add one missing policy branch (wrap mode)

Your textbox plan is right, but you need one extra decision:

### Add this policy

* If textbox wrap is **inline** (behaves like text in flow): treat as normal block
* If textbox is  **floating** :
  * if simple rect: `textpos + tcolorbox`
  * else: image fallback

This avoids misplacing inline “callout boxes” that Word treats as anchored in flow.

---

## 4) Tables: switch from tabularx to explicit p{} columns when tblGrid exists

You wrote:

> “Generate tabularx with p{} instead of X ratios…”

If you already know the grid widths, **don’t use tabularx at all** unless you need automatic wrapping behavior. The most faithful is:

* Use `tabular` or `array` with **explicit `p{}` / `m{}` widths**
* The table width should equal the Word table width.

### Why this matters

`tabularx` introduces its own optimization step. Word already decided widths. Your goal is to reproduce Word, so remove extra degrees of freedom.

**Exception:** huge tables that overflow → then use `ltablex` or image fallback, but treat it as a policy decision.

---

## 5) Update your acceptance checks (SSIM alone isn’t enough)

Keep your SSIM targets, but add two fidelity KPIs that reflect real Word equivalence:

### Add these

1. **Pagination drift (non-table-heavy docs):** within ±10% pages
2. **Right-aligned tab regions:** SSIM > 0.75 (these are the “resume date alignment” cases)

Also track a reproducibility flag:

* `fonts_reproducible: true/false`
  If false, don’t compare page counts as a hard KPI.

---

## 6) What I would prioritize next (action list)

Since tabs are “completed” and you want solutions:

1. **Reproducible serif font** (bundle Termes or another consistent serif)
2. **Fix tab coordinate model to include indent + margins** (this is the hidden bug)
3. **tblGrid tables rendered with explicit p{} widths (no tabularx when grid known)**
4. **textbox wrap mode policy branch**

That will move you from “works” to “Word-like.”

---

## Minimal edits to your plan text

If you only want to patch wording:

* Replace “\tabto{in}” with “\tabto{}”
* Replace “tabularx with p{width}” with “tabular/array with explicit p{width} columns when tblGrid exists”
* Add “wrap mode” branch in textbox policy
* Replace Calibri→Arial with Carlito preferred + fallback only if fonts missing

---

If you want, I can write the exact **tab position formula** (twips → pt, paragraph origin, indent interplay) in 5–10 lines of deterministic pseudocode, but the key is: **tab stops must be computed relative to paragraph origin, not the page.**



Nice — this is the right direction. Now let’s stay solution-focused and close the gap from **0.532 aligned SSIM** to  **>0.6 strict** . Your numbers already tell you what’s left:

* **Aligned SSIM > raw SSIM** + **drift [-12, -12] px** ⇒ this is still  **systemic coordinate drift** , not missing content.
* With fonts + line spacing stabilized, the remaining drift is almost always one of:  **page box mismatch** ,  **margin origin mismatch** ,  **PDF cropbox/mediabox** , or  **render DPI / rounding** .

Below is the practical fix list, in priority order, with what to implement.

---

## 1) Fix the drift at the source: use the same page box everywhere (most common cause)

### Problem

Word-rendered PDFs and LaTeX PDFs often differ in:

* MediaBox vs CropBox vs TrimBox
* origin placement (some PDFs have non-zero origins)
* “visible page” differs from “physical page”

If your region_diff crops using one box but generator/Oracle uses another, you get a constant offset like  **[-12, -12]** .

### Solution

In **both** Layout Oracle and region_diff, normalize everything to a single box:

* Prefer `page.rect` in PyMuPDF  **but explicitly choose which one** :
  * `page.mediabox` vs `page.cropbox`

**Do this consistently:**

* Use `page.cropbox` for cropping and bbox normalization **if it exists and differs**
* Otherwise use `page.mediabox`
* Record which box is used in `layout_hints.json` (per page)

**Implementation detail**
When converting normalized bbox → points:

* map to the chosen box, not `page.rect` blindly.

This alone often removes the constant drift.

---

## 2) Fix the coordinate direction + rounding (12px often comes from rounding/DPI mismatch)

### Problem

* Oracle regions are normalized floats.
* When you map to pixels, you round, then crop.
* If ref and gen use slightly different DPI or rounding strategy, you get consistent shifts.

### Solution

Make the mapping deterministic and symmetric:

* Use **points space** first (PDF native units), then rasterize.
* Avoid “pixel space conversion” before cropping.

**Recommended pipeline in region_diff:**

1. Convert bbox_norm → **PDF points rect** (float)
2. Crop the PDF page render using `clip=rect` when rasterizing (`get_pixmap(clip=rect, dpi=...)`)
3. Then SSIM on resulting raster

This avoids pixel rounding errors from “render full page then crop pixels”.

---

## 3) Confirm geometry is applied correctly: Word margins vs LaTeX geometry semantics

You’re extracting `w:pgSz` and `w:pgMar`, good — but the last few percent often comes from:

* Word header/footer distance
* gutter / mirror margins
* different default top baseline start

### Solution

Parse and apply these if present:

* `w:pgMar @header` and `@footer`
* `w:gutter`
* `w:mirrorMargins`

Then emit in LaTeX:

* `\geometry{includehead, includefoot, headheight=..., headsep=..., footskip=...}` if you model header/footer
* or if you don’t support headers yet, explicitly disable them so baseline is consistent.

If Resume_1 has no headers, still check: Word sometimes keeps a non-zero header distance.

---

## 4) Stop using baselineskip for “exact” unless you’re matching Word’s rule precisely

This is subtle, but important:

### Word “exact” line spacing ≠ LaTeX baselineskip exactly

Word’s “exact” is effectively a fixed line height box, while LaTeX baseline spacing interacts with font ascenders/descenders.

### Better solution for “exact”

For paragraph runs with exact line spacing:

* Wrap with `\linespread{1}` and use `\setlength{\baselineskip}{Xpt}` is okay, but also consider:
  * `\setlength{\lineskip}{0pt}`
  * `\setlength{\lineskiplimit}{0pt}`
    This makes LaTeX less “helpful” and closer to Word’s fixed behavior.

Also ensure you’re not mixing setspace + baselineskip in a conflicting way (choose one per paragraph).

---

## 5) Upgrade region_diff to report the drift cause automatically (so you don’t guess)

You already compute aligned offsets. Now use them to classify:

* If best_dx, best_dy are almost constant across titles: **global page-box offset**
* If dy grows with page Y: **scale mismatch**
* If dx differs by column: **column width mismatch**

Add to the report:

* per-label median (dx,dy)
* per-page median (dx,dy)

This turns “investigate margins” into a deterministic diagnosis.

---

# What I would do next (action plan)

To get Resume_1 titles from 0.53 aligned → >0.6 strict:

### Step A (most likely to fix -12,-12)

 **Make region_diff crop using PDF clip rect in points space** , and make Oracle + region_diff agree on cropbox/mediabox.

### Step B

**Record and use page box type** (`cropbox` vs `mediabox`) in `layout_hints.json`.

### Step C

If drift persists:
Parse and apply `header/footer/gutter/mirrorMargins` from `w:pgMar`.

---

## Quick sanity check you can run (no guesswork)

Compute drift offsets for all title regions:

* if 80%+ titles report the same best_dx/best_dy ≈ (-12,-12), it’s page box / origin mismatch.
* if offsets vary, it’s paragraph model or column width.

---

If you want, paste just these two things from your run:

1. one entry from `layout_hints.json` for a title region (bbox_norm + page index)
2. the page box values from PyMuPDF for ref/gen (`mediabox`, `cropbox`) for page 0

…and I’ll tell you exactly which box mismatch you have and what mapping fix will eliminate the constant offset.
