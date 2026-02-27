#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # pymupdf
import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SSIM = True
except Exception:
    HAS_SSIM = False


# ---------- Helpers ----------

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def to_grayscale_uint8(arr_rgb: np.ndarray) -> np.ndarray:
    # arr_rgb: HxWx3 uint8
    # luminance weights reduce noise vs simple mean
    r = arr_rgb[..., 0].astype(np.float32)
    g = arr_rgb[..., 1].astype(np.float32)
    b = arr_rgb[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(gray, 0, 255).astype(np.uint8)

def is_blank(gray: np.ndarray, mean_thresh: float = 250.0, std_thresh: float = 2.0) -> bool:
    if gray.size == 0:
        return True
    m = float(gray.mean())
    s = float(gray.std())
    return (m >= mean_thresh) and (s <= std_thresh)

def dynamic_win_size(h: int, w: int, preferred: int = 11) -> Optional[int]:
    m = min(h, w)
    if m < 7:
        return None
    win = min(preferred, m)
    # SSIM needs odd win_size
    if win % 2 == 0:
        win -= 1
    if win < 7:
        return None
    return win

def mad_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # 1 - normalized mean absolute difference
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float(max(0.0, 1.0 - (diff.mean() / 255.0)))

def absdiff_heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # returns grayscale diff image uint8
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    return diff

def get_canonical_rect(page: fitz.Page) -> fitz.Rect:
    """Return the canonical page rect (cropbox if valid, else mediabox)."""
    cb = page.cropbox
    mb = page.mediabox
    # Use cropbox if it's not identical to mediabox and represents a proper box
    if cb and cb != mb and cb.width > 0 and cb.height > 0:
        return cb
    return mb

def bbox_norm_to_points(page: fitz.Page, bbox_norm: List[float]) -> fitz.Rect:
    x0, y0, x1, y1 = bbox_norm
    canonical = get_canonical_rect(page)
    w = canonical.width
    h = canonical.height
    # Offset by canonical origin (e.g., if cropbox has non-zero x0/y0)
    ox = canonical.x0
    oy = canonical.y0
    # convert normalized to points
    return fitz.Rect(ox + x0 * w, oy + y0 * h, ox + x1 * w, oy + y1 * h)

def clamp_rect_to_page(page: fitz.Page, rect: fitz.Rect) -> fitz.Rect:
    r = fitz.Rect(rect)
    # clamp to canonical page rect
    pr = get_canonical_rect(page)
    r.x0 = max(pr.x0, min(r.x0, pr.x1))
    r.x1 = max(pr.x0, min(r.x1, pr.x1))
    r.y0 = max(pr.y0, min(r.y0, pr.y1))
    r.y1 = max(pr.y0, min(r.y1, pr.y1))
    # ensure proper ordering
    if r.x1 < r.x0:
        r.x0, r.x1 = r.x1, r.x0
    if r.y1 < r.y0:
        r.y0, r.y1 = r.y1, r.y0
    return r

def render_full_page_gray(page: fitz.Page, dpi: int) -> np.ndarray:
    """Render full page to grayscale numpy array at given DPI."""
    try:
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        return to_grayscale_uint8(arr)
    except Exception:
        return np.zeros((0, 0), dtype=np.uint8)

def crop_from_raster(full_gray: np.ndarray, bbox_norm: List[float]) -> np.ndarray:
    """Crop a region from full-page raster using normalized bbox (same frame as Oracle ML)."""
    if full_gray.size == 0:
        return full_gray
    h, w = full_gray.shape[:2]
    x0, y0, x1, y1 = bbox_norm
    # Convert to pixel coordinates (top-left origin)
    px0 = int(round(x0 * w))
    py0 = int(round(y0 * h))
    px1 = int(round(x1 * w))
    py1 = int(round(y1 * h))
    # Clamp to image bounds
    px0 = max(0, min(px0, w))
    px1 = max(0, min(px1, w))
    py0 = max(0, min(py0, h))
    py1 = max(0, min(py1, h))
    if px1 <= px0 or py1 <= py0:
        return np.zeros((0, 0), dtype=np.uint8)
    return full_gray[py0:py1, px0:px1]

def render_clip_gray(page: fitz.Page, rect: fitz.Rect, dpi: int) -> np.ndarray:
    """LEGACY: Render with point-space clip. Use crop_from_raster for Oracle-aligned cropping."""
    try:
        pix = page.get_pixmap(dpi=dpi, clip=rect, alpha=False)
        # pix.n should be 3 (RGB)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if arr.shape[2] >= 3:
            arr = arr[:, :, :3]
        return to_grayscale_uint8(arr)
    except Exception:
        return np.zeros((0, 0), dtype=np.uint8)

def save_gray_png(path: Path, gray: np.ndarray) -> None:
    if gray.size == 0:
        return
    Image.fromarray(gray).save(path)

def _compute_drift_stats(drifts_x: List[int], drifts_y: List[int]) -> Dict[str, Any]:
    """Compute median and IQR for drift values."""
    if not drifts_x or not drifts_y:
        return {"median_dx": None, "median_dy": None, "iqr_dx": None, "iqr_dy": None, "count": 0}
    
    arr_x = np.array(drifts_x, dtype=np.float32)
    arr_y = np.array(drifts_y, dtype=np.float32)
    
    return {
        "median_dx": float(np.median(arr_x)),
        "median_dy": float(np.median(arr_y)),
        "p25_dx": float(np.percentile(arr_x, 25)),
        "p75_dx": float(np.percentile(arr_x, 75)),
        "p25_dy": float(np.percentile(arr_y, 25)),
        "p75_dy": float(np.percentile(arr_y, 75)),
        "iqr_dx": float(np.percentile(arr_x, 75) - np.percentile(arr_x, 25)),
        "iqr_dy": float(np.percentile(arr_y, 75) - np.percentile(arr_y, 25)),
        "count": len(drifts_x)
    }

# ---------- Core ----------

def compute_region_score(ref_gray: np.ndarray, gen_gray: np.ndarray) -> Tuple[str, float, str]:
    """
    Returns (method_used, score, status_detail)
    """
    # Ensure same shape: strict baseline uses same coords, but images can differ due to page boxes
    # Use center-crop to common size (no resizing warps geometry)
    if ref_gray.size == 0 or gen_gray.size == 0:
        return ("error", 0.0, "empty_crop")
        
    h = min(ref_gray.shape[0], gen_gray.shape[0])
    w = min(ref_gray.shape[1], gen_gray.shape[1])
    
    if h == 0 or w == 0:
        return ("error", 0.0, "empty_intersection")

    ref_c = ref_gray[:h, :w]
    gen_c = gen_gray[:h, :w]

    if HAS_SSIM:
        win = dynamic_win_size(h, w, preferred=11)
        if win is not None:
            try:
                score = float(ssim(ref_c, gen_c, data_range=255, win_size=win))
                return ("ssim", score, f"win_size={win}")
            except Exception:
                pass # fallback to MAD
                
    score = mad_similarity(ref_c, gen_c)
    return ("mad", score, "fallback")

def process_doc(case_dir: Path, dpi: int, min_region_px: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    hints_path = case_dir / "layout_hints.json"
    ref_pdf = case_dir / "ref.pdf"
    gen_pdf = case_dir / "gen.pdf"

    out_stats_path = case_dir / "region_stats.json"
    worst_dir = case_dir / "worst_regions"
    if worst_dir.exists():
        shutil.rmtree(worst_dir)
    ensure_dir(worst_dir)

    if not (hints_path.exists() and ref_pdf.exists() and gen_pdf.exists()):
        return ({"doc_id": case_dir.name, "error": "missing_inputs"}, [])

    hints = load_json(hints_path)
    regions = hints.get("regions", [])
    doc_id = hints.get("doc_id", case_dir.name)

    ref = fitz.open(ref_pdf)
    gen = fitz.open(gen_pdf)

    per_region: List[Dict[str, Any]] = []
    scored_regions: List[Dict[str, Any]] = []

    ref_pages = len(ref)
    gen_pages = len(gen)
    page_count_match = (ref_pages == gen_pages)

    # Detailed page box logging (critical for drift diagnosis)
    page_boxes = []
    for i in range(min(ref_pages, gen_pages)):
        ref_page = ref[i]
        gen_page = gen[i]
        
        page_boxes.append({
            "page": i,
            "ref": {
                "rect": [ref_page.rect.x0, ref_page.rect.y0, ref_page.rect.x1, ref_page.rect.y1],
                "mediabox": [ref_page.mediabox.x0, ref_page.mediabox.y0, ref_page.mediabox.x1, ref_page.mediabox.y1],
                "cropbox": [ref_page.cropbox.x0, ref_page.cropbox.y0, ref_page.cropbox.x1, ref_page.cropbox.y1],
                "rotation": ref_page.rotation,
            },
            "gen": {
                "rect": [gen_page.rect.x0, gen_page.rect.y0, gen_page.rect.x1, gen_page.rect.y1],
                "mediabox": [gen_page.mediabox.x0, gen_page.mediabox.y0, gen_page.mediabox.x1, gen_page.mediabox.y1],
                "cropbox": [gen_page.cropbox.x0, gen_page.cropbox.y0, gen_page.cropbox.x1, gen_page.cropbox.y1],
                "rotation": gen_page.rotation,
            }
        })

    # Stats tracking per label
    label_stats = {} 
    # structure: label -> {scores: [], drifts_x: [], drifts_y: [], total: 0, scored: 0, missing: 0, tiny: 0, blank_both: 0}

    for idx, r in enumerate(regions):
        page = int(r.get("page", 0))
        label = str(r.get("label", "unknown"))
        source = str(r.get("source", "unknown"))
        conf = r.get("confidence", None)

        if label not in label_stats:
            label_stats[label] = {"scores": [], "drifts_x": [], "drifts_y": [], "total": 0, "scored": 0, "missing": 0, "tiny": 0, "blank_both": 0}
        
        label_stats[label]["total"] += 1

        bbox_norm = r.get("bbox_norm") or r.get("bbox")  # be forgiving
        region_id = r.get("id") or f"p{page}_{label}_{idx:03d}"

        record = {
            "region_id": region_id,
            "doc_id": doc_id,
            "page": page,
            "label": label,
            "source": source,
            "confidence": conf,
            "bbox_norm": bbox_norm,
            "status": None,
            "score": None,
            "method": None,
            "detail": None,
            "blank_ref": None,
            "blank_gen": None,
        }

        if bbox_norm is None or len(bbox_norm) != 4:
            record["status"] = "INVALID_BBOX"
            per_region.append(record)
            continue

        if page >= ref_pages or page >= gen_pages:
            record["status"] = "MISSING_PAGE"
            label_stats[label]["missing"] += 1
            per_region.append(record)
            continue

        ref_page = ref[page]
        gen_page = gen[page]

        # === ORACLE-ALIGNED RENDERING ===
        # Render full page at same DPI (150), then crop in pixel space
        # This matches how Oracle ML detects regions (top-left origin, pixel coords)
        ref_full = render_full_page_gray(ref_page, dpi=dpi)
        gen_full = render_full_page_gray(gen_page, dpi=dpi)
        
        # Crop from full-page raster using normalized bbox
        ref_gray = crop_from_raster(ref_full, bbox_norm)
        gen_gray = crop_from_raster(gen_full, bbox_norm)

        # tiny region skip
        if ref_gray.shape[0] < min_region_px or ref_gray.shape[1] < min_region_px:
            record["status"] = "SKIPPED_TINY_REGION"
            label_stats[label]["tiny"] += 1
            per_region.append(record)
            continue

        blank_ref = is_blank(ref_gray)
        blank_gen = is_blank(gen_gray)
        record["blank_ref"] = blank_ref
        record["blank_gen"] = blank_gen

        if blank_ref and blank_gen:
            record["status"] = "SKIPPED_BLANK_REGION"
            label_stats[label]["blank_both"] += 1
            per_region.append(record)
            continue


        method, score, detail = compute_region_score(ref_gray, gen_gray)
        record["status"] = "SCORED"
        record["score"] = score
        record["method"] = method
        record["detail"] = detail

        # --- Diagnostic: Aligned SSIM ---
        # Search for better match by shifting window to detect drift
        aligned_score = score
        drift_x = 0
        drift_y = 0
        hit_boundary = False
        search_pad_px = 64
        
        # Only run costly alignment search if valid score exists
        if score is not None:
             # Adaptive padding: 64px base, larger for bigger regions
             rh_est, rw_est = ref_gray.shape if ref_gray.size > 0 else (50, 50)
             search_pad_px = max(64, int(0.05 * min(rw_est, rh_est)))
             pad_px = search_pad_px
             
             h_full, w_full = gen_full.shape[:2]
             
             # Convert bbox_norm to pixel coords with padding
             x0, y0, x1, y1 = bbox_norm
             px0 = int(round(x0 * w_full)) - pad_px
             py0 = int(round(y0 * h_full)) - pad_px
             px1 = int(round(x1 * w_full)) + pad_px
             py1 = int(round(y1 * h_full)) + pad_px
             
             # Clamp to image bounds
             px0 = max(0, px0)
             py0 = max(0, py0)
             px1 = min(w_full, px1)
             py1 = min(h_full, py1)
             
             # Get expanded gen crop
             gen_gray_large = gen_full[py0:py1, px0:px1]
             
             # We want to match ref_gray inside gen_gray_large
             rh, rw = ref_gray.shape
             gh, gw = gen_gray_large.shape
             
             if gh >= rh and gw >= rw:
                 best_s = -1.0
                 best_dx = 0
                 best_dy = 0
                 
                 # Coarse search (step 8 for speed with larger 64px window)
                 for dy in range(0, gh - rh + 1, 8):
                     for dx in range(0, gw - rw + 1, 8):
                         cand = gen_gray_large[dy:dy+rh, dx:dx+rw]
                         
                         if method == "ssim" and HAS_SSIM:
                             win = dynamic_win_size(rh, rw, preferred=7)
                             if win:
                                try:
                                    s = float(ssim(ref_gray, cand, data_range=255, win_size=win))
                                except:
                                    s = mad_similarity(ref_gray, cand)
                             else:
                                 s = mad_similarity(ref_gray, cand)
                         else:
                             s = mad_similarity(ref_gray, cand)
                             
                         if s > best_s:
                             best_s = s
                             best_dx = dx
                             best_dy = dy
                 
                 # Fine search in 5x5 window around coarse best (step 1)
                 coarse_dx, coarse_dy = best_dx, best_dy
                 for dy in range(max(0, coarse_dy - 4), min(gh - rh + 1, coarse_dy + 5)):
                     for dx in range(max(0, coarse_dx - 4), min(gw - rw + 1, coarse_dx + 5)):
                         cand = gen_gray_large[dy:dy+rh, dx:dx+rw]
                         
                         if method == "ssim" and HAS_SSIM:
                             win = dynamic_win_size(rh, rw, preferred=7)
                             if win:
                                try:
                                    s = float(ssim(ref_gray, cand, data_range=255, win_size=win))
                                except:
                                    s = mad_similarity(ref_gray, cand)
                             else:
                                 s = mad_similarity(ref_gray, cand)
                         else:
                             s = mad_similarity(ref_gray, cand)
                             
                         if s > best_s:
                             best_s = s
                             best_dx = dx
                             best_dy = dy
                 
                 # Drift = offset from center of padded region
                 drift_x = best_dx - pad_px
                 drift_y = best_dy - pad_px
                 
                 # Check if hit boundary (drift at edge of search window)
                 boundary_margin = 2  # 2px tolerance
                 if abs(drift_x) >= (pad_px - boundary_margin) or abs(drift_y) >= (pad_px - boundary_margin):
                     hit_boundary = True
                 
                 aligned_score = max(score, best_s)

        record["aligned_score"] = aligned_score
        record["drift_x"] = drift_x
        record["drift_y"] = drift_y
        record["search_pad_px"] = search_pad_px
        record["hit_boundary"] = hit_boundary
        record["ref_raster_dims"] = list(ref_full.shape[:2]) if ref_full.size > 0 else None
        record["gen_raster_dims"] = list(gen_full.shape[:2]) if gen_full.size > 0 else None

        label_stats[label]["scored"] += 1
        label_stats[label]["scores"].append(score)
        label_stats[label]["drifts_x"].append(drift_x)
        label_stats[label]["drifts_y"].append(drift_y)
        
        per_region.append(record)
        scored_regions.append(record)

    # Compile Per-Label Summary with Coverage
    by_label_summary = {}
    for label, stats in label_stats.items():
        arr = np.array(stats["scores"], dtype=np.float32)
        
        # Coverage calculation
        total = stats["total"]
        scored_count = stats["scored"]
        effective_total = total - stats["missing"] - stats["tiny"] - stats["blank_both"]
        
        cov_strict = scored_count / total if total > 0 else 0.0
        cov_effective = scored_count / effective_total if effective_total > 0 else 0.0
        
        by_label_summary[label] = {
            "count_scored": int(arr.size),
            "avg": float(arr.mean()) if arr.size else None,
            "p50": float(np.percentile(arr, 50)) if arr.size else None,
            "p10": float(np.percentile(arr, 10)) if arr.size else None,
            "min": float(arr.min()) if arr.size else None,
            "coverage_strict": float(cov_strict),
            "coverage_effective": float(cov_effective),
            "stats": {
                "total": total,
                "missing": stats["missing"],
                "tiny": stats["tiny"],
                "blank_both": stats["blank_both"]
            },
            "drift": _compute_drift_stats(stats["drifts_x"], stats["drifts_y"])
        }

    doc_stats = {
        "doc_id": doc_id,
        "case_dir": case_dir.name,
        "ref_pages": ref_pages,
        "gen_pages": gen_pages,
        "page_count_match": page_count_match,
        "page_boxes": page_boxes,
        "by_label": by_label_summary,
        "regions": per_region,
    }

    save_json(out_stats_path, doc_stats)

    ref.close()
    gen.close()

    return (doc_stats, scored_regions)

def save_worst_regions(worst: List[Dict[str, Any]], corpus_root: Path, dpi: int, out_dir: Path) -> None:
    ensure_dir(out_dir)
    # Create images for worst regions (ref/gen/diff)
    for i, rec in enumerate(worst):
        case_dir = corpus_root / rec["case_dir"]
        ref_pdf = case_dir / "ref.pdf"
        gen_pdf = case_dir / "gen.pdf"
        hints_path = case_dir / "layout_hints.json"
        
        if not (ref_pdf.exists() and gen_pdf.exists() and hints_path.exists()):
            continue

        # Look up bbox using hints again (stateless, robust)
        hints = load_json(hints_path)
        regions_list = hints.get("regions", [])
        region_id = rec["region_id"]
        
        target = None
        # Try to match by ID if present, otherwise reconstruct ID logic
        for idx, r in enumerate(regions_list):
            page_r = int(r.get("page", 0))
            label_r = str(r.get("label", "unknown"))
            rid = r.get("id") or f"p{page_r}_{label_r}_{idx:03d}"
            
            if rid == region_id:
                target = r
                break
                
        if target is None:
            continue

        page = int(target.get("page", 0))
        bbox_norm = target.get("bbox_norm") or target.get("bbox")
        if bbox_norm is None:
            continue

        ref = fitz.open(ref_pdf)
        gen = fitz.open(gen_pdf)
        if page >= len(ref) or page >= len(gen):
            ref.close()
            gen.close()
            continue

        ref_page = ref[page]
        gen_page = gen[page]
        rect_ref = clamp_rect_to_page(ref_page, bbox_norm_to_points(ref_page, bbox_norm))
        rect_gen = clamp_rect_to_page(gen_page, bbox_norm_to_points(gen_page, bbox_norm))

        ref_gray = render_clip_gray(ref_page, rect_ref, dpi=dpi)
        gen_gray = render_clip_gray(gen_page, rect_gen, dpi=dpi)

        # center-crop to common size
        h = min(ref_gray.shape[0], gen_gray.shape[0])
        w = min(ref_gray.shape[1], gen_gray.shape[1])
        
        if h > 0 and w > 0:
            ref_c = ref_gray[:h, :w]
            gen_c = gen_gray[:h, :w]
            diff = absdiff_heatmap(ref_c, gen_c)

            prefix = f"{i+1:02d}_{rec['case_dir']}_{rec['label']}_p{page}"
            save_gray_png(out_dir / f"{prefix}_ref.png", ref_c)
            save_gray_png(out_dir / f"{prefix}_gen.png", gen_c)
            save_gray_png(out_dir / f"{prefix}_diff.png", diff)

        ref.close()
        gen.close()


def generate_top_offenders_md(worst: List[Dict[str, Any]], corpus_root: Path, out_dir: Path) -> None:
    """Generate a Markdown artifact summarizing worst regions with image links."""
    lines = [
        "# Top Offenders Report",
        "",
        "Worst-performing regions by strict SSIM score.",
        "",
        "| # | Document | Page | Label | Strict SSIM | Aligned SSIM | Drift (dx, dy) |",
        "|---|----------|------|-------|-------------|--------------|----------------|",
    ]
    
    for i, w in enumerate(worst):
        doc = w.get("case_dir", "?")
        page = w.get("page", 0)
        label = w.get("label", "?")
        score = w.get("score", 0)
        aligned = w.get("aligned_score", 0)
        drift_x = w.get("drift_x", 0)
        drift_y = w.get("drift_y", 0)
        
        lines.append(
            f"| {i+1} | {doc} | {page} | {label} | {score:.3f} | {aligned:.3f} | ({drift_x}, {drift_y}) |"
        )
    
    lines.append("")
    lines.append("## Image Previews")
    lines.append("")
    
    for i, w in enumerate(worst[:10]):  # Limit embedded images to top 10
        doc = w.get("case_dir", "?")
        page = w.get("page", 0)
        label = w.get("label", "?")
        prefix = f"{i+1:02d}_{doc}_{label}_p{page}"
        
        ref_img = out_dir / f"{prefix}_ref.png"
        gen_img = out_dir / f"{prefix}_gen.png"
        diff_img = out_dir / f"{prefix}_diff.png"
        
        if ref_img.exists():
            lines.append(f"### {i+1}. {doc} p{page} {label}")
            lines.append("")
            lines.append(f"| Reference | Generated | Diff |")
            lines.append(f"|-----------|-----------|------|")
            lines.append(f"| ![ref]({ref_img.name}) | ![gen]({gen_img.name}) | ![diff]({diff_img.name}) |")
            lines.append("")
    
    md_path = out_dir / "top_offenders.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Top offenders artifact saved: {md_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-root", type=str, default="tests/corpus")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--min-region-px", type=int, default=32)
    ap.add_argument("--max-worst", type=int, default=20)
    args = ap.parse_args()

    corpus_root = Path(args.corpus_root)
    if not corpus_root.exists():
        raise SystemExit(f"Corpus root not found: {corpus_root}")

    all_scored: List[Dict[str, Any]] = []
    global_docs: List[Dict[str, Any]] = []

    for case_dir in sorted([p for p in corpus_root.iterdir() if p.is_dir()]):
        stats, scored = process_doc(case_dir, dpi=args.dpi, min_region_px=args.min_region_px)
        if "error" in stats:
            continue
        # attach case_dir for global worst tracking
        for r in scored:
            r2 = dict(r)
            r2["case_dir"] = stats["case_dir"]
            all_scored.append(r2)
        global_docs.append(stats)

    # global worst (lowest scores)
    scored_only = [r for r in all_scored if r.get("status") == "SCORED" and r.get("score") is not None]
    scored_only.sort(key=lambda x: float(x["score"]))
    worst = scored_only[: args.max_worst]

    global_summary = {
        "corpus_root": str(corpus_root),
        "docs_processed": len(global_docs),
        "regions_scored": len(scored_only),
        "worst_regions": [
            {
                "case_dir": w["case_dir"],
                "doc_id": w.get("doc_id"),
                "region_id": w["region_id"],
                "label": w["label"],
                "page": w["page"],
                "score": w["score"],
                "aligned_score": w.get("aligned_score"),
                "method": w.get("method"),
                "drift": [w.get("drift_x"), w.get("drift_y")]
            }
            for w in worst
        ],
    }

    save_json(corpus_root / "region_diff_global.json", global_summary)
    save_worst_regions(worst, corpus_root=corpus_root, dpi=args.dpi, out_dir=corpus_root / "worst_regions")
    generate_top_offenders_md(worst, corpus_root=corpus_root, out_dir=corpus_root / "worst_regions")

    # console report
    if worst:
        print("\nTop worst regions (strict baseline):")
        for i, w in enumerate(global_summary["worst_regions"]):
            print(f"- {i+1}. {w['case_dir']} p{w['page']} {w['label']} score={w['score']:.3f} aligned={w.get('aligned_score', 0):.3f} drift={w.get('drift', [0,0])} ({w['method']})")
    
    print(f"\nStats Summary saved: {corpus_root / 'region_diff_global.json'}")
    print(f"Heatmap triplets saved: {corpus_root / 'worst_regions'}")

if __name__ == "__main__":
    main()
