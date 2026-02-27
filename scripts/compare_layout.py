#!/usr/bin/env python
"""Compare generated LaTeX PDF with reference to identify layout improvements."""

import argparse
from pathlib import Path
import json
import sys
import fitz
from wordtolatex.layout_oracle import extract_layout, compare_layouts, create_layout_hints


def generate_overlay_images(layout, pdf_path: Path, output_dir: Path):
    """Generate images with layout regions overlaid for debugging."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    doc = fitz.open(pdf_path)
    
    # colors for different region types
    colors = {
        "table": (1, 0, 0),    # Red
        "figure": (0, 0, 1),   # Blue
        "textbox": (0, 1, 0),  # Green
        "list": (1, 0.5, 0),   # Orange
        "title": (1, 0, 1),    # Magenta
        "text": (0.5, 0.5, 0.5) # Gray
    }
    
    for i in range(len(doc)):
        page = doc[i]
        w, h = page.rect.width, page.rect.height
        
        # 1. Draw Regions
        page_regions = [r for r in layout.regions if r.page == i]
        
        for r in page_regions:
            # bbox is normalized [0..1]
            rect = fitz.Rect(r.bbox[0]*w, r.bbox[1]*h, r.bbox[2]*w, r.bbox[3]*h)
            
            color = colors.get(r.label, (0.5, 0.5, 0.5))
            
            # Draw rectangle
            page.draw_rect(rect, color=color, width=2)
            # Draw label
            page.insert_text(
                (rect.x0, rect.y0 - 5), 
                f"{r.label} ({r.confidence:.2f})", 
                color=color, 
                fontsize=10
            )
            
        # 2. Draw Columns
        if i < len(layout.column_structure):
            cols = layout.column_structure[i]
            if cols.count > 1:
                for boundary in cols.boundaries:
                    page.draw_line((boundary, 0), (boundary, h), color=(0, 1, 1), width=1)
                    
        # Save
        img_path = output_dir / f"overlay_p{i+1}.png"
        pix = page.get_pixmap(dpi=150)
        pix.save(str(img_path))
        print(f"Saved overlay: {img_path}")
        
    doc.close()


def main():
    parser = argparse.ArgumentParser(description="Compare PDF layouts.")
    parser.add_argument("--ref", type=Path, help="Reference PDF (original Word->PDF)")
    parser.add_argument("--gen", type=Path, help="Generated PDF (LaTeX->PDF)")
    parser.add_argument("--out", type=Path, default=Path("report.json"), help="Output report JSON")
    parser.add_argument("--hints", type=Path, default=Path("layout_hints.json"), help="Output layout hints JSON")
    parser.add_argument("--visuals", action="store_true", help="Generate visual overlays")
    parser.add_argument("--vis-dir", type=Path, default=Path("debug_visuals"), help="Directory for visual overlays")
    parser.add_argument("--run-id", type=str, default="", help="Run ID for freshness tracking")

    args = parser.parse_args()
    
    if not args.ref:
        # Fallback for dev testing
        args.ref = Path("tests/samples/resume.pdf")
        if not args.ref.exists():
            args.ref = Path("tests/samples/resume_styled.pdf")

    if not args.ref.exists():
        print(f"Error: Reference PDF {args.ref} not found.")
        sys.exit(1)

    print(f"Analyzing Reference: {args.ref}")
    ref_layout = extract_layout(args.ref)
    
    # 1. Generate Hints (Oracle)
    print("\n--- EXTRACTING LAYOUT HINTS ---")
    hints = create_layout_hints(ref_layout)
    print(f"Detected Regions: {len(hints['regions'])}")
    
    # Summary of Oracle Regions with canonical labels
    oracle_summary = {"by_label": {}, "by_source": {}, "regions": []}
    CANONICAL_LABELS = {"title", "list", "table", "figure", "textbox", "text"}
    
    for r in hints['regions']:
        label_raw = r['label'].lower()
        label = label_raw if label_raw in CANONICAL_LABELS else "text"
        source = r.get('source', 'unknown')
        
        # By label
        oracle_summary["by_label"][label] = oracle_summary["by_label"].get(label, 0) + 1
        # By source
        oracle_summary["by_source"][source] = oracle_summary["by_source"].get(source, 0) + 1
        # Individual regions (first 20 for brevity)
        if len(oracle_summary["regions"]) < 20:
            oracle_summary["regions"].append({
                "label": label,
                "source": source,
                "confidence": r['confidence'],
                "page": r.get('page', 0)
            })
        
        page_info = f" (p{r['page']})" if 'page' in r else ""
        print(f"  [{label.upper()}] {r['bbox']} conf={r['confidence']:.2f} src={source}{page_info}")
        
    print(f"\nDetected Columns: {len(hints['columns'])}")
    for c in hints['columns']:
        print(f"  Page {c['page']}: {c['count']} columns (boundaries: {c['boundaries']})")
    
    # Save Hints (raw)
    with open(args.hints, "w") as f:
        hints["section_positions"] = [(text, float(y)) for text, y in hints.get("section_positions", [])]
        json.dump(hints, f, indent=2)
    print(f"Hints saved to: {args.hints}")

    # 1.5 Generate Visuals
    if args.visuals:
        print(f"\n--- GENERATING VISUALS ---")
        generate_overlay_images(ref_layout, args.ref, args.vis_dir)

    # Use proper doc_id from parent directory
    doc_id = args.ref.parent.name if args.ref.parent.name != "." else args.ref.stem
    
    # 2. Build Report with new schema
    report = {
        "report_version": "1.2",
        "run_id": args.run_id,
        "doc_id": doc_id,
        "oracle_summary": oracle_summary,
        "columns": [c for c in hints['columns']],
        "diff_health": None,  # VALID, INVALID_DIFF_EXTRACTION, INVALID_MATCH_COVERAGE, etc.
        "layout_quality": {
            "warnings": [],
            "avg_drift": 0.0,
            "max_drift": 0.0,
        },
        "match_stats": {
            "original_blocks": 0,
            "generated_blocks": 0,
            "matched_pairs": 0,
            "match_rate": 0.0,
        },
        "drift_summary": [],
        "policy_trace": [],  # Merged from conversion_report.json later
    }

    if args.gen and args.gen.exists() and args.gen.stat().st_size > 0:
        print(f"\nAnalyzing Generated: {args.gen}")
        try:
            gen_layout = extract_layout(args.gen)
        except Exception as e:
            print(f"ERROR: Failed to extract layout from {args.gen}: {e}")
            report["diff_health"] = "INVALID_GEN_EXTRACTION"
            gen_layout = None
        
        if gen_layout:
            print("\n--- COMPARING LAYOUTS ---")
            diff = compare_layouts(ref_layout, gen_layout)
            
            orig_blocks = diff['original_blocks']
            gen_blocks = diff['generated_blocks']
            matched_pairs = diff.get('matched_pairs', 0)
            
            print(f"Total Block Count: Ref={orig_blocks} vs Gen={gen_blocks}")
            print(f"Matched Pairs: {matched_pairs}")
            
            # Update match stats
            match_rate = matched_pairs / max(1, min(orig_blocks, gen_blocks))
            report["match_stats"] = {
                "original_blocks": orig_blocks,
                "generated_blocks": gen_blocks,
                "matched_pairs": matched_pairs,
                "match_rate": match_rate,  # Keep full precision
            }
            
            # Log unrounded value for clarity
            print(f"Match Rate: {match_rate:.6f}")
            
            # Determine diff_health
            MATCH_COVERAGE_THRESHOLD = 0.2
            threshold = max(10, int(0.1 * orig_blocks))
            
            if gen_blocks < threshold:
                print(f"ERROR: Generated block count ({gen_blocks}) < threshold ({threshold}). Marking diff INVALID.")
                report["diff_health"] = "INVALID_DIFF_EXTRACTION"
                report["diff_error"] = f"generated_blocks={gen_blocks} < {threshold} (10% of {orig_blocks})"
            elif match_rate <= MATCH_COVERAGE_THRESHOLD:  # Use <= to be explicit
                print(f"WARNING: Match rate ({match_rate:.6f}) <= {MATCH_COVERAGE_THRESHOLD}. Drift stats unreliable.")
                report["diff_health"] = "INVALID_MATCH_COVERAGE"
                report["diff_error"] = f"match_rate={match_rate:.6f} <= {MATCH_COVERAGE_THRESHOLD}"
            else:
                report["diff_health"] = "VALID"
            
            # Process drift if we have valid matches
            if diff.get("region_shifts"):
                print("\n--- SHIFTS BY REGION TYPE ---")
                max_drift = 0.0
                total_drift = 0.0
                total_count = 0
                
                for label, data in diff["region_shifts"].items():
                    avg_shift = data["total_y_shift"] / data["count"]
                    max_drift = max(max_drift, avg_shift)
                    total_drift += data["total_y_shift"]
                    total_count += data["count"]
                    
                    print(f"  {label.upper()}: {data['count']} items shifted (Avg Y-shift: {avg_shift:.1f}pt)")
                    report["drift_summary"].append({
                        "type": label,
                        "count": data["count"],
                        "avg_shift": avg_shift
                    })
                
                # Update layout quality
                avg_drift = total_drift / max(1, total_count)
                report["layout_quality"]["avg_drift"] = avg_drift
                report["layout_quality"]["max_drift"] = max_drift
                
                # Add warnings for high drift
                if max_drift > 100:
                    report["layout_quality"]["warnings"].append(f"WARN_HIGH_DRIFT_{int(max_drift)}pt")
                if avg_drift > 50:
                    report["layout_quality"]["warnings"].append(f"WARN_AVG_DRIFT_{int(avg_drift)}pt")
    else:
        print("\nSkipping comparison (no valid generated PDF provided).")
        report["diff_health"] = "SKIPPED_NO_GEN"

    # Save Unified Report
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Unified Report saved to: {args.out}")


if __name__ == "__main__":
    main()
