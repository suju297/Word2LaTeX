#!/usr/bin/env python3
"""CI Corpus Check - Regenerate PDFs, run region_diff, and evaluate gates.

Supports schema_version: 2 with:
- defaults.labels.<label>.structure/geometry
- docs.<doc_id>.mode: enforce | advisory
- docs.<doc_id>.overrides for targeted exceptions

Exit codes:
    0 - All enforced gates passed (advisory failures are warnings)
    1 - One or more enforced gates failed
    2 - Error during execution
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_gates(gates_path: Path) -> Dict[str, Any]:
    """Load gating thresholds from YAML config."""
    with open(gates_path) as f:
        return yaml.safe_load(f)


def regenerate_pdfs(corpus_root: Path, doc_filter: str = None) -> List[str]:
    """Regenerate PDFs for all corpus documents. Returns list of failures."""
    sys.path.insert(0, str(corpus_root.parent.parent / "src"))
    from wordtolatex.docx_parser.document import parse_docx
    from wordtolatex.generator import generate_latex

    failures = []
    for case_dir in sorted(corpus_root.iterdir()):
        if not case_dir.is_dir() or case_dir.name == "worst_regions":
            continue
            
        if doc_filter and case_dir.name != doc_filter:
            continue

        docx_path = case_dir / "src.docx"
        if not docx_path.exists():
            continue

        print(f"  Regenerating {case_dir.name}...", end=" ")
        try:
            doc = parse_docx(str(docx_path))
            latex = generate_latex(doc)
            (case_dir / "gen.tex").write_text(latex)

            proc = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "gen.tex"],
                cwd=case_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )

            pdf_path = case_dir / "gen.pdf"
            if pdf_path.exists() and pdf_path.stat().st_size > 5000:
                print("[OK]")
            else:
                print("[FAIL - PDF empty/missing]")
                failures.append(case_dir.name)
        except Exception as e:
            print(f"[ERROR] {e}")
            failures.append(case_dir.name)

    return failures


def run_region_diff(corpus_root: Path) -> int:
    """Run region_diff.py and return exit code."""
    scripts_dir = corpus_root.parent.parent / "scripts"
    proc = subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "region_diff.py"),
            "--corpus-root",
            str(corpus_root),
            "--min-region-px",
            "32",
            "--max-worst",
            "20",
        ],
        cwd=corpus_root.parent.parent,
        capture_output=True,
    )
    return proc.returncode


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def get_label_gates(gates: Dict[str, Any], doc_id: str, label: str) -> Dict[str, Any]:
    """Get effective gates for a label, with doc-level overrides applied."""
    defaults = gates.get("defaults", {})
    default_labels = defaults.get("labels", {})
    
    # Start with label defaults (fallback to title if label not defined)
    label_gates = dict(default_labels.get(label, default_labels.get("title", {})))
    
    # Apply doc-level overrides if present
    doc_cfg = gates.get("docs", {}).get(doc_id, {})
    overrides = doc_cfg.get("overrides", {})
    label_overrides = overrides.get("labels", {}).get(label, {})
    
    if label_overrides:
        label_gates = deep_merge(label_gates, label_overrides)
    
    return label_gates


def get_doc_mode(gates: Dict[str, Any], doc_id: str) -> str:
    """Get doc mode: 'enforce' (default) or 'advisory'."""
    doc_cfg = gates.get("docs", {}).get(doc_id, {})
    return doc_cfg.get("mode", "enforce")


def evaluate_gates_v2(corpus_root: Path, gates: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all docs against gates (v2 schema). Returns summary."""
    defaults = gates.get("defaults", {})
    default_coverage = defaults.get("coverage", {})
    
    results = {
        "docs": {},
        "enforced_pass": True,
        "enforced_failures": [],
        "advisory_failures": [],
    }

    for case_dir in sorted(corpus_root.iterdir()):
        if not case_dir.is_dir() or case_dir.name == "worst_regions":
            continue

        stats_path = case_dir / "region_stats.json"
        if not stats_path.exists():
            continue

        data = json.loads(stats_path.read_text())
        doc_id = data.get("doc_id", case_dir.name)
        mode = get_doc_mode(gates, doc_id)
        
        doc_result = {
            "mode": mode,
            "status": "PASS",
            "labels": {},
            "failures": [],
            "warnings": [],  # Classified advisory warnings
            "warn_class": None,  # Summary classification
        }

        # Phase 6 Prep: Object Counting
        try:
            from wordtolatex.docx_parser.document import parse_docx
            doc_ir = parse_docx(case_dir / "src.docx")
            doc_result["object_counts"] = doc_ir.metadata.get("object_counts", {})
        except Exception:
            pass

        # Page count check (critical for enforced docs)
        ref_pages = data.get("ref_pages", 1)
        gen_pages = data.get("gen_pages", 1)
        page_diff = abs(gen_pages - ref_pages)
        
        if page_diff >= 2:
            # Catastrophic layout blowup - skip expensive drift computation
            doc_result["status"] = "FAIL"
            doc_result["warn_class"] = "WARN_CATASTROPHIC_LAYOUT"
            doc_result["failures"].append(f"PAGE_COUNT: gen={gen_pages} != ref={ref_pages} (diff={page_diff})")
            doc_result["failures"].append(f"SKIP_DRIFT: page_diff={page_diff} (drift invalid across mismatched pages)")
            results["docs"][doc_id] = doc_result
            if mode == "enforce":
                results["enforced_pass"] = False
                results["enforced_failures"].append(doc_id)
            else:
                results["advisory_failures"].append(doc_id)
            continue  # Skip expensive per-label analysis
        elif page_diff == 1:
            doc_result["status"] = "FAIL"
            doc_result["warn_class"] = "WARN_PAGE_COUNT"
            doc_result["failures"].append(f"PAGE_COUNT: gen={gen_pages} != ref={ref_pages}")

        # Evaluate per-label
        by_label = data.get("by_label", {})
        regions = data.get("regions", [])

        for label, stats in by_label.items():
            label_gates = get_label_gates(gates, doc_id, label)
            structure_gates = label_gates.get("structure", {})
            geometry_gates = label_gates.get("geometry", {})
            
            label_result = {"status": "PASS", "violations": []}
            
            # Check min_regions
            min_regions = label_gates.get("min_regions", 1)
            scored_count = stats.get("count_scored", 0)
            if scored_count < min_regions:
                # Not enough regions to evaluate meaningfully - skip
                label_result["status"] = "SKIP"
                label_result["violations"].append(f"regions={scored_count} < min={min_regions}")
                doc_result["labels"][label] = label_result
                continue

            # Structure gates (aggregate from regions)
            label_regions = [r for r in regions if r.get("label") == label]
            hit_count = sum(1 for r in label_regions if r.get("hit_boundary"))
            blank_count = sum(1 for r in label_regions if r.get("blank_gen"))
            missing_count = sum(1 for r in label_regions if r.get("status") == "MISSING_PAGE")

            hit_max = structure_gates.get("hit_boundary_max", 0)
            blank_max = structure_gates.get("blank_gen_max", 0)
            missing_max = structure_gates.get("missing_page_max", 0)

            if hit_count > hit_max:
                label_result["status"] = "FAIL"
                label_result["violations"].append(f"hit_boundary={hit_count} > {hit_max}")

            if blank_count > blank_max:
                label_result["status"] = "FAIL"
                label_result["violations"].append(f"blank_gen={blank_count} > {blank_max}")

            if missing_count > missing_max:
                label_result["status"] = "FAIL"
                label_result["violations"].append(f"missing_page={missing_count} > {missing_max}")

            # Geometry gates (use abs for median)
            drift = stats.get("drift", {})
            median_dy = abs(drift.get("median_dy") or 0)
            median_dx = abs(drift.get("median_dx") or 0)

            max_dy = geometry_gates.get("median_abs_dy_max", 999)
            max_dx = geometry_gates.get("median_abs_dx_max", 999)

            if median_dy > max_dy:
                label_result["status"] = "FAIL"
                label_result["violations"].append(f"|median_dy|={median_dy:.1f} > {max_dy}")

            if median_dx > max_dx:
                label_result["status"] = "FAIL"
                label_result["violations"].append(f"|median_dx|={median_dx:.1f} > {max_dx}")

            # Coverage gate
            cov = stats.get("coverage_effective", 1.0)
            min_cov = label_gates.get("coverage_min_effective", default_coverage.get("min_effective", 0.70))
            if cov < min_cov:
                label_result["status"] = "FAIL"
                label_result["violations"].append(f"coverage={cov:.2f} < {min_cov}")

            doc_result["labels"][label] = label_result
            if label_result["status"] == "FAIL":
                doc_result["status"] = "FAIL"
                doc_result["failures"].extend(
                    [f"{label}: {v}" for v in label_result["violations"]]
                )

        # Classify warning type if no warn_class set yet
        if doc_result["status"] == "FAIL" and not doc_result.get("warn_class"):
            # Parse failure types more carefully
            has_geometry_fail = any(
                "median_dy" in f or "median_dx" in f 
                for f in doc_result["failures"]
            )
            # Figure/blank issues = true structure drift
            has_figure_fail = any(
                "figure:" in f for f in doc_result["failures"]
            )
            has_blank_gen = any(
                "blank_gen" in f for f in doc_result["failures"]
            )
            # Title/list hit_boundary = layout mismatch (not figure problem)
            has_text_hit_boundary = any(
                ("title:" in f or "list:" in f) and "hit_boundary" in f
                for f in doc_result["failures"]
            )
            
            if has_figure_fail or has_blank_gen:
                doc_result["warn_class"] = "WARN_STRUCTURE_DRIFT"
            elif has_geometry_fail or has_text_hit_boundary:
                doc_result["warn_class"] = "WARN_STRUCTURAL_LAYOUT_MISMATCH"

        results["docs"][doc_id] = doc_result
        
        if doc_result["status"] == "FAIL":
            if mode == "enforce":
                results["enforced_pass"] = False
                results["enforced_failures"].append(doc_id)
            else:
                results["advisory_failures"].append(doc_id)

    return results


def print_report_v2(results: Dict[str, Any]) -> None:
    """Print a human-readable gate evaluation report (v2)."""
    print("\n" + "=" * 60)
    print("CORPUS GATE EVALUATION REPORT (v2)")
    print("=" * 60)

    for doc_id, doc_result in results["docs"].items():
        status = doc_result["status"]
        mode = doc_result.get("mode", "enforce")
        warn_class = doc_result.get("warn_class", "")
        
        if status == "PASS":
            icon = "✅"
        elif mode == "advisory":
            icon = "⚠️"  # Advisory fail = warning
        else:
            icon = "❌"  # Enforced fail = error
            
        mode_tag = f"[{mode.upper()}]"
        class_tag = f" ({warn_class})" if warn_class else ""
        print(f"\n{icon} {doc_id} {mode_tag}: {status}{class_tag}")

        cts = doc_result.get("object_counts", {})
        if any(cts.values()):
            print(f"   [OBJECTS] textboxes={cts.get('textboxes')} shapes={cts.get('shapes')} images={cts.get('images')} v_images={cts.get('v_images')}")

        if doc_result.get("failures"):
            for f in doc_result["failures"]:
                print(f"   - {f}")

    print("\n" + "-" * 60)
    
    if results["enforced_pass"]:
        if results["advisory_failures"]:
            print(f"🏆 ALL ENFORCED GATES PASSED")
            print(f"⚠️  Advisory warnings: {', '.join(results['advisory_failures'])}")
        else:
            print("🏆 ALL GATES PASSED")
    else:
        print(f"❌ ENFORCED GATES FAILED for: {', '.join(results['enforced_failures'])}")
        if results["advisory_failures"]:
            print(f"⚠️  Advisory warnings: {', '.join(results['advisory_failures'])}")
    
    print("-" * 60)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus-root",
        type=str,
        default="tests/corpus",
        help="Path to corpus directory",
    )
    ap.add_argument(
        "--skip-regen",
        action="store_true",
        help="Skip PDF regeneration (use existing gen.pdf, still run diff + gates)",
    )
    ap.add_argument(
        "--doc",
        type=str,
        help="Run only for specific document ID",
    )
    args = ap.parse_args()

    corpus_root = Path(args.corpus_root)
    gates_path = corpus_root / "corpus_gates.yaml"

    if not corpus_root.exists():
        print(f"ERROR: Corpus root not found: {corpus_root}")
        return 2

    if not gates_path.exists():
        print(f"ERROR: Gates config not found: {gates_path}")
        return 2

    gates = load_gates(gates_path)

    # Step 1: Regenerate PDFs
    if not args.skip_regen:
        print("\n[1/3] Regenerating PDFs...")
        failures = regenerate_pdfs(corpus_root, doc_filter=args.doc)
        if failures:
            print(f"WARNING: PDF generation failed for: {failures}")
    else:
        print("\n[1/3] Skipping PDF regeneration (--skip-regen)")

    # Step 2: Run region_diff (always, unless we add a separate flag later)
    print("\n[2/3] Running region_diff...")
    rc = run_region_diff(corpus_root)
    if rc != 0:
        print(f"WARNING: region_diff exited with code {rc}")

    # Step 3: Evaluate gates
    print("\n[3/3] Evaluating gates...")
    results = evaluate_gates_v2(corpus_root, gates)

    # Save results
    results_path = corpus_root / "gate_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Results saved to: {results_path}")

    # Print report
    print_report_v2(results)

    return 0 if results["enforced_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
