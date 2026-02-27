"""Run Nougat analysis on full corpus.

Runs structural comparison and parameter extraction on all corpus documents.
"""

from pathlib import Path
import json
import sys
import subprocess

# Add scripts to path
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from nougat_compare import compare_structures, extract_sections_from_markdown, extract_sections_from_latex
from nougat_params import extract_parameters


def run_nougat_on_pdf(pdf_path: Path, output_dir: Path) -> Path | None:
    """Run Nougat on a PDF file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    mmd_files = list(output_dir.glob("*.mmd"))
    if mmd_files:
        return mmd_files[0]
    
    try:
        result = subprocess.run(
            ["uv", "run", "nougat", str(pdf_path), "-o", str(output_dir), "--no-skipping"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=Path(__file__).parent.parent
        )
        mmd_files = list(output_dir.glob("*.mmd"))
        return mmd_files[0] if mmd_files else None
    except Exception as e:
        print(f"  Error running Nougat: {e}")
        return None


def analyze_corpus(corpus_root: Path) -> dict:
    """Analyze all documents in corpus with Nougat."""
    results = {}
    
    doc_dirs = sorted([d for d in corpus_root.iterdir() if d.is_dir()])
    
    for doc_dir in doc_dirs:
        doc_id = doc_dir.name
        ref_pdf = doc_dir / "ref.pdf"
        gen_tex = doc_dir / "gen.tex"
        
        if not ref_pdf.exists():
            print(f"Skipping {doc_id}: no ref.pdf")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {doc_id}")
        print(f"{'='*60}")
        
        result = {
            "doc_id": doc_id,
            "nougat_processed": False,
            "structural_comparison": None,
            "parameters": None,
            "errors": []
        }
        
        # Step 1: Run Nougat
        nougat_dir = doc_dir / "nougat_output"
        print(f"  Running Nougat on {ref_pdf.name}...")
        mmd_path = run_nougat_on_pdf(ref_pdf, nougat_dir)
        
        if not mmd_path:
            result["errors"].append("Nougat failed to process PDF")
            results[doc_id] = result
            continue
        
        result["nougat_processed"] = True
        print(f"  ✓ Nougat output: {mmd_path.name}")
        
        # Step 2: Extract parameters
        try:
            params = extract_parameters(mmd_path)
            result["parameters"] = {
                "tables": len(params.tables),
                "equations": len(params.equations),
                "heading_pattern": params.heading_pattern,
                "sections": params.total_sections,
                "lists": params.list_count,
                "references": params.reference_count
            }
            print(f"  ✓ Parameters: {params.total_sections} sections, {len(params.tables)} tables, {len(params.equations)} equations")
        except Exception as e:
            result["errors"].append(f"Parameter extraction failed: {e}")
        
        # Step 3: Structural comparison (if gen.tex exists)
        if gen_tex.exists():
            try:
                report = compare_structures(mmd_path, gen_tex)
                result["structural_comparison"] = {
                    "ref_sections": len(report.ref_sections),
                    "gen_sections": len(report.gen_sections),
                    "matched": len(report.matched_sections),
                    "similarity": len(report.matched_sections) / max(1, len(report.ref_sections)),
                    "missing_in_gen": len(report.missing_in_gen),
                    "extra_in_gen": len(report.extra_in_gen),
                    "heading_mismatches": len(report.heading_mismatches)
                }
                print(f"  ✓ Structure: {result['structural_comparison']['similarity']:.1%} match, {len(report.heading_mismatches)} level mismatches")
            except Exception as e:
                result["errors"].append(f"Structural comparison failed: {e}")
        else:
            print(f"  ⚠ No gen.tex found, skipping structural comparison")
        
        results[doc_id] = result
    
    return results


def print_summary(results: dict):
    """Print summary of corpus analysis."""
    print("\n" + "="*80)
    print("CORPUS NOUGAT ANALYSIS SUMMARY")
    print("="*80)
    
    total = len(results)
    processed = sum(1 for r in results.values() if r["nougat_processed"])
    
    print(f"\nDocuments: {processed}/{total} processed with Nougat")
    
    # Structural comparison summary
    structural_results = [r for r in results.values() if r.get("structural_comparison")]
    if structural_results:
        avg_similarity = sum(r["structural_comparison"]["similarity"] for r in structural_results) / len(structural_results)
        total_mismatches = sum(r["structural_comparison"]["heading_mismatches"] for r in structural_results)
        
        print(f"\nStructural Comparison:")
        print(f"  Average similarity: {avg_similarity:.1%}")
        print(f"  Total heading mismatches: {total_mismatches}")
        
        print(f"\n  Per-document breakdown:")
        for r in structural_results:
            doc_id = r["doc_id"]
            sc = r["structural_comparison"]
            print(f"    {doc_id}:")
            print(f"      Similarity: {sc['similarity']:.1%} ({sc['matched']}/{sc['ref_sections']} sections)")
            print(f"      Heading mismatches: {sc['heading_mismatches']}")
    
    # Parameter summary
    param_results = [r for r in results.values() if r.get("parameters")]
    if param_results:
        total_tables = sum(r["parameters"]["tables"] for r in param_results)
        total_equations = sum(r["parameters"]["equations"] for r in param_results)
        
        print(f"\nParameters Extracted:")
        print(f"  Total tables detected: {total_tables}")
        print(f"  Total equations detected: {total_equations}")
        
        print(f"\n  Per-document breakdown:")
        for r in param_results:
            doc_id = r["doc_id"]
            p = r["parameters"]
            print(f"    {doc_id}: {p['sections']} sections, {p['tables']} tables, {p['equations']} equations, {p['heading_pattern']} numbering")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Nougat analysis on corpus")
    parser.add_argument("--corpus-root", type=Path, default=Path("tests/corpus"))
    parser.add_argument("--output", type=Path, default=Path("nougat_corpus_report.json"))
    
    args = parser.parse_args()
    
    results = analyze_corpus(args.corpus_root)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_summary(results)
    print(f"\nFull report saved to: {args.output}")
