"""Compare Layout Models against Gemini Oracle.
Usage:
    uv run python scripts/compare_models.py <path/to/document.pdf> [--pages 1]
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from wordtolatex.model_arbitrator import ModelArbitrator, ArbitrationResult

def main():
    parser = argparse.ArgumentParser(description="Compare Layout Models")
    parser.add_argument("pdf_path", type=str, help="Path to PDF document")
    parser.add_argument("--pages", type=int, default=1, help="Number of pages to verify (default: 1)")
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found")
        sys.exit(1)
        
    print(f"Initializing Arbitrator...")
    arb = ModelArbitrator()
    
    report = {
        "document": str(pdf_path),
        "pages": []
    }
    
    for i in range(args.pages):
        print(f"\n--- Verifying Page {i} ---")
        try:
            # Verify page (0-indexed)
            results = arb.verify_page(pdf_path, i)
            
            page_data = {
                "page": i,
                "scores": {}
            }
            
            for model_name, res in results.items():
                print(f"Model: {model_name}")
                print(f"  Precision: {res.precision:.2f}")
                print(f"  Recall:    {res.recall:.2f}")
                print(f"  F1 Score:  {res.f1:.2f}")
                print(f"  TP: {len(res.matches)}, FP: {len(res.false_positives)}, FN: {len(res.false_negatives)}")
                
                # Convert dataclass to dict for JSON serialization
                # We need to serialize BBox objects specifically if we want full detail
                # For summary, just scores are fine
                page_data["scores"][model_name] = {
                    "precision": res.precision,
                    "recall": res.recall,
                    "f1": res.f1,
                    "tp_count": len(res.matches),
                    "fp_count": len(res.false_positives),
                    "fn_count": len(res.false_negatives)
                }
            
            report["pages"].append(page_data)
            
        except Exception as e:
            print(f"Error processing page {i}: {e}")
            import traceback
            traceback.print_exc()
            
    # Save report
    output_path = pdf_path.parent / "model_verification_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nReport saved to {output_path}")

if __name__ == "__main__":
    main()
