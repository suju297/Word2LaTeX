#!/usr/bin/env python3
"""Comprehensive validation for Word to LaTeX conversion.

Combines text comparison, image parity, and visual overlay into a single report.
Reports issues by severity: CRITICAL (missing text), WARNING (missing images), INFO (layout drift).

Usage: python scripts/validate_corpus.py [doc_id]
       python scripts/validate_corpus.py --all
"""

import argparse
import sys
import difflib
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF


@dataclass
class ValidationResult:
    doc_id: str
    text_similarity: float = 0.0
    word_coverage: float = 0.0  # Word-set comparison (order-independent)
    image_ref: int = 0
    image_gen: int = 0
    image_missing: int = 0
    page_count_ref: int = 0
    page_count_gen: int = 0
    structural_match: float = 0.0  # DocTR-based structural comparison
    structural_enabled: bool = False
    missing_sentences: list = field(default_factory=list)
    issues: list = field(default_factory=list)
    
    @property
    def status(self) -> str:
        # Use word_coverage as primary metric (order-independent)
        if self.word_coverage < 0.85 or self.image_missing > 10:
            return "CRITICAL"
        elif self.word_coverage < 0.95 or self.image_missing > 0 or self.text_similarity < 0.70:
            return "WARNING"
        else:
            return "PASS"


def extract_text(pdf_path: Path) -> tuple[int, str]:
    """Extract all text from PDF, return (page_count, full_text)."""
    doc = fitz.open(pdf_path)
    pages = doc.page_count
    text = ""
    for page in doc:
        text += " ".join(page.get_text("text").split()) + " "
    doc.close()
    return pages, text.strip()


def count_images(pdf_path: Path) -> int:
    """Count total images in PDF."""
    doc = fitz.open(pdf_path)
    total = 0
    for page in doc:
        total += len(page.get_images(full=True))
    doc.close()
    return total


def find_missing_sentences(ref_text: str, gen_text: str, threshold: float = 0.85) -> list[str]:
    """Find sentences in ref that are missing from gen."""
    # Split into sentence-like chunks
    ref_sentences = [s.strip() for s in ref_text.replace(". ", ".\n").split("\n") if len(s.strip()) > 30]
    
    missing = []
    for sent in ref_sentences:
        # Check if similar sentence exists in gen
        best_ratio = 0.0
        for gen_sent in gen_text.split(". "):
            ratio = difflib.SequenceMatcher(None, sent.lower(), gen_sent.lower()).ratio()
            best_ratio = max(best_ratio, ratio)
            if ratio > threshold:
                break
        
        if best_ratio < threshold:
            truncated = sent[:60] + "..." if len(sent) > 60 else sent
            missing.append(truncated)
    
    return missing[:10]  # Limit to 10


def get_structural_match(doc_dir: Path, cross_page: bool = True) -> float:
    """Get structural match rate using DocTR. Returns 0.0 if unavailable."""
    try:
        # Try to load from cached result first
        cache_path = doc_dir / "structural_compare.json"
        if cache_path.exists():
            import json
            data = json.load(open(cache_path))
            return data.get("match_rate", 0.0)
        
        # Run structural comparison
        # Run structural comparison using Detectron2 (vision_diff)
        from wordtolatex.vision_diff import extract_vision_diffs
        ref_pdf = doc_dir / "ref.pdf"
        gen_pdf = doc_dir / "gen.pdf"
        report = extract_vision_diffs(ref_pdf, gen_pdf, use_cache=True)
        return report.match_rate
    except Exception as e:
        print(f"  [structural] Error: {e}")
        return 0.0


def validate_document(doc_dir: Path, include_structural: bool = False) -> ValidationResult:
    """Validate a single document."""
    doc_id = doc_dir.name
    result = ValidationResult(doc_id=doc_id)
    
    ref_pdf = doc_dir / "ref.pdf"
    gen_pdf = doc_dir / "gen.pdf"
    
    if not ref_pdf.exists() or not gen_pdf.exists():
        result.issues.append("Missing PDF files")
        return result
    
    # Text comparison
    ref_pages, ref_text = extract_text(ref_pdf)
    gen_pages, gen_text = extract_text(gen_pdf)
    
    result.page_count_ref = ref_pages
    result.page_count_gen = gen_pages
    
    matcher = difflib.SequenceMatcher(None, ref_text, gen_text)
    result.text_similarity = matcher.ratio()
    
    # Word coverage (order-independent) - Jaccard-like metric
    # Measures what % of reference words appear in generated
    ref_words = set(ref_text.lower().split())
    gen_words = set(gen_text.lower().split())
    if ref_words:
        # How many ref words are in gen? (recall)
        common = ref_words & gen_words
        result.word_coverage = len(common) / len(ref_words)
    
    # Find missing sentences
    result.missing_sentences = find_missing_sentences(ref_text, gen_text)
    
    # Image comparison
    result.image_ref = count_images(ref_pdf)
    result.image_gen = count_images(gen_pdf)
    result.image_missing = max(0, result.image_ref - result.image_gen)
    
    # Generate issues
    if result.word_coverage < 0.85:
        result.issues.append(f"CRITICAL: Low word coverage ({result.word_coverage*100:.0f}%)")
    elif result.word_coverage < 0.95:
        result.issues.append(f"WARNING: Moderate word coverage ({result.word_coverage*100:.0f}%)")
    
    if result.text_similarity < 0.70:
        result.issues.append(f"INFO: Text order differs significantly ({result.text_similarity*100:.0f}%)")
    
    if result.page_count_ref != result.page_count_gen:
        result.issues.append(f"INFO: Page count mismatch (Ref={result.page_count_ref}, Gen={result.page_count_gen})")
    
    if result.image_missing > 10:
        result.issues.append(f"WARNING: {result.image_missing} images missing (likely PDF artifacts)")
    
    # Structural comparison (optional, slow)
    if include_structural:
        print(f"  Running structural analysis for {doc_id}...")
        result.structural_enabled = True
        result.structural_match = get_structural_match(doc_dir)
        if result.structural_match < 0.7:
            result.issues.append(f"INFO: Low structural match ({result.structural_match*100:.0f}%)")
    
    return result


def print_report(results: list[ValidationResult]):
    """Print comprehensive validation report."""
    print("\n" + "="*70)
    print("COMPREHENSIVE VALIDATION REPORT")
    print("="*70)
    
    # Summary table header
    if any(r.structural_enabled for r in results):
        print("\n{:<25} {:>8} {:>8} {:>8} {:>8} {:>10}".format(
            "Document", "Words%", "Struct%", "Images", "Pages", "Status"))
    else:
        print("\n{:<25} {:>8} {:>8} {:>8} {:>10}".format(
            "Document", "Words%", "Images", "Pages", "Status"))
    print("-"*80)
    
    for r in results:
        img_str = f"{r.image_gen}/{r.image_ref}"
        page_str = f"{r.page_count_gen}/{r.page_count_ref}"
        status_icon = {"PASS": "✅", "WARNING": "⚠️", "CRITICAL": "❌"}.get(r.status, "?")
        
        if r.structural_enabled:
            print("{:<25} {:>7.0f}% {:>7.0f}% {:>8} {:>8} {:>10}".format(
                r.doc_id[:25], r.word_coverage*100, r.structural_match*100, 
                img_str, page_str, f"{status_icon} {r.status}"))
        else:
            print("{:<25} {:>7.0f}% {:>8} {:>8} {:>10}".format(
                r.doc_id[:25], r.word_coverage*100, img_str, page_str, f"{status_icon} {r.status}"))
    
    # Detailed issues
    print("\n" + "-"*70)
    print("DETAILED ISSUES:")
    print("-"*70)
    
    for r in results:
        if r.issues or r.missing_sentences:
            print(f"\n{r.doc_id}:")
            for issue in r.issues:
                print(f"  • {issue}")
            if r.missing_sentences and r.text_similarity < 0.95:
                print("  Missing content samples:")
                for sent in r.missing_sentences[:3]:
                    print(f"    - \"{sent}\"")
    
    # Summary
    pass_count = sum(1 for r in results if r.status == "PASS")
    warn_count = sum(1 for r in results if r.status == "WARNING")
    crit_count = sum(1 for r in results if r.status == "CRITICAL")
    
    print("\n" + "="*70)
    print(f"SUMMARY: {pass_count} PASS, {warn_count} WARNING, {crit_count} CRITICAL")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive conversion validation")
    parser.add_argument("doc_id", nargs="?", help="Document ID or --all for all docs")
    parser.add_argument("--all", action="store_true", help="Validate all corpus documents")
    parser.add_argument("--corpus-root", default="tests/corpus", help="Corpus root directory")
    parser.add_argument("--structural", action="store_true", help="Include DocTR structural comparison (slow)")
    parser.add_argument("--run-feedback", action="store_true", help="Run feedback loop to re-generate documents before validation")
    args = parser.parse_args()
    
    corpus_root = Path(args.corpus_root)
    
    if args.all or args.doc_id is None:
        # Validate all
        doc_dirs = sorted([d for d in corpus_root.iterdir() if d.is_dir()])
    else:
        doc_dirs = [corpus_root / args.doc_id]
        
    # Import feedback controller if needed
    if args.run_feedback:
        from wordtolatex.feedback_loop import FeedbackController
        print("Initializing Feedback Controller...")
    
    results = []
    for doc_dir in doc_dirs:
        ref_pdf = doc_dir / "ref.pdf"
        src_docx = doc_dir / "src.docx"
        
        # Check prerequisites
        if not ref_pdf.exists():
            continue
            
        if args.run_feedback:
            if not src_docx.exists():
                print(f"Skipping {doc_dir.name}: src.docx not found")
                continue
                
            print(f"\nRunning feedback loop for {doc_dir.name}...")
            try:
                controller = FeedbackController(doc_dir)
                controller.run_loop(src_docx, ref_pdf)
            except Exception as e:
                print(f"Feedback loop failed for {doc_dir.name}: {e}")
        
        # Validate
        if (doc_dir / "gen.pdf").exists():
            results.append(validate_document(doc_dir, include_structural=args.structural or args.run_feedback))
    
    print_report(results)
    
    # Exit with error if any critical
    if any(r.status == "CRITICAL" for r in results):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
