#!/usr/bin/env python3
"""Compare text content between ref.pdf and gen.pdf for a corpus document.

Uses PyMuPDF to extract text and difflib to find missing content.

Usage: python scripts/text_compare.py <doc_id>
Example: python scripts/text_compare.py IEEE_TEMPLATE
"""

import argparse
import difflib
import sys
from pathlib import Path

import fitz  # PyMuPDF


def extract_text_by_page(pdf_path: Path) -> list[str]:
    """Extract text from each page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        # Normalize whitespace
        text = " ".join(text.split())
        pages.append(text)
    doc.close()
    return pages


def compare_texts(ref_text: str, gen_text: str) -> dict:
    """Compare two text strings and return similarity metrics."""
    # Use SequenceMatcher for similarity ratio
    matcher = difflib.SequenceMatcher(None, ref_text, gen_text)
    ratio = matcher.ratio()
    
    # Find missing chunks (in ref but not in gen)
    ref_words = set(ref_text.split())
    gen_words = set(gen_text.split())
    missing_words = ref_words - gen_words
    extra_words = gen_words - ref_words
    
    # Find missing sentences (more useful)
    ref_sentences = [s.strip() for s in ref_text.replace(".", ".\n").split("\n") if len(s.strip()) > 20]
    gen_sentences = [s.strip() for s in gen_text.replace(".", ".\n").split("\n") if len(s.strip()) > 20]
    
    missing_sentences = []
    for sent in ref_sentences:
        # Check if sentence appears in gen (fuzzy match)
        found = False
        for gen_sent in gen_sentences:
            if difflib.SequenceMatcher(None, sent, gen_sent).ratio() > 0.85:
                found = True
                break
        if not found:
            missing_sentences.append(sent[:80] + "..." if len(sent) > 80 else sent)
    
    return {
        "similarity": ratio,
        "missing_word_count": len(missing_words),
        "extra_word_count": len(extra_words),
        "missing_sentences": missing_sentences[:5],  # Limit to 5
    }


def compare_pdfs(ref_path: Path, gen_path: Path) -> dict:
    """Compare two PDFs page by page."""
    ref_pages = extract_text_by_page(ref_path)
    gen_pages = extract_text_by_page(gen_path)
    
    results = {
        "ref_pages": len(ref_pages),
        "gen_pages": len(gen_pages),
        "page_results": [],
        "overall_similarity": 0.0,
    }
    
    max_pages = max(len(ref_pages), len(gen_pages))
    total_similarity = 0.0
    
    for i in range(max_pages):
        ref_text = ref_pages[i] if i < len(ref_pages) else ""
        gen_text = gen_pages[i] if i < len(gen_pages) else ""
        
        page_result = compare_texts(ref_text, gen_text)
        page_result["page"] = i + 1
        results["page_results"].append(page_result)
        total_similarity += page_result["similarity"]
    
    results["overall_similarity"] = total_similarity / max_pages if max_pages > 0 else 0.0
    
    return results


def print_report(doc_id: str, results: dict):
    """Print a human-readable comparison report."""
    print(f"\n{'='*60}")
    print(f"TEXT PARITY REPORT: {doc_id}")
    print(f"{'='*60}")
    
    print(f"\nPages: Ref={results['ref_pages']}, Gen={results['gen_pages']}")
    print(f"Overall Similarity: {results['overall_similarity']*100:.1f}%")
    
    # Show per-page results
    print(f"\nPer-Page Analysis:")
    for pr in results["page_results"]:
        status = "✅" if pr["similarity"] > 0.95 else ("⚠️" if pr["similarity"] > 0.80 else "❌")
        print(f"  Page {pr['page']}: {pr['similarity']*100:.1f}% {status}")
        
        if pr["missing_sentences"]:
            print(f"    Missing content:")
            for sent in pr["missing_sentences"][:3]:
                print(f"      - \"{sent}\"")
    
    # Overall verdict
    if results["overall_similarity"] >= 0.95:
        print(f"\n✅ EXCELLENT: Text content highly preserved ({results['overall_similarity']*100:.1f}%)")
    elif results["overall_similarity"] >= 0.85:
        print(f"\n⚠️ GOOD: Minor text differences ({results['overall_similarity']*100:.1f}%)")
    else:
        print(f"\n❌ POOR: Significant text missing ({results['overall_similarity']*100:.1f}%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare text content in ref.pdf vs gen.pdf")
    parser.add_argument("doc_id", help="Document ID (e.g. IEEE_TEMPLATE)")
    parser.add_argument("--corpus-root", default="tests/corpus", help="Corpus root directory")
    args = parser.parse_args()
    
    corpus_root = Path(args.corpus_root)
    doc_dir = corpus_root / args.doc_id
    
    if not doc_dir.exists():
        print(f"Error: {doc_dir} not found")
        sys.exit(1)
    
    ref_path = doc_dir / "ref.pdf"
    gen_path = doc_dir / "gen.pdf"
    
    if not ref_path.exists():
        print(f"Error: {ref_path} not found")
        sys.exit(1)
    
    if not gen_path.exists():
        print(f"Error: {gen_path} not found")
        sys.exit(1)
    
    results = compare_pdfs(ref_path, gen_path)
    print_report(args.doc_id, results)
    
    # Exit with error if poor match
    if results["overall_similarity"] < 0.85:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
