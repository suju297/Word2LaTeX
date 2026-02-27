"""Nougat-based Structural Comparison Tool.

Compares Nougat's extracted Markdown (ref.mmd) with our generated LaTeX (gen.tex)
to identify structural differences: heading hierarchy, section order, content gaps.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import re
import json
import difflib


@dataclass
class Section:
    """A section extracted from document."""
    level: int  # 1=H1, 2=H2, 3=H3
    title: str
    content_preview: str  # First ~100 chars of content
    line_number: int
    
    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "title": self.title,
            "content_preview": self.content_preview[:100],
            "line_number": self.line_number
        }


@dataclass  
class StructuralReport:
    """Comparison report between ref and gen structures."""
    ref_sections: List[Section]
    gen_sections: List[Section]
    matched_sections: List[Tuple[Section, Section, float]]  # (ref, gen, similarity)
    missing_in_gen: List[Section]
    extra_in_gen: List[Section]
    heading_mismatches: List[Dict]  # Level differences
    
    def to_dict(self) -> dict:
        return {
            "ref_section_count": len(self.ref_sections),
            "gen_section_count": len(self.gen_sections),
            "matched_count": len(self.matched_sections),
            "missing_in_gen": [s.to_dict() for s in self.missing_in_gen],
            "extra_in_gen": [s.to_dict() for s in self.extra_in_gen],
            "heading_mismatches": self.heading_mismatches,
            "structure_similarity": len(self.matched_sections) / max(1, len(self.ref_sections))
        }
    
    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2))


def extract_sections_from_markdown(text: str) -> List[Section]:
    """Extract sections from Nougat's Markdown output."""
    sections = []
    lines = text.split('\n')
    
    current_content = []
    
    for i, line in enumerate(lines):
        # Match Markdown headings: #, ##, ###
        match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            
            # Get content preview (next non-empty lines)
            preview_lines = []
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].strip() and not lines[j].startswith('#'):
                    preview_lines.append(lines[j].strip())
                if len(' '.join(preview_lines)) > 100:
                    break
            
            sections.append(Section(
                level=level,
                title=title,
                content_preview=' '.join(preview_lines),
                line_number=i+1
            ))
    
    return sections


def extract_sections_from_latex(text: str) -> List[Section]:
    """Extract sections from LaTeX output."""
    sections = []
    lines = text.split('\n')
    
    # Match \section{}, \subsection{}, \subsubsection{}
    section_patterns = [
        (r'\\section\*?\{([^}]+)\}', 1),
        (r'\\subsection\*?\{([^}]+)\}', 2),
        (r'\\subsubsection\*?\{([^}]+)\}', 3),
    ]
    
    for i, line in enumerate(lines):
        for pattern, level in section_patterns:
            match = re.search(pattern, line)
            if match:
                title = match.group(1).strip()
                
                # Get content preview
                preview_lines = []
                for j in range(i+1, min(i+5, len(lines))):
                    clean_line = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', lines[j])
                    clean_line = re.sub(r'[\\{}]', '', clean_line).strip()
                    if clean_line and not re.match(r'^\\(sub)*section', lines[j]):
                        preview_lines.append(clean_line)
                    if len(' '.join(preview_lines)) > 100:
                        break
                
                sections.append(Section(
                    level=level,
                    title=title,
                    content_preview=' '.join(preview_lines),
                    line_number=i+1
                ))
                break  # Only match first pattern per line
    
    return sections


def normalize_title(title: str) -> str:
    """Normalize title for comparison."""
    # Remove numbering, punctuation, lowercase
    title = re.sub(r'^[\d.]+\s*', '', title)  # Remove leading numbers
    title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
    return title.lower().strip()


def match_sections(ref_sections: List[Section], gen_sections: List[Section]) -> StructuralReport:
    """Match sections between reference and generated documents."""
    matched = []
    used_gen = set()
    
    for ref_sec in ref_sections:
        ref_title_norm = normalize_title(ref_sec.title)
        best_match = None
        best_score = 0.0
        best_idx = -1
        
        for idx, gen_sec in enumerate(gen_sections):
            if idx in used_gen:
                continue
                
            gen_title_norm = normalize_title(gen_sec.title)
            
            # Calculate title similarity
            score = difflib.SequenceMatcher(None, ref_title_norm, gen_title_norm).ratio()
            
            if score > best_score and score > 0.5:  # Threshold
                best_match = gen_sec
                best_score = score
                best_idx = idx
        
        if best_match:
            matched.append((ref_sec, best_match, best_score))
            used_gen.add(best_idx)
    
    # Find missing and extra
    matched_ref_titles = {normalize_title(m[0].title) for m in matched}
    matched_gen_titles = {normalize_title(m[1].title) for m in matched}
    
    missing = [s for s in ref_sections if normalize_title(s.title) not in matched_ref_titles]
    extra = [s for s in gen_sections if normalize_title(s.title) not in matched_gen_titles]
    
    # Find heading level mismatches
    mismatches = []
    for ref_sec, gen_sec, score in matched:
        if ref_sec.level != gen_sec.level:
            mismatches.append({
                "title": ref_sec.title,
                "ref_level": ref_sec.level,
                "gen_level": gen_sec.level,
                "expected": f"H{ref_sec.level}",
                "actual": f"H{gen_sec.level}"
            })
    
    return StructuralReport(
        ref_sections=ref_sections,
        gen_sections=gen_sections,
        matched_sections=matched,
        missing_in_gen=missing,
        extra_in_gen=extra,
        heading_mismatches=mismatches
    )


def compare_structures(ref_mmd_path: Path, gen_tex_path: Path) -> StructuralReport:
    """Main entry: compare Nougat mmd with generated LaTeX."""
    ref_text = ref_mmd_path.read_text(encoding='utf-8')
    gen_text = gen_tex_path.read_text(encoding='utf-8')
    
    ref_sections = extract_sections_from_markdown(ref_text)
    gen_sections = extract_sections_from_latex(gen_text)
    
    return match_sections(ref_sections, gen_sections)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python nougat_compare.py ref.mmd gen.tex")
        sys.exit(1)
    
    ref_path = Path(sys.argv[1])
    gen_path = Path(sys.argv[2])
    
    report = compare_structures(ref_path, gen_path)
    
    print(f"\n=== Structural Comparison ===")
    print(f"Reference sections: {len(report.ref_sections)}")
    print(f"Generated sections: {len(report.gen_sections)}")
    print(f"Matched: {len(report.matched_sections)}")
    print(f"Structure similarity: {len(report.matched_sections)/max(1,len(report.ref_sections)):.1%}")
    
    if report.missing_in_gen:
        print(f"\nMissing in Generated ({len(report.missing_in_gen)}):")
        for s in report.missing_in_gen[:5]:
            print(f"  - [{s.level}] {s.title}")
    
    if report.extra_in_gen:
        print(f"\nExtra in Generated ({len(report.extra_in_gen)}):")
        for s in report.extra_in_gen[:5]:
            print(f"  - [{s.level}] {s.title}")
    
    if report.heading_mismatches:
        print(f"\nHeading Level Mismatches ({len(report.heading_mismatches)}):")
        for m in report.heading_mismatches[:5]:
            print(f"  - '{m['title']}': {m['expected']} → {m['actual']}")
    
    # Save report
    output_path = gen_path.parent / "structural_report.json"
    report.save(output_path)
    print(f"\nReport saved to: {output_path}")
