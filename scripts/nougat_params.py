"""Nougat Parameter Extraction - Extract formatting hints from Nougat output.

Analyzes Nougat's Markdown/LaTeX output to extract:
- Table structures (column counts, header patterns)
- Equation formats (numbered vs unnumbered)
- List styles (bullet vs numbered)
- Content density metrics
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import re
import json


@dataclass
class TableInfo:
    """Information about a table detected by Nougat."""
    line_number: int
    column_count: int
    row_count: int
    header_row: List[str]
    has_borders: bool = True
    
    def to_dict(self) -> dict:
        return {
            "line_number": self.line_number,
            "column_count": self.column_count,
            "row_count": self.row_count,
            "header_preview": self.header_row[:3],
            "has_borders": self.has_borders
        }


@dataclass
class EquationInfo:
    """Information about equations detected by Nougat."""
    line_number: int
    is_numbered: bool
    latex_content: str
    equation_number: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "line_number": self.line_number,
            "is_numbered": self.is_numbered,
            "equation_number": self.equation_number,
            "preview": self.latex_content[:50] + "..." if len(self.latex_content) > 50 else self.latex_content
        }


@dataclass
class NougatParameters:
    """Extracted parameters from Nougat output."""
    tables: List[TableInfo] = field(default_factory=list)
    equations: List[EquationInfo] = field(default_factory=list)
    heading_pattern: str = "arabic"  # arabic, roman, alpha
    list_count: int = 0
    numbered_list_count: int = 0
    bullet_list_count: int = 0
    total_sections: int = 0
    avg_section_length: float = 0.0
    has_references: bool = False
    reference_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "tables": [t.to_dict() for t in self.tables],
            "equations": [e.to_dict() for e in self.equations],
            "heading_pattern": self.heading_pattern,
            "list_stats": {
                "total": self.list_count,
                "numbered": self.numbered_list_count,
                "bullet": self.bullet_list_count
            },
            "section_stats": {
                "count": self.total_sections,
                "avg_length": self.avg_section_length
            },
            "references": {
                "has_references": self.has_references,
                "count": self.reference_count
            }
        }
    
    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2))


def extract_tables(text: str) -> List[TableInfo]:
    """Extract table information from Nougat output."""
    tables = []
    lines = text.split('\n')
    
    # Pattern for LaTeX tables
    table_pattern = re.compile(r'\\begin\{tabular\}\{([^}]+)\}')
    
    i = 0
    while i < len(lines):
        match = table_pattern.search(lines[i])
        if match:
            col_spec = match.group(1)
            col_count = len(re.sub(r'[^lcr|p]', '', col_spec.replace('|', '')))
            
            # Count rows (look for \\ or \\hline)
            row_count = 0
            header_row = []
            table_content = []
            
            for j in range(i, min(i+100, len(lines))):
                if '\\end{tabular}' in lines[j]:
                    break
                if '\\\\' in lines[j] or '\\hline' in lines[j]:
                    row_count += 1
                    if not header_row and '&' in lines[j]:
                        header_row = [c.strip() for c in lines[j].split('&')]
            
            tables.append(TableInfo(
                line_number=i+1,
                column_count=col_count,
                row_count=row_count,
                header_row=header_row[:5]
            ))
        i += 1
    
    return tables


def extract_equations(text: str) -> List[EquationInfo]:
    """Extract equation information from Nougat output."""
    equations = []
    lines = text.split('\n')
    
    for i, line in enumerate(lines):
        # Numbered equation with \tag{}
        if '\\tag{' in line:
            tag_match = re.search(r'\\tag\{(\d+)\}', line)
            eq_num = int(tag_match.group(1)) if tag_match else None
            
            # Extract equation content
            eq_content = re.sub(r'\\tag\{\d+\}', '', line)
            eq_content = re.sub(r'\\\[|\\\]', '', eq_content).strip()
            
            equations.append(EquationInfo(
                line_number=i+1,
                is_numbered=True,
                latex_content=eq_content,
                equation_number=eq_num
            ))
        # Display equations without tag
        elif line.strip().startswith('\\[') or line.strip().startswith('$$'):
            eq_content = re.sub(r'\\\[|\\\]|\$\$', '', line).strip()
            if eq_content:
                equations.append(EquationInfo(
                    line_number=i+1,
                    is_numbered=False,
                    latex_content=eq_content
                ))
    
    return equations


def detect_heading_pattern(text: str) -> str:
    """Detect the heading numbering pattern used."""
    lines = text.split('\n')
    
    roman_count = 0
    arabic_count = 0
    alpha_count = 0
    
    for line in lines:
        # Check for roman numerals (I, II, III, IV, V)
        if re.match(r'^#{1,4}\s+(I{1,3}|IV|V|VI{0,3}|IX|X)\s', line):
            roman_count += 1
        # Check for arabic (1, 2, 3)
        elif re.match(r'^#{1,4}\s+\d+[\.\s]', line):
            arabic_count += 1
        # Check for alpha (A, B, C)
        elif re.match(r'^#{1,4}\s+[A-Z][\.\s]', line):
            alpha_count += 1
    
    if roman_count >= max(arabic_count, alpha_count):
        return "roman"
    elif alpha_count >= arabic_count:
        return "alpha"
    return "arabic"


def count_lists(text: str) -> tuple:
    """Count numbered and bullet lists."""
    lines = text.split('\n')
    
    numbered = 0
    bullet = 0
    
    for line in lines:
        if re.match(r'^\s*\d+[\.\)]\s', line):
            numbered += 1
        elif re.match(r'^\s*[\*\-\•]\s', line):
            bullet += 1
    
    return numbered, bullet


def count_references(text: str) -> int:
    """Count reference entries."""
    # Match patterns like [1], [2], etc. in References section
    ref_section = re.search(r'References?\n\n([\s\S]+?)(?=\n##|\Z)', text)
    if ref_section:
        refs = re.findall(r'^\s*\[\d+\]', ref_section.group(1), re.MULTILINE)
        return len(refs)
    return 0


def extract_parameters(mmd_path: Path) -> NougatParameters:
    """Extract all parameters from Nougat output."""
    text = mmd_path.read_text(encoding='utf-8')
    
    # Extract various elements
    tables = extract_tables(text)
    equations = extract_equations(text)
    heading_pattern = detect_heading_pattern(text)
    numbered_lists, bullet_lists = count_lists(text)
    ref_count = count_references(text)
    
    # Count sections
    sections = re.findall(r'^#{1,4}\s+.+$', text, re.MULTILINE)
    section_count = len(sections)
    
    # Calculate average section length
    section_lengths = []
    section_starts = [m.start() for m in re.finditer(r'^#{1,4}\s+.+$', text, re.MULTILINE)]
    for i, start in enumerate(section_starts):
        end = section_starts[i+1] if i+1 < len(section_starts) else len(text)
        section_lengths.append(end - start)
    avg_length = sum(section_lengths) / max(1, len(section_lengths))
    
    return NougatParameters(
        tables=tables,
        equations=equations,
        heading_pattern=heading_pattern,
        list_count=numbered_lists + bullet_lists,
        numbered_list_count=numbered_lists,
        bullet_list_count=bullet_lists,
        total_sections=section_count,
        avg_section_length=avg_length,
        has_references=ref_count > 0,
        reference_count=ref_count
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python nougat_params.py ref.mmd")
        sys.exit(1)
    
    mmd_path = Path(sys.argv[1])
    params = extract_parameters(mmd_path)
    
    print(f"\n=== Nougat Parameter Extraction ===")
    print(f"Tables: {len(params.tables)}")
    for t in params.tables:
        print(f"  - Table at line {t.line_number}: {t.column_count} cols, {t.row_count} rows")
    
    print(f"\nEquations: {len(params.equations)}")
    print(f"  - Numbered: {sum(1 for e in params.equations if e.is_numbered)}")
    print(f"  - Unnumbered: {sum(1 for e in params.equations if not e.is_numbered)}")
    
    print(f"\nHeading Pattern: {params.heading_pattern}")
    print(f"Lists: {params.list_count} (numbered: {params.numbered_list_count}, bullet: {params.bullet_list_count})")
    print(f"Sections: {params.total_sections}")
    print(f"References: {params.reference_count}")
    
    # Save
    output_path = mmd_path.parent / "nougat_params.json"
    params.save(output_path)
    print(f"\nParams saved to: {output_path}")
