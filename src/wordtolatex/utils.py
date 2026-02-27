"""Utility functions for Word to LaTeX conversion."""

import subprocess
from pathlib import Path
import logging
import shutil
import os

logger = logging.getLogger(__name__)

def compile_latex(tex_path: Path, output_pdf: Path = None) -> bool:
    """Compile LaTeX to PDF using xelatex.
    
    Args:
        tex_path: Path to .tex file
        output_pdf: Path to output .pdf file (default: same logic as xelatex)
    
    Returns:
        bool: True if compilation succeeded
    """
    tex_path = Path(tex_path).resolve()
    output_dir = tex_path.parent
    
    if output_pdf is None:
        output_pdf = tex_path.with_suffix(".pdf")
        
    # Xelatex always outputs to output_dir with same basename
    expected_pdf = output_dir / tex_path.with_suffix(".pdf").name
    
    cmd = [
        "xelatex",
        "-interaction=nonstopmode",
        "-output-directory", ".",
        tex_path.name
    ]
    
    try:
        # Run twice for cross-references if needed, but once is usually fine for feedback loop
        subprocess.run(
            cmd, 
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=60,
            cwd=output_dir # Run in output dir to resolve assets
        )
        
        # If output_pdf is different from default location, move it
        if output_pdf != expected_pdf and expected_pdf.exists():
            shutil.move(expected_pdf, output_pdf)
            
        return output_pdf.exists()
        
    except Exception as e:
        logger.error(f"Unexpected error during compilation: {e}")
        return False


def escape_latex(text: str) -> str:
    """Escape special LaTeX characters in text.

    Handles: & % $ # _ { } ~ ^ \\
    Normalizes common Unicode punctuation/spaces to LaTeX-safe equivalents.
    Prevents double-escaping of generated commands.
    """
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Normalize punctuation that is safe to inline as ASCII.
    text = text.replace("\u2013", "--")  # en-dash
    text = text.replace("\u2014", "---")  # em-dash
    text = text.replace("\u2212", "-")  # minus sign

    # Replace Unicode characters that require LaTeX commands after escaping.
    special_macros = {
        "\u00a0": "~",  # non-breaking space
        "\u2003": r"\quad ",
        "\u2002": r"\enspace ",
        "\u2022": r"\ensuremath{\bullet}",
        "\u2192": r"\ensuremath{\rightarrow}",
        "\u2190": r"\ensuremath{\leftarrow}",
    }
    token_map: dict[str, str] = {}
    for idx, (char, repl) in enumerate(special_macros.items()):
        if char in text:
            token = f"<LATEX_SPECIAL_{idx}>"
            token_map[token] = repl
            text = text.replace(char, token)

    # 1. Replace backslash with a temporary token to protect it from subsequent escapes.
    # Use a token unlikely to be in the text.
    SLASH_TOKEN = "<LATEX_BACKSLASH_TOKEN>"
    text = text.replace("\\", SLASH_TOKEN)

    # 2. Escape other special characters.
    chars = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }

    for char, repl in chars.items():
        text = text.replace(char, repl)

    # 3. Restore backslash as safe latex command.
    # Note: \textbackslash{} contains {} which should NOT be escaped again.
    text = text.replace(SLASH_TOKEN, "\\textbackslash{}")

    # 4. Restore LaTeX macro replacements.
    for token, repl in token_map.items():
        text = text.replace(token, repl)

    return text
