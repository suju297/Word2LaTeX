#!/usr/bin/env python
"""Run corpus validation: convert DOCX→PDF, run pipeline, compile LaTeX, compare layouts."""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
import json
from datetime import datetime

DATASET_DIR = Path("dataset").resolve()
OUTPUT_ROOT = Path("tests/corpus").resolve()
MIN_PDF_SIZE = 5 * 1024  # 5KB minimum for valid gen.pdf


def get_run_id() -> str:
    """Generate run ID from timestamp + optional git hash."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"{ts}_{git_hash}"
    except Exception:
        return ts


def convert_docx_to_pdf(docx_path: Path, pdf_path: Path):
    """Convert DOCX to PDF using Word (preferred) or LibreOffice headless."""
    if pdf_path.exists():
        print(f"   [Cache] PDF already exists: {pdf_path}")
        return

    # Try Word first (better fidelity)
    word_app = Path("/Applications/Microsoft Word.app")
    if word_app.exists():
        print(f"   [Word] Converting {docx_path.name} to PDF...")
        abs_docx = str(docx_path.absolute())
        abs_pdf = str(pdf_path.absolute())
        
        script = f'''
        tell application "Microsoft Word"
            set theDoc to open "{abs_docx}"
            save as theDoc file name "{abs_pdf}" file format format PDF
            close theDoc saving no
        end tell
        '''
        
        try:
            subprocess.run(["osascript", "-e", script], check=True, capture_output=True, timeout=60)
            if pdf_path.exists():
                return  # Success with Word
        except Exception as e:
            print(f"   [Word] Failed: {e}, falling back to LibreOffice...")

    # Fallback to LibreOffice
    print(f"   [LibreOffice] Converting {docx_path.name} to PDF...")
    
    soffice = shutil.which("soffice")
    if not soffice:
        mac_soffice = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
        if mac_soffice.exists():
            soffice = str(mac_soffice)
            
    if not soffice:
        raise FileNotFoundError("Neither Word nor LibreOffice found.")
        
    cmd = [
        soffice, "--headless", "--convert-to", "pdf",
        "--outdir", str(pdf_path.parent),
        str(docx_path)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
        
    expected_out = pdf_path.parent / (docx_path.stem + ".pdf")
    if expected_out.exists() and expected_out != pdf_path:
        expected_out.replace(pdf_path)
    elif not expected_out.exists() and not pdf_path.exists():
        raise RuntimeError("PDF conversion failed.")


def run_command(cmd_list, cwd=None):
    subprocess.run(cmd_list, check=True, cwd=cwd)


def sanitize_tex_for_pdflatex(tex_path: Path) -> None:
    """Sanitize LaTeX file for pdfLaTeX compatibility.
    
    Replaces problematic Unicode characters with LaTeX-safe equivalents.
    """
    # Character replacements for pdfLaTeX compatibility
    UNICODE_REPLACEMENTS = {
        '│': '|',  # U+2502 box drawing vertical
        '─': '-',  # U+2500 box drawing horizontal
        '┌': '+',  # U+250C box drawing corner
        '┐': '+',  # U+2510 box drawing corner
        '└': '+',  # U+2514 box drawing corner
        '┘': '+',  # U+2518 box drawing corner
        '├': '+',  # U+251C box drawing
        '┤': '+',  # U+2524 box drawing
        '┬': '+',  # U+252C box drawing
        '┴': '+',  # U+2534 box drawing
        '┼': '+',  # U+253C box drawing
        '•': r'\textbullet{}',  # bullet
        '→': r'$\rightarrow$',  # arrow
        '←': r'$\leftarrow$',
        '↔': r'$\leftrightarrow$',
        '≤': r'$\leq$',
        '≥': r'$\geq$',
        '≠': r'$\neq$',
        '×': r'$\times$',
        '÷': r'$\div$',
        '±': r'$\pm$',
        '∞': r'$\infty$',
        '…': r'\ldots{}',
        ''': "'",
        ''': "'",
        '"': "``",
        '"': "''",
        '–': '--',  # en-dash
        '—': '---',  # em-dash
        '\u00A0': ' ',  # non-breaking space
        '\u200B': '',  # zero-width space
        '\uFEFF': '',  # BOM
    }
    
    content = tex_path.read_text(encoding='utf-8')
    original = content
    
    for char, replacement in UNICODE_REPLACEMENTS.items():
        content = content.replace(char, replacement)
    
    if content != original:
        tex_path.write_text(content, encoding='utf-8')
        print(f"   [Sanitize] Fixed Unicode characters in {tex_path.name}")


def compile_latex(gen_tex: Path, case_dir: Path, run_id: str, engine: str = "xelatex") -> dict:
    """Compile LaTeX with proper error handling. Returns build metadata.
    
    Args:
        gen_tex: Path to the .tex file
        case_dir: Output directory
        run_id: Run identifier for tracking
        engine: 'xelatex' (default, better Unicode) or 'pdflatex'
    """
    build_meta = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "latex_exit_code": None,
        "latex_first_error": None,
        "gen_pdf_size": 0,
        "engine": engine,
    }
    
    gen_pdf = case_dir / "gen.pdf"
    latex_log = case_dir / "latex.log"
    latex_err = case_dir / "latex.err"
    
    # Check engine availability
    latex_cmd = shutil.which(engine)
    if not latex_cmd:
        # Fallback to pdflatex if xelatex not found
        if engine == "xelatex":
            latex_cmd = shutil.which("pdflatex")
            if latex_cmd:
                engine = "pdflatex"
                build_meta["engine"] = engine
                print(f"   [LaTeX] xelatex not found, falling back to pdflatex")
    
    if not latex_cmd:
        print(f"   [LaTeX] No LaTeX engine found, skipping compilation.")
        build_meta["latex_first_error"] = "No LaTeX engine found"
        build_meta["latex_exit_code"] = 1
        return build_meta
    
    # Sanitize for pdfLaTeX (XeLaTeX handles Unicode natively)
    if engine == "pdflatex":
        sanitize_tex_for_pdflatex(gen_tex)
    
    print(f"   [LaTeX] Compiling with {engine}...")
    
    # Run with flags that surface errors
    result = subprocess.run(
        [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={case_dir}",
            str(gen_tex)
        ],
        cwd=case_dir,
        capture_output=True,
        text=True
    )
    
    build_meta["latex_exit_code"] = result.returncode
    
    # Save stdout as log
    with open(latex_log, "w") as f:
        f.write(result.stdout)
    
    # Save stderr as err
    with open(latex_err, "w") as f:
        f.write(result.stderr)
    
    # Rename output.pdf to gen.pdf if needed
    out_pdf = case_dir / "output.pdf"
    if out_pdf.exists():
        out_pdf.replace(gen_pdf)
    
    # Check output size
    if gen_pdf.exists():
        build_meta["gen_pdf_size"] = gen_pdf.stat().st_size
    
    # HARD FAILURE: Treat 0-byte or too-small PDF as failure even if exit code was 0
    if build_meta["gen_pdf_size"] < MIN_PDF_SIZE:
        if build_meta["latex_exit_code"] == 0:
            build_meta["latex_exit_code"] = 1  # Force failure
        build_meta["latex_first_error"] = f"Output too small: {build_meta['gen_pdf_size']} bytes (min {MIN_PDF_SIZE})"
        print(f"   [LaTeX] ❌ Failed: {build_meta['latex_first_error']}")
        return build_meta
    
    # Extract first error line from log if failed
    if result.returncode != 0:
        for line in result.stdout.split("\n"):
            if line.startswith("!") or ":error:" in line.lower():
                build_meta["latex_first_error"] = line.strip()[:200]
                break
        if not build_meta["latex_first_error"]:
            build_meta["latex_first_error"] = f"Exit code {result.returncode}"
        print(f"   [LaTeX] ❌ Failed: {build_meta['latex_first_error']}")
    else:
        print(f"   [LaTeX] ✅ Compilation successful ({build_meta['gen_pdf_size']} bytes)")
    
    return build_meta


def process_document(docx_file: Path, run_id: str):
    """Process a single document through the full pipeline."""
    doc_name = docx_file.stem
    case_dir = OUTPUT_ROOT / doc_name
    case_dir.mkdir(parents=True, exist_ok=True)
    
    src_docx = case_dir / "src.docx"
    ref_pdf = case_dir / "ref.pdf"
    gen_tex = case_dir / "output.tex"
    gen_pdf = case_dir / "gen.pdf"
    build_meta_path = case_dir / "build_meta.json"
    
    # 1. Copy Source
    if not src_docx.exists():
        shutil.copy(docx_file, src_docx)
        
    # 2. Generate Reference PDF
    try:
        convert_docx_to_pdf(src_docx, ref_pdf)
    except Exception as e:
        print(f"Skipping {doc_name} due to PDF conversion failure: {e}")
        return

    # 3. Run Pipeline (ML enabled)
    print(f"   [Pipeline] Running Word to LaTeX...")
    run_command([
        sys.executable, "-m", "wordtolatex.pipeline",
        str(src_docx), str(ref_pdf), str(gen_tex)
    ], cwd=os.getcwd())

    # 4. Compile LaTeX to PDF with proper tracking
    build_meta = compile_latex(gen_tex, case_dir, run_id)
    
    # Save build metadata
    with open(build_meta_path, "w") as f:
        json.dump(build_meta, f, indent=2)

    # 5. Compare Layouts
    print(f"   [Compare] Generating Report & Visuals...")
    cmd = [
        sys.executable, "scripts/compare_layout.py",
        "--ref", str(ref_pdf),
        "--out", str(case_dir / "report.json"),
        "--hints", str(case_dir / "layout_hints.json"),
        "--visuals", "--vis-dir", str(case_dir / "overlays"),
        "--run-id", run_id
    ]
    if gen_pdf.exists():
        cmd.extend(["--gen", str(gen_pdf)])
        
    run_command(cmd, cwd=os.getcwd())
    
    # 5.5. Compute Image Diff (if gen.pdf exists)
    image_diff_path = case_dir / "image_diff.json"
    if gen_pdf.exists():
        print(f"   [ImageDiff] Computing visual similarity...")
        try:
            subprocess.run([
                sys.executable, "scripts/image_diff.py",
                "--ref", str(ref_pdf),
                "--gen", str(gen_pdf),
                "--out", str(image_diff_path),
                "--dpi", "100"  # Lower DPI for speed
            ], cwd=os.getcwd(), capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"   [ImageDiff] Warning: Failed - {e}")
        except Exception as e:
            print(f"   [ImageDiff] Warning: {e}")
    
    # 6. Merge Conversion Report (Policy Trace) + Image Diff into Final Report
    conversion_report_path = case_dir / "conversion_report.json"
    final_report_path = case_dir / "report.json"
    
    if final_report_path.exists():
        try:
            with open(final_report_path) as f:
                layout_report = json.load(f)
            
            # Merge policy trace
            if conversion_report_path.exists():
                with open(conversion_report_path) as f:
                    conv_report = json.load(f)
                layout_report["policy_trace"] = conv_report.get("policy_trace", [])
                layout_report["policy_stats"] = conv_report.get("policy_stats", {})
            
            # Merge image diff
            if image_diff_path.exists():
                with open(image_diff_path) as f:
                    img_diff = json.load(f)
                layout_report["image_diff"] = {
                    "avg_similarity": img_diff.get("avg_similarity", 0),
                    "method": img_diff.get("method", "unknown"),
                    "page_count_match": img_diff.get("page_count_match", False),
                    "ref_pages": img_diff.get("ref_pages", 0),
                    "gen_pages": img_diff.get("gen_pages", 0),
                }
            
            layout_report["run_id"] = run_id
            
            with open(final_report_path, "w") as f:
                json.dump(layout_report, f, indent=2)
            print(f"   [Merge] Policy trace + image diff merged into report.json")
        except Exception as e:
            print(f"   [Merge] Warning: Could not merge reports: {e}")


def validate_corpus() -> bool:
    """Validate all docs have valid gen.pdf. Returns True if all valid."""
    issues = []
    
    for case_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not case_dir.is_dir():
            continue
            
        gen_pdf = case_dir / "gen.pdf"
        report = case_dir / "report.json"
        build_meta_path = case_dir / "build_meta.json"
        
        # Check 1: gen.pdf exists and has content
        if not gen_pdf.exists():
            issues.append(f"{case_dir.name}: gen.pdf missing")
            continue
        if gen_pdf.stat().st_size < MIN_PDF_SIZE:
            issues.append(f"{case_dir.name}: gen.pdf too small ({gen_pdf.stat().st_size} bytes)")
            continue
            
        # Check 2: LaTeX exit code was 0
        if build_meta_path.exists():
            with open(build_meta_path) as f:
                meta = json.load(f)
            if meta.get("latex_exit_code") != 0:
                err = meta.get("latex_first_error", "unknown error")
                issues.append(f"{case_dir.name}: LaTeX failed - {err}")
                
        # Check 3: Content Validation (Policy Trace & Text Blocks)
        if report.exists():
            try:
                with open(report) as f:
                    data = json.load(f)
                
                # Check 3a: Block count (via match_stats)
                match_stats = data.get("match_stats", {})
                gen_blocks = match_stats.get("generated_blocks", 0)
                # Stricter check: avoid passing docs that are mostly empty
                if gen_blocks < 20:
                     issues.append(f"{case_dir.name}: Too few text blocks ({gen_blocks} < 20) - likely poor extraction")
                
                # Check 3b: Page count sanity (allow some drift)
                image_diff = data.get("image_diff", {})
                ref_pages = image_diff.get("ref_pages", 0)
                gen_pages = image_diff.get("gen_pages", 0)
                
                if ref_pages > 0 and gen_pages > 0:
                     diff = abs(ref_pages - gen_pages)
                     tolerance = max(1, ref_pages * 0.2)
                     if diff > tolerance:
                          issues.append(f"{case_dir.name}: Page count mismatch (ref={ref_pages}, gen={gen_pages}) - exceeds tolerance")
                          
                # Check 3c: No empty pages
                if image_diff.get("has_empty_gen_pages", False):
                    issues.append(f"{case_dir.name}: Contains empty pages {image_diff.get('gen_empty_pages')}")
                
            except Exception as e:
                issues.append(f"{case_dir.name}: Report validation failed - {e}")

        # Check 4: report.json is fresher than gen.pdf
        if report.exists() and gen_pdf.exists():
            if report.stat().st_mtime < gen_pdf.stat().st_mtime:
                issues.append(f"{case_dir.name}: report.json is stale (older than gen.pdf)")
    
    if issues:
        print(f"\n❌ CORPUS VALIDATION FAILED: {len(issues)} issue(s)")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    # Count valid docs
    valid_docs = [d for d in OUTPUT_ROOT.iterdir() if d.is_dir() and (d / "gen.pdf").exists()]
    print(f"\n✅ CORPUS PASSED: {len(valid_docs)} docs validated successfully")
    return True


def clean_corpus():
    """Delete all case directories for a fresh run."""
    if not OUTPUT_ROOT.exists():
        return
    
    for case_dir in OUTPUT_ROOT.iterdir():
        if case_dir.is_dir():
            print(f"   Removing {case_dir.name}...")
            shutil.rmtree(case_dir)
    print(f"Cleaned {OUTPUT_ROOT}")


def main():
    parser = argparse.ArgumentParser(description="Run corpus validation pipeline")
    parser.add_argument("--clean", action="store_true", help="Delete all case dirs before running")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't process")
    parser.add_argument("--no-fail", action="store_true", help="Don't exit with error code on validation failure")
    args = parser.parse_args()
    
    if args.validate_only:
        success = validate_corpus()
        if not success and not args.no_fail:
            sys.exit(1)
        return
    
    if args.clean:
        clean_corpus()
    
    if not DATASET_DIR.exists():
        print(f"Dataset directory not found: {DATASET_DIR}")
        return

    run_id = get_run_id()
    print(f"Run ID: {run_id}")

    files = sorted(list(DATASET_DIR.glob("*.docx")))
    print(f"Found {len(files)} documents in corpus.")
    
    for f in files:
        if f.name.startswith("~$"):
            continue  # Skip temp files
        print(f"\n=== Processing {f.name} ===")
        process_document(f, run_id)
    
    # Validate at the end
    print("\n" + "=" * 50)
    print("CORPUS VALIDATION")
    print("=" * 50)
    success = validate_corpus()
    
    if not success and not args.no_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
