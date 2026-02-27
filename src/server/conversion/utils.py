"""Utility helpers for conversion artifacts."""

from __future__ import annotations

import shutil
import zipfile
import shlex
import subprocess
from pathlib import Path

from server.config import settings
from server.conversion.exceptions import ConvertFailed


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir))


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def generate_reference_pdf(docx_path: Path, output_dir: Path) -> Path:
    if not settings.auto_reference_pdf:
        raise ConvertFailed("Reference PDF generation is disabled on this server.")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = shlex.quote(str(docx_path))
    outdir_path = shlex.quote(str(output_dir))
    args = settings.reference_pdf_args.format(input=input_path, outdir=outdir_path)
    cmd = [settings.reference_pdf_command] + shlex.split(args)

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=settings.reference_pdf_timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise ConvertFailed(
            "Reference PDF generation failed (command not found). "
            "Install LibreOffice or configure REFERENCE_PDF_COMMAND."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ConvertFailed("Reference PDF generation timed out.") from exc
    except subprocess.CalledProcessError as exc:
        error_detail = exc.stderr.decode(errors="ignore").strip()
        message = "Reference PDF generation failed."
        if error_detail:
            message = f"{message} {error_detail}"
        raise ConvertFailed(message) from exc

    expected = output_dir / f"{docx_path.stem}.pdf"
    if not expected.exists():
        raise ConvertFailed("Reference PDF generation failed to produce output.")
    return expected
