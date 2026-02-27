"""Conversion API routes."""

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from server.config import settings
from server.conversion.dependencies import enforce_rate_limit, parse_options
from server.conversion.schemas import ConversionOptions, ConversionResponse
from server.conversion.service import convert_document
from server.conversion.utils import safe_rmtree
from server.stats import record_conversion


router = APIRouter(prefix="/convert", tags=["conversion"])


def _validate_upload(upload: UploadFile, label: str) -> None:
    if not upload.filename:
        raise HTTPException(status_code=400, detail=f"Missing {label} filename")


def _ensure_bytes(data: bytes, label: str) -> bytes:
    if not data:
        raise HTTPException(status_code=400, detail=f"{label} is empty")
    return data


@router.post("")
async def convert_to_tex(
    background_tasks: BackgroundTasks,
    docx: UploadFile = File(...),
    ref_pdf: UploadFile | None = File(default=None),
    options: ConversionOptions = Depends(parse_options),
    _: None = Depends(enforce_rate_limit),
):
    _validate_upload(docx, "docx")
    docx_bytes = _ensure_bytes(await docx.read(), "docx")

    ref_pdf_bytes = None
    if ref_pdf:
        _validate_upload(ref_pdf, "ref_pdf")
        ref_pdf_bytes = _ensure_bytes(await ref_pdf.read(), "ref_pdf")

    result = await run_in_threadpool(convert_document, docx_bytes, ref_pdf_bytes, options)
    record_conversion()
    background_tasks.add_task(safe_rmtree, result.work_dir)

    return FileResponse(
        result.zip_path,
        media_type="application/zip",
        filename="wordtolatex_output.zip",
        background=background_tasks,
    )


@router.post("/json", response_model=ConversionResponse)
async def convert_to_json(
    background_tasks: BackgroundTasks,
    docx: UploadFile = File(...),
    ref_pdf: UploadFile | None = File(default=None),
    options: ConversionOptions = Depends(parse_options),
    _: None = Depends(enforce_rate_limit),
):
    _validate_upload(docx, "docx")
    docx_bytes = _ensure_bytes(await docx.read(), "docx")

    ref_pdf_bytes = None
    if ref_pdf:
        _validate_upload(ref_pdf, "ref_pdf")
        ref_pdf_bytes = _ensure_bytes(await ref_pdf.read(), "ref_pdf")

    result = await run_in_threadpool(convert_document, docx_bytes, ref_pdf_bytes, options)
    record_conversion()
    background_tasks.add_task(safe_rmtree, result.work_dir)

    return ConversionResponse(
        latex=result.latex,
        doc_type=result.doc_type,
        layout_style=result.layout_style,
        metadata=result.metadata if settings.expose_metadata else {},
    )
