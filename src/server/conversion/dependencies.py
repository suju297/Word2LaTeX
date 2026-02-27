"""Conversion dependency helpers."""

from fastapi import Form, HTTPException, Request

from server.config import settings
from server.conversion.schemas import ConversionOptions
from server.rate_limit import check_rate_limit


async def parse_options(options_json: str | None = Form(default=None)) -> ConversionOptions:
    if not settings.allow_user_options:
        return ConversionOptions()
    if not options_json:
        return ConversionOptions()
    try:
        return ConversionOptions.model_validate_json(options_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid options JSON: {exc}") from exc


async def enforce_rate_limit(request: Request) -> None:
    check_rate_limit(request)
