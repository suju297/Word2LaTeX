"""Shared API exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AppError(Exception):
    message: str
    status_code: int = 400
    code: str = "app_error"
    headers: Optional[dict[str, str]] = None


class ConversionError(AppError):
    status_code: int = 422
    code: str = "conversion_error"


class RateLimitError(AppError):
    def __init__(self, message: str, retry_after: int) -> None:
        super().__init__(
            message=message,
            status_code=429,
            code="rate_limited",
            headers={"Retry-After": str(retry_after)},
        )
