"""Conversion domain exceptions."""

from server.exceptions import AppError


class ConvertFailed(AppError):
    status_code = 500
    code = "convert_failed"
