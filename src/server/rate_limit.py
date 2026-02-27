"""Rate limiting utilities for the API."""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

from fastapi import Request

from server.config import settings
from server.exceptions import RateLimitError


class CooldownLimiter:
    """Simple in-memory cooldown limiter keyed by client identifier."""

    def __init__(self, cooldown_seconds: int, now: Callable[[], float] | None = None) -> None:
        self.cooldown_seconds = max(0, int(cooldown_seconds))
        self._now = now or time.monotonic
        self._lock = threading.Lock()
        self._hits: dict[str, float] = {}

    def check(self, key: str) -> int:
        """Return retry-after seconds if within cooldown, otherwise 0."""
        now = self._now()
        with self._lock:
            last = self._hits.get(key)
            if last is None or (now - last) >= self.cooldown_seconds:
                self._hits[key] = now
                return 0
            remaining = self.cooldown_seconds - (now - last)
            return max(1, int(math.ceil(remaining)))


_limiter = CooldownLimiter(settings.rate_limit_cooldown_seconds)


def _client_key(request: Request) -> str:
    if settings.rate_limit_trust_forwarded:
        forwarded_for = request.headers.get("x-forwarded-for") or request.headers.get("x-real-ip")
        if forwarded_for:
            first = forwarded_for.split(",")[0].strip()
            if first:
                return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _format_retry(retry_after: int) -> str:
    if retry_after < 90:
        unit = "second" if retry_after == 1 else "seconds"
        return f"{retry_after} {unit}"
    minutes = int((retry_after + 59) // 60)
    unit = "minute" if minutes == 1 else "minutes"
    return f"{minutes} {unit}"


def check_rate_limit(request: Request) -> None:
    if not settings.rate_limit_enabled:
        return
    retry_after = _limiter.check(_client_key(request))
    if retry_after:
        message = f"Rate limit exceeded. Try again in {_format_retry(retry_after)}."
        raise RateLimitError(message=message, retry_after=retry_after)
