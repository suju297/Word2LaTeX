"""In-memory usage statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Dict

from fastapi import Request

from server.config import settings


@dataclass
class UsageStats:
    visitors: set[str] = field(default_factory=set)
    conversions: int = 0


_LOCK = threading.Lock()
_STATS = UsageStats()


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


def record_visit(request: Request) -> None:
    key = _client_key(request)
    with _LOCK:
        _STATS.visitors.add(key)


def record_conversion() -> None:
    with _LOCK:
        _STATS.conversions += 1


def get_stats() -> Dict[str, int]:
    with _LOCK:
        return {
            "visitor_count": len(_STATS.visitors),
            "conversion_count": _STATS.conversions,
        }
