"""Web UI service helpers."""

from __future__ import annotations

from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

from server.config import settings
from server.stats import get_stats
from server.web.constants import INDEX_TEMPLATE, TEMPLATE_DIR

_ENV = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)


def render_homepage() -> HTMLResponse:
    cooldown_minutes = max(1, int((settings.rate_limit_cooldown_seconds + 59) // 60))
    stats = get_stats()
    html = _ENV.get_template(INDEX_TEMPLATE).render(
        cooldown_minutes=cooldown_minutes,
        visitor_count=stats["visitor_count"],
        conversion_count=stats["conversion_count"],
    )
    return HTMLResponse(content=html)
