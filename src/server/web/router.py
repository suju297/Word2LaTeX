"""Web UI routes."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from server.stats import get_stats, record_visit
from server.web.service import render_homepage


router = APIRouter(tags=["web"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    record_visit(request)
    return render_homepage()


@router.get("/stats")
async def stats():
    return get_stats()
