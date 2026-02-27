"""FastAPI entrypoint."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from server.config import settings
from server.conversion.router import router as conversion_router
from server.exceptions import AppError
from server.web.router import router as web_router


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    @app.exception_handler(AppError)
    async def app_error_handler(_, exc: AppError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.message},
            headers=exc.headers,
        )

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "environment": settings.environment}

    app.include_router(web_router)
    app.include_router(conversion_router, prefix=settings.api_prefix)
    return app


app = create_app()
