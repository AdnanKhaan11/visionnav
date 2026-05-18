"""FastAPI application factory."""
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from visionnav.api.middleware import ErrorHandlerMiddleware, RequestIDMiddleware
from visionnav.api.v1.router import v1_router
from visionnav.settings import Settings
from visionnav.utils.logging import setup_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()
    setup_logging()
    app = FastAPI(
        title="VisionNav API", version="1.0.0",
        docs_url="/docs" if settings.env != "production" else None,
        redoc_url=None,
    )
    app.add_middleware(CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(ErrorHandlerMiddleware)
    app.include_router(v1_router, prefix="/v1")
    return app


def run_server() -> None:
    import uvicorn
    s = Settings()
    uvicorn.run("visionnav.api.app:create_app", factory=True,
                host=s.api.host, port=s.api.port,
                reload=s.env == "development")
