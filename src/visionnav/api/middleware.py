"""Custom middleware — request IDs and error handling."""
from __future__ import annotations
import uuid
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid      = str(uuid.uuid4())[:8]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            log.error("unhandled", path=request.url.path, error=str(exc))
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=500,
                content={"error": "internal_server_error", "detail": str(exc)})
