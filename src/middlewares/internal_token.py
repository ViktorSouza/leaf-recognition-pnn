import secrets

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from constants import BYPASS_ENDPOINTS, INTERNAL_TOKEN


class InternalTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path in BYPASS_ENDPOINTS:
            return await call_next(request)

        if not INTERNAL_TOKEN or not secrets.compare_digest(
            request.headers.get("INTERNAL-TOKEN", "").encode("utf-8"),
            INTERNAL_TOKEN.encode("utf-8") if INTERNAL_TOKEN else b"",
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Token interno inválido ou ausente."},
            )

        return await call_next(request)
