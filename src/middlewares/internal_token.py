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

        return await call_next(request)
