"""
Security and transport hardening middleware.

Applies standard HTTP security headers to all outgoing responses.
"""

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from core import logger

logging = logger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware applying standard security headers to every HTTP response.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Intercept request and inject hardened HTTP security headers.

        Args:
            request (Request): Inbound HTTP request.
            call_next (RequestResponseEndpoint): Next middleware or endpoint handler.

        Returns:
            Response: Modified HTTP response containing hardened security headers.

        Raises:
            Exception: If an unhandled error occurs during downstream execution.
        """
        response: Response = await call_next(request)

        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline';"
        )
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response
