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


import hashlib
import time
from collections import defaultdict
from fastapi.responses import JSONResponse
from core.config import get_settings

settings = get_settings()

_RATE_LIMITED_PREFIXES = ("/v1/tts", "/v1/vc", "/v1/reference/", "/v1/text/", "/v1/jobs/")


def _is_rate_limited_path(path: str) -> bool:
    return path.startswith(_RATE_LIMITED_PREFIXES)


def get_hashed_client_key(request: Request) -> str:
    """
    Generate an anonymous one-way cryptographic SHA-256 hash of the client's IP.

    Guarantees that raw user IP addresses are never logged or stored.

    Args:
        request (Request): Inbound HTTP request.

    Returns:
        str: 16-character hexadecimal client hash token.
    """
    # X-Forwarded-For is client-controlled; trusting it lets anyone bypass rate limits
    # and quotas by sending a random header. Only honour it behind a trusted proxy.
    forwarded = request.headers.get("X-Forwarded-For") if settings.TRUST_PROXY_HEADERS else None
    if forwarded:
        raw_ip = forwarded.split(",")[-1].strip()
    elif request.client and request.client.host:
        raw_ip = request.client.host
    else:
        raw_ip = "127.0.0.1"

    return hashlib.sha256(raw_ip.encode("utf-8")).hexdigest()[:16]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    In-memory rate limiter enforcing sliding window request caps per hashed client IP.
    """

    def __init__(self, app, requests_per_minute: int = 30) -> None:
        """
        Initialize the rate limiter middleware.

        Args:
            app: ASGI application.
            requests_per_minute (int): Maximum allowed requests in a rolling 60-second window.
        """
        super().__init__(app)
        self.rpm = requests_per_minute
        self._history: defaultdict = defaultdict(list)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Check rate limit before forwarding request to downstream handlers.

        Args:
            request (Request): Inbound HTTP request.
            call_next (RequestResponseEndpoint): Downstream handler.

        Returns:
            Response: Response or HTTP 429 JSON response.
        """
        path = request.url.path
        # Every endpoint that costs GPU, ffmpeg, or disk is rate limited.
        if request.method != "POST" or not _is_rate_limited_path(path):
            return await call_next(request)

        client_key = get_hashed_client_key(request)
        now = time.time()
        window = 60.0

        # Purge timestamps outside the 60-second sliding window
        self._history[client_key] = [t for t in self._history[client_key] if now - t < window]
        # Drop idle clients so the history map cannot grow without bound
        if len(self._history) > 1000:
            for key in [k for k, v in self._history.items() if not v]:
                del self._history[key]

        # Enforce rate limit
        limit = settings.RATE_LIMIT_PER_MINUTE if hasattr(settings, "RATE_LIMIT_PER_MINUTE") else self.rpm
        if len(self._history[client_key]) >= limit:
            oldest = self._history[client_key][0]
            retry_after = max(1, int(window - (now - oldest)))
            logging.warning(f"Rate limit exceeded for client hash '{client_key}' on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "429",
                        "message": f"Rate limit exceeded. Please wait {retry_after} seconds before retrying.",
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        self._history[client_key].append(now)
        return await call_next(request)

