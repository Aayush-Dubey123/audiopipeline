"""
FastAPI application composition root.

Assembles middleware, routes, lifecycle handlers, and centralized exception envelopes.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from core import logger
from core.config import get_settings
from core.database.database import init_db
from core.apis.middleware import SecurityHeadersMiddleware
from core.apis.routes.catalog_routes import router as catalog_router

logging = logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application startup and shutdown lifecycle.

    Initializes database tables on launch and performs graceful cleanup on exit.

    Args:
        app (FastAPI): The root application instance.

    Yields:
        None

    Raises:
        Exception: If database synchronization or initialization fails.
    """
    logging.info(f"Starting AI Voice Studio backend (ENV={settings.ENV}, DEBUG={settings.DEBUG})")
    try:
        init_db()
    except Exception as error:
        logging.error(f"Failed to initialize database on startup: {error}")
        raise error

    yield

    logging.info("Shutting down AI Voice Studio backend cleanly")


def create_app() -> FastAPI:
    """
    Construct and configure the root FastAPI application.

    Binds middleware, lifecycle events, global error handlers, and API routers.

    Returns:
        FastAPI: Fully configured ASGI application instance.

    Raises:
        None
    """
    docs_url = None if settings.ENV.lower() == "production" else "/docs"
    redoc_url = None if settings.ENV.lower() == "production" else "/redoc"

    application = FastAPI(
        title="AI Voice Studio API",
        description="High-performance backend for AI voice generation, cloning, and conversion.",
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        lifespan=lifespan,
    )

    # 1. Inject security headers middleware
    application.add_middleware(SecurityHeadersMiddleware)

    # 2. Inject CORS policy middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Global Exception Handlers returning standard {"error": {"code", "message"}} envelope
    @application.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """
        Format HTTP status exceptions into standardized JSON envelopes.

        Args:
            request (Request): The incoming request.
            exc (StarletteHTTPException): The raised HTTP exception.

        Returns:
            JSONResponse: Standardized error envelope.

        Raises:
            None
        """
        logging.warning(f"HTTP exception on {request.method} {request.url.path}: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": str(exc.status_code),
                    "message": exc.detail,
                }
            },
        )

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """
        Format request schema validation errors into standardized JSON envelopes.

        Args:
            request (Request): The incoming request.
            exc (RequestValidationError): The validation error details.

        Returns:
            JSONResponse: Standardized validation error envelope.

        Raises:
            None
        """
        logging.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid request parameters or payload format.",
                    "details": exc.errors(),
                }
            },
        )

    @application.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all handler for unhandled internal exceptions, masking internal tracebacks.

        Args:
            request (Request): The incoming request.
            exc (Exception): The unhandled exception.

        Returns:
            JSONResponse: Safe 500 error envelope without leaked stack traces.

        Raises:
            None
        """
        logging.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred. Please try again later.",
                }
            },
        )

    # 4. Mount API v1 routers
    application.include_router(catalog_router, prefix="/v1")

    return application


app = create_app()
