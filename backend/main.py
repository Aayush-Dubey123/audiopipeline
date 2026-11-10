"""
ASGI application entrypoint.

Initializes configuration and binds the Uvicorn web server instance.
"""

import uvicorn
from core.config import get_settings


def run_server() -> None:
    """
    Launch the ASGI server instance with environment-aware configurations.

    Retrieves host and port settings from centralized configuration and binds Uvicorn.

    Returns:
        None

    Raises:
        Exception: If the server fails to bind or encounters a fatal runtime failure.
    """
    settings = get_settings()
    is_development = settings.ENV.lower() == "development"

    uvicorn.run(
        "core.apis.api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=is_development,
        server_header=False,
    )


if __name__ == "__main__":
    run_server()
