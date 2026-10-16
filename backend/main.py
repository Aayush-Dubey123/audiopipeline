"""
ASGI Application Entrypoint.

Starts the Uvicorn web server and manages runtime environment initialization.
"""

import os
import uvicorn
from dotenv import load_dotenv

# Load local environment configuration
load_dotenv()


def run_server() -> None:
    """
    Launch the ASGI server instance with environment-aware configurations.

    Reads host, port, and reload options from the environment and binds Uvicorn.

    Returns:
        None

    Raises:
        Exception: If the server fails to bind or encountered runtime termination.
    """
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    environment = os.getenv("ENVIRONMENT", "development").lower()
    is_development = environment == "development"

    uvicorn.run(
        "core.apis.api:app",
        host=host,
        port=port,
        reload=is_development,
        server_header=False,
    )


if __name__ == "__main__":
    run_server()
