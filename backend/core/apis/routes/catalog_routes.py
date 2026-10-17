"""
HTTP routes for system catalog and operational health endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from core import logger
from core.apis.schemas.responses.catalog_responses import HealthResponse
from core.controllers.catalog_controller import CatalogController

logging = logger(__name__)

router = APIRouter(tags=["Catalog & Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="System Health & Readiness Check",
    description="Returns service availability, build version, and current UTC server time.",
)
def get_health() -> HealthResponse:
    """
    Handle GET /v1/health requests.

    Delegates status assembly to the controller layer and wraps execution
    in standardized error handling.

    Returns:
        HealthResponse: Status payload with current system state.

    Raises:
        HTTPException 500: If health check evaluation fails unexpectedly.
    """
    logging.info("Calling GET /v1/health endpoint")
    try:
        return CatalogController.get_health_status()
    except HTTPException as httperror:
        logging.error(f"Error in GET /v1/health endpoint: {httperror}")
        raise httperror
    except Exception as error:
        logging.error(f"Error in GET /v1/health endpoint: {error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )
