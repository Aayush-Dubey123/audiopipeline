"""
Controller handling voice catalog and system lifecycle operations.
"""

from datetime import datetime, timezone
from core import logger
from core.apis.schemas.responses.catalog_responses import HealthResponse

logging = logger(__name__)


class CatalogController:
    """
    Business logic and operational control for catalog and service health.
    """

    @staticmethod
    def get_health_status() -> HealthResponse:
        """
        Evaluate and assemble the current operational health of the application.

        Constructs a timestamped health payload confirming system availability.

        Returns:
            HealthResponse: Structured health contract containing status, version, and UTC time.

        Raises:
            Exception: If an unexpected error occurs during status generation.
        """
        logging.info("Executing CatalogController.get_health_status")
        try:
            current_time = datetime.now(timezone.utc).isoformat()
            return HealthResponse(
                status="ok",
                version="0.1.0",
                time=current_time,
            )
        except Exception as error:
            logging.error(f"Error in CatalogController.get_health_status: {error}")
            raise error
