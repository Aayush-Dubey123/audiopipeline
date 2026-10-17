"""
Application root package.

Initializes environment loading from .env and re-exports the central application logger.
"""

from dotenv import load_dotenv
from commons.logger import logger

# Load environment configuration from .env at package initialization
load_dotenv()

__all__ = ["logger"]
