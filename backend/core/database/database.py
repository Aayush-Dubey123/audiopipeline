"""
Process-wide database singleton client and engine factory.

Provides SQLite and relational database connection pooling and session management.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from core import logger
from core.config import get_settings
from core.database.base_class import Base

logging = logger(__name__)
settings = get_settings()

# Configure engine with thread safety for SQLite
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)


def init_db() -> None:
    """
    Initialize database tables defined in metadata Base.

    Creates all tables bound to the DeclarativeBase if they do not yet exist.

    Returns:
        None

    Raises:
        Exception: If table creation fails during database synchronization.
    """
    logging.info("Executing init_db table synchronization")
    try:
        Base.metadata.create_all(bind=engine)
        logging.info("Database schema initialized successfully")
    except Exception as error:
        logging.error(f"Error during init_db schema initialization: {error}")
        raise error


def get_db() -> Generator[Session, None, None]:
    """
    Dependency generator providing a transactional database session.

    Yields a scoped session and ensures teardown upon request completion.

    Yields:
        Session: Active SQLAlchemy session instance.

    Raises:
        None
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
