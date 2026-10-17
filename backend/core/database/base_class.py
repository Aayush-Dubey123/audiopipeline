"""
Declarative base class for SQLAlchemy persistence models.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class inherited by all ORM entities in AI Voice Studio.
    """
    pass
