"""Database session management for FastAPI."""

from typing import Generator

from fastapi import Depends
from sqlalchemy.orm import Session

from .connection import get_session_factory

# Create session factory
SessionLocal = get_session_factory()


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Type alias for dependency injection
DB = Depends(get_db)
