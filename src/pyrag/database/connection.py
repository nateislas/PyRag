"""Database connection management."""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import get_config
from ..models import Base


def get_engine() -> Engine:
    """Get SQLAlchemy engine instance."""
    config = get_config()
    # For now, use a simple SQLite database
    database_url = "sqlite:///./pyrag.db"
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
    )


def get_session_factory() -> sessionmaker[Session]:
    """Get SQLAlchemy session factory."""
    engine = get_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Get database session with automatic cleanup."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_tables() -> None:
    """Create all database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def drop_tables() -> None:
    """Drop all database tables."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
