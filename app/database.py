"""
Database initialization and session management.
Uses PostgreSQL for production (asyncpg driver).
"""
import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

# PostgreSQL connection from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://ecovoice:ecovoice_password@localhost:5432/ecovoice_analytics"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set True for SQL debugging
    pool_size=10,
    max_overflow=20
)

# Session factory
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_database():
    """Create all tables if they don't exist."""
    try:
        Base.metadata.create_all(bind=engine)
        print(f"📊 [DB] PostgreSQL database initialized")
    except Exception as e:
        print(f"⚠️ [DB] Database init error (will retry on first request): {e}")


def get_db() -> Session:
    """Get database session (for FastAPI dependency injection)."""
    db = SessionLocal()
    try:
        return db
    except Exception:
        db.close()
        raise


@contextmanager
def get_db_context():
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# Note: Database initialization moved to app startup to handle Docker timing
