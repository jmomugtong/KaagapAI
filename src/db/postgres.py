"""
MedQuery Database Connection Management

PostgreSQL async connection pool with SQLAlchemy 2.0.
Includes session handling and health checks.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy import text

# ============================================
# Configuration Constants
# ============================================

# Connection pool settings
POOL_SIZE = 5
MAX_OVERFLOW = 10
POOL_RECYCLE = 1800  # 30 minutes
POOL_PRE_PING = True

# Default database URL (overridden by environment)
DEFAULT_DATABASE_URL = "postgresql+asyncpg://medquery_user:medquery_password@localhost:5432/medquery"


# ============================================
# Database URL
# ============================================


def get_database_url() -> str:
    """
    Get database URL from environment variable.
    
    Returns:
        Database connection URL string.
    """
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


# ============================================
# Engine & Session Factory
# ============================================


def create_db_engine(database_url: str | None = None) -> AsyncEngine:
    """
    Create async database engine with connection pooling.
    
    Args:
        database_url: Optional database URL. Uses environment if not provided.
        
    Returns:
        AsyncEngine configured with connection pool.
    """
    url = database_url or get_database_url()
    
    return create_async_engine(
        url,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_recycle=POOL_RECYCLE,
        pool_pre_ping=POOL_PRE_PING,
        echo=os.environ.get("DB_ECHO", "false").lower() == "true",
    )


# Global engine instance (lazy initialization)
_engine: AsyncEngine | None = None


def get_engine() -> AsyncEngine:
    """
    Get or create the global async engine instance.
    
    Returns:
        AsyncEngine instance.
    """
    global _engine
    if _engine is None:
        _engine = create_db_engine()
    return _engine


# Expose as 'engine' for backward compatibility
engine = property(lambda self: get_engine())


def get_async_session_maker() -> async_sessionmaker[AsyncSession]:
    """
    Get async session maker bound to the engine.
    
    Returns:
        Async session maker instance.
    """
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )


# Global session maker
AsyncSessionLocal = get_async_session_maker()


# ============================================
# Session Context Manager
# ============================================


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.
    
    Automatically commits on success, rolls back on exception.
    
    Yields:
        AsyncSession for database operations.
        
    Example:
        async with get_db_session() as session:
            result = await session.execute(query)
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database session injection.
    
    Use with FastAPI Depends():
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    
    Yields:
        AsyncSession for database operations.
    """
    async with get_db_session() as session:
        yield session


# ============================================
# Health Check
# ============================================


async def check_database_health() -> dict:
    """
    Check database connectivity and health.
    
    Returns:
        Dict with status and connection details.
    """
    try:
        async with get_db_session() as session:
            # Execute simple query to verify connection
            result = await session.execute(text("SELECT 1 AS health_check"))
            row = result.scalar()
            
            if row == 1:
                return {
                    "status": "healthy",
                    "database": "connected",
                    "pool_size": POOL_SIZE,
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
        }
    
    return {
        "status": "unknown",
        "database": "check_failed",
    }


async def check_pgvector_extension() -> bool:
    """
    Verify pgvector extension is installed.
    
    Returns:
        True if pgvector extension is available.
    """
    try:
        async with get_db_session() as session:
            result = await session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            extension = result.scalar()
            return extension == "vector"
    except Exception:
        return False


# ============================================
# Lifecycle Management
# ============================================


async def init_db() -> None:
    """
    Initialize database connection pool.
    
    Call during application startup.
    """
    global _engine
    _engine = create_db_engine()
    
    # Verify connection
    health = await check_database_health()
    if health["status"] != "healthy":
        raise RuntimeError(f"Database connection failed: {health}")


async def close_db() -> None:
    """
    Close database connection pool.
    
    Call during application shutdown.
    """
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
