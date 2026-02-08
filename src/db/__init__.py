"""
MedQuery Database Module

Database components:
- PostgreSQL with pgvector integration
- SQLAlchemy models
- Connection management
"""

from src.db.models import (
    EMBEDDING_DIMENSION,
    Base,
    ClinicalDoc,
    EmbeddingsCache,
    Hospital,
    QueryLog,
    User,
    UserFeedback,
)
from src.db.postgres import (
    MAX_OVERFLOW,
    POOL_RECYCLE,
    POOL_SIZE,
    check_database_health,
    check_pgvector_extension,
    close_db,
    get_db,
    get_db_session,
    get_engine,
    init_db,
)

__all__ = [
    # Base
    "Base",
    # Models
    "Hospital",
    "User",
    "ClinicalDoc",
    "EmbeddingsCache",
    "QueryLog",
    "UserFeedback",
    # Constants
    "EMBEDDING_DIMENSION",
    "POOL_SIZE",
    "MAX_OVERFLOW",
    "POOL_RECYCLE",
    # Functions
    "get_db",
    "get_db_session",
    "get_engine",
    "init_db",
    "close_db",
    "check_database_health",
    "check_pgvector_extension",
]
