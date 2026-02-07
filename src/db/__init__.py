"""
MedQuery Database Module

Database components:
- PostgreSQL with pgvector integration
- SQLAlchemy models
- Connection management
"""

from src.db.models import (
    Base,
    Hospital,
    User,
    ClinicalDoc,
    EmbeddingsCache,
    QueryLog,
    UserFeedback,
    EMBEDDING_DIMENSION,
)
from src.db.postgres import (
    get_db,
    get_db_session,
    get_engine,
    init_db,
    close_db,
    check_database_health,
    check_pgvector_extension,
    POOL_SIZE,
    MAX_OVERFLOW,
    POOL_RECYCLE,
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
