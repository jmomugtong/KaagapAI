"""
MedQuery SQLAlchemy Models

Database models for clinical documentation RAG system.
All models use SQLAlchemy 2.0 patterns with async support.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# ============================================
# Configuration
# ============================================

# Embedding vector dimension (all-MiniLM-L6-v2 produces 384-dim vectors)
EMBEDDING_DIMENSION = 384


# ============================================
# Base Model
# ============================================


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
        list[uuid.UUID]: ARRAY(UUID(as_uuid=True)),
    }


# ============================================
# Helper Mixins
# ============================================


class TimestampMixin:
    """Mixin adding created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# ============================================
# Hospital Model
# ============================================


class Hospital(Base, TimestampMixin):
    """Hospital entity for multi-tenant isolation."""

    __tablename__ = "hospitals"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)

    # Relationships
    users: Mapped[list["User"]] = relationship(
        "User", back_populates="hospital", lazy="selectin"
    )
    documents: Mapped[list["ClinicalDoc"]] = relationship(
        "ClinicalDoc", back_populates="hospital", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Hospital(id={self.id}, name='{self.name}', code='{self.code}')>"


# ============================================
# User Model
# ============================================


class User(Base, TimestampMixin):
    """User entity for authentication and authorization."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hospital_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hospitals.id"),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    hospital: Mapped[Optional["Hospital"]] = relationship(
        "Hospital", back_populates="users"
    )
    uploaded_documents: Mapped[list["ClinicalDoc"]] = relationship(
        "ClinicalDoc", back_populates="uploader", lazy="selectin"
    )
    query_logs: Mapped[list["QueryLog"]] = relationship(
        "QueryLog", back_populates="user", lazy="selectin"
    )
    feedback: Mapped[list["UserFeedback"]] = relationship(
        "UserFeedback", back_populates="user", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}')>"


# ============================================
# Clinical Document Model
# ============================================


class ClinicalDoc(Base, TimestampMixin):
    """Clinical document metadata and storage info."""

    __tablename__ = "clinical_docs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    hospital_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hospitals.id"),
        nullable=False,
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False)
    upload_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    uploader_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )
    total_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doc_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    # Relationships
    hospital: Mapped["Hospital"] = relationship("Hospital", back_populates="documents")
    uploader: Mapped[Optional["User"]] = relationship(
        "User", back_populates="uploaded_documents"
    )
    embeddings: Mapped[list["EmbeddingsCache"]] = relationship(
        "EmbeddingsCache",
        back_populates="document",
        lazy="selectin",
        cascade="all, delete-orphan",
    )

    # Indexes
    __table_args__ = (
        Index("idx_clinical_docs_hospital_id", "hospital_id"),
        Index("idx_clinical_docs_document_type", "document_type"),
        Index("idx_clinical_docs_upload_timestamp", "upload_timestamp"),
    )

    def __repr__(self) -> str:
        return f"<ClinicalDoc(id={self.id}, filename='{self.filename}')>"


# ============================================
# Embeddings Cache Model
# ============================================


class EmbeddingsCache(Base):
    """Cached document chunk embeddings with vector storage."""

    __tablename__ = "embeddings_cache"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    chunk_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=False,
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clinical_docs.id", ondelete="CASCADE"),
        nullable=True,
    )
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    extra_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    document: Mapped[Optional["ClinicalDoc"]] = relationship(
        "ClinicalDoc", back_populates="embeddings"
    )

    # Indexes (IVFFlat for vector similarity search)
    __table_args__ = (
        Index("idx_embeddings_chunk_hash", "chunk_hash"),
        Index("idx_embeddings_document_id", "document_id"),
        # Note: IVFFlat index created via migration (requires specific syntax)
    )

    def __repr__(self) -> str:
        return f"<EmbeddingsCache(id={self.id}, chunk_hash='{self.chunk_hash[:8]}...')>"


# ============================================
# Query Log Model
# ============================================


class QueryLog(Base):
    """Audit log for all queries with metrics."""

    __tablename__ = "queries_log"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )
    hospital_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hospitals.id"),
        nullable=True,
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hallucination_detected: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    citations: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    retrieved_doc_ids: Mapped[list[uuid.UUID] | None] = mapped_column(
        ARRAY(UUID(as_uuid=True)),
        nullable=True,
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="query_logs")
    feedback: Mapped[list["UserFeedback"]] = relationship(
        "UserFeedback", back_populates="query_log", lazy="selectin"
    )

    # Indexes
    __table_args__ = (
        Index("idx_queries_user_id", "user_id"),
        Index("idx_queries_hospital_id", "hospital_id"),
        Index("idx_queries_timestamp", "timestamp"),
        Index("idx_queries_hallucination", "hallucination_detected"),
    )

    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, query='{self.query_text[:30]}...')>"


# ============================================
# User Feedback Model
# ============================================


class UserFeedback(Base):
    """User feedback on query responses."""

    __tablename__ = "user_feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    query_log_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("queries_log.id"),
        nullable=True,
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=True,
    )
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    query_log: Mapped[Optional["QueryLog"]] = relationship(
        "QueryLog", back_populates="feedback"
    )
    user: Mapped[Optional["User"]] = relationship("User", back_populates="feedback")

    # Constraints
    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name="rating_range"),
        Index("idx_feedback_query_id", "query_log_id"),
        Index("idx_feedback_user_id", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<UserFeedback(id={self.id}, rating={self.rating})>"
