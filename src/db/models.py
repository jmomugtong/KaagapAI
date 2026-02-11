from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import String, Text, ForeignKey, TIMESTAMP, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship, DeclarativeBase
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class ClinicalDoc(Base):
    __tablename__ = "clinical_docs"

    id: Mapped[int] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    document_type: Mapped[str] = mapped_column(String(50))
    metadata_: Mapped[Dict[str, Any]] = mapped_column("metadata", JSON, default={})
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow)

    chunks: Mapped[List["DocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "embeddings_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("clinical_docs.id"))
    content: Mapped[str] = mapped_column("chunk_text", Text)
    chunk_index: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(384))
    
    document: Mapped["ClinicalDoc"] = relationship(back_populates="chunks")

class QueriesLog(Base):
    __tablename__ = "queries_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    query_text: Mapped[str] = mapped_column(Text)
    response_text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, default=datetime.utcnow)
    execution_time_ms: Mapped[float] = mapped_column(Integer) # milliseconds
    
