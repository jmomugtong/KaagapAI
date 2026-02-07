"""Initial schema with all tables

Revision ID: 001_initial
Revises: 
Create Date: 2026-02-07

Initial database schema for MedQuery RAG system.
Creates all tables as defined in PRD Section 6.1.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create all tables and indexes."""
    
    # Enable extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # ========================================
    # Hospitals table
    # ========================================
    op.create_table(
        "hospitals",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, 
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("code", sa.String(50), unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), 
                  server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), 
                  server_default=sa.text("NOW()"), nullable=False),
    )
    
    # ========================================
    # Users table
    # ========================================
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("hospital_id", postgresql.UUID(as_uuid=True), 
                  sa.ForeignKey("hospitals.id"), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("is_superuser", sa.Boolean(), default=False, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()"), nullable=False),
    )
    
    # ========================================
    # Clinical docs table
    # ========================================
    op.create_table(
        "clinical_docs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("hospital_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("hospitals.id"), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("document_type", sa.String(50), nullable=False),
        sa.Column("upload_timestamp", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()"), nullable=False),
        sa.Column("uploader_user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("total_chunks", sa.Integer(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("file_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()"), nullable=False),
    )
    
    # Clinical docs indexes
    op.create_index("idx_clinical_docs_hospital_id", "clinical_docs", ["hospital_id"])
    op.create_index("idx_clinical_docs_document_type", "clinical_docs", ["document_type"])
    op.create_index("idx_clinical_docs_upload_timestamp", "clinical_docs", ["upload_timestamp"])
    
    # ========================================
    # Embeddings cache table
    # ========================================
    op.create_table(
        "embeddings_cache",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("chunk_hash", sa.String(64), unique=True, nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("clinical_docs.id", ondelete="CASCADE"), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=True),
        sa.Column("section_title", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()"), nullable=False),
    )
    
    # Embeddings indexes
    op.create_index("idx_embeddings_chunk_hash", "embeddings_cache", ["chunk_hash"])
    op.create_index("idx_embeddings_document_id", "embeddings_cache", ["document_id"])
    
    # IVFFlat vector index for similarity search
    op.execute("""
        CREATE INDEX idx_embeddings_vector ON embeddings_cache 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)
    
    # ========================================
    # Queries log table
    # ========================================
    op.create_table(
        "queries_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("hospital_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("hospitals.id"), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_hash", sa.String(64), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column("cache_hit", sa.Boolean(), nullable=True),
        sa.Column("hallucination_detected", sa.Boolean(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()"), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=True),
        sa.Column("retrieved_doc_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
    )
    
    # Queries log indexes
    op.create_index("idx_queries_user_id", "queries_log", ["user_id"])
    op.create_index("idx_queries_hospital_id", "queries_log", ["hospital_id"])
    op.create_index("idx_queries_timestamp", "queries_log", ["timestamp"])
    op.create_index("idx_queries_hallucination", "queries_log", ["hallucination_detected"])
    
    # ========================================
    # User feedback table
    # ========================================
    op.create_table(
        "user_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("uuid_generate_v4()")),
        sa.Column("query_log_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("queries_log.id"), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("users.id"), nullable=True),
        sa.Column("rating", sa.Integer(), nullable=True),
        sa.Column("feedback_text", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True),
                  server_default=sa.text("NOW()"), nullable=False),
        sa.CheckConstraint("rating >= 1 AND rating <= 5", name="rating_range"),
    )
    
    # User feedback indexes
    op.create_index("idx_feedback_query_id", "user_feedback", ["query_log_id"])
    op.create_index("idx_feedback_user_id", "user_feedback", ["user_id"])
    
    # ========================================
    # Row Level Security
    # ========================================
    op.execute("ALTER TABLE clinical_docs ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE queries_log ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY")
    
    # ========================================
    # Updated_at trigger function
    # ========================================
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql'
    """)
    
    # Triggers for updated_at
    op.execute("""
        CREATE TRIGGER update_hospitals_updated_at
        BEFORE UPDATE ON hospitals
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
    """)
    
    op.execute("""
        CREATE TRIGGER update_users_updated_at
        BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
    """)
    
    op.execute("""
        CREATE TRIGGER update_clinical_docs_updated_at
        BEFORE UPDATE ON clinical_docs
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
    """)
    
    # ========================================
    # Seed data (development hospital)
    # ========================================
    op.execute("""
        INSERT INTO hospitals (id, name, code)
        VALUES ('00000000-0000-0000-0000-000000000001', 'Development Hospital', 'DEV')
        ON CONFLICT (code) DO NOTHING
    """)


def downgrade() -> None:
    """Drop all tables and extensions."""
    
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS update_clinical_docs_updated_at ON clinical_docs")
    op.execute("DROP TRIGGER IF EXISTS update_users_updated_at ON users")
    op.execute("DROP TRIGGER IF EXISTS update_hospitals_updated_at ON hospitals")
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")
    
    # Drop tables (reverse order due to foreign keys)
    op.drop_table("user_feedback")
    op.drop_table("queries_log")
    op.drop_table("embeddings_cache")
    op.drop_table("clinical_docs")
    op.drop_table("users")
    op.drop_table("hospitals")
    
    # Note: Extensions are not dropped to avoid affecting other databases
