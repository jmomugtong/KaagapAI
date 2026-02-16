import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.db.models import Base, ClinicalDoc, DocumentChunk

TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://medquery_user:medquery_password@localhost:5432/medquery",
)

pytestmark = [pytest.mark.requires_db, pytest.mark.integration]


@pytest.fixture
async def engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)
    try:
        async with engine.begin() as conn:
            # Create pgvector extension if not exists
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Explicitly drop tables with CASCADE to handle foreign keys from init_db.sql
            await conn.execute(
                text(
                    "DROP TABLE IF EXISTS user_feedback, queries_log, embeddings_cache, clinical_docs, users, hospitals CASCADE"
                )
            )

            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    except (ConnectionError, OSError, Exception) as e:
        await engine.dispose()
        pytest.skip(f"PostgreSQL not available: {e}")
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
async def db_session(engine):
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


@pytest.mark.asyncio
async def test_database_connection(db_session):
    result = await db_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_pgvector_extension(db_session):
    result = await db_session.execute(
        text("SELECT * FROM pg_extension WHERE extname = 'vector'")
    )
    assert result.fetchone() is not None


@pytest.mark.asyncio
async def test_clinical_doc_crud(db_session):
    # Create
    doc = ClinicalDoc(
        filename="test_protocol.pdf",
        document_type="protocol",
        metadata_={"department": "cardiology"},
    )
    db_session.add(doc)
    await db_session.commit()
    await db_session.refresh(doc)
    assert doc.id is not None

    # Read
    saved_doc = await db_session.get(ClinicalDoc, doc.id)
    assert saved_doc.filename == "test_protocol.pdf"


@pytest.mark.asyncio
async def test_document_chunk_crud(db_session):
    # Setup parent doc
    doc = ClinicalDoc(filename="chunk_test.pdf", document_type="ref", metadata_={})
    db_session.add(doc)
    await db_session.commit()

    # Create Chunk
    chunk = DocumentChunk(
        document_id=doc.id,
        content="Test content chunk",
        chunk_index=0,
        embedding=[0.1] * 768,  # 768-dim vector
    )
    db_session.add(chunk)
    await db_session.commit()

    # Read
    saved_chunk = await db_session.get(DocumentChunk, chunk.id)
    assert saved_chunk.content == "Test content chunk"
