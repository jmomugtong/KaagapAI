"""
KaagapAI Test Configuration

Pytest fixtures and configuration for the test suite.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.main import app

# ============================================
# Event Loop Configuration
# ============================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================
# Client Fixtures
# ============================================


@pytest.fixture(autouse=True)
def reset_rate_limiter() -> Generator[None, None, None]:
    """Clear in-memory rate limiter counters before each test."""
    from src.security.rate_limiter import _limiter

    _limiter._counters.clear()
    yield


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a synchronous test client."""
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# ============================================
# Database Fixtures
# ============================================


@pytest.fixture
def mock_db_session():
    """Mock database session for unit tests."""
    # TODO: Implement mock database session
    pass


@pytest_asyncio.fixture
async def test_db():
    """Create a test database for integration tests."""
    # TODO: Setup test database with pgvector
    # TODO: Run migrations
    # TODO: Seed test data
    yield
    # TODO: Cleanup test database


@pytest_asyncio.fixture
async def async_session():
    """Create an async database session for integration tests."""
    import os

    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    database_url = os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+asyncpg://kaagapai_user:kaagapai_password@localhost:5432/kaagapai_test",
    )

    from src.db.models import Base

    engine = create_async_engine(database_url, echo=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session
        await session.rollback()  # Rollback any changes after test

    # Drop tables after test (optional, but good for cleanup if sharing DB)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


# ============================================
# Redis Fixtures
# ============================================


@pytest.fixture
def mock_redis():
    """Mock Redis client for unit tests."""
    # TODO: Implement mock Redis client
    pass


# ============================================
# LLM Fixtures
# ============================================


@pytest.fixture
def mock_ollama():
    """Mock Ollama client for unit tests."""
    # TODO: Implement mock Ollama responses
    pass


# ============================================
# Sample Data Fixtures
# ============================================


@pytest.fixture
def sample_clinical_query() -> dict:
    """Sample clinical query for testing."""
    return {
        "question": "What is the post-operative pain protocol for knee replacement?",
        "max_results": 5,
        "confidence_threshold": 0.70,
    }


@pytest.fixture
def sample_chunk() -> dict:
    """Sample document chunk for testing."""
    return {
        "text": "Administer acetaminophen 1000mg every 6 hours for post-operative pain management.",
        "metadata": {
            "document": "Post-Op Pain Management Protocol v3.2",
            "section": "Knee Replacement Procedures",
            "page": 12,
            "chunk_index": 5,
        },
    }


@pytest.fixture
def sample_embedding() -> list[float]:
    """Sample 768-dimensional embedding vector."""
    import random

    random.seed(42)
    return [random.random() for _ in range(768)]


@pytest.fixture
def sample_pii_text() -> str:
    """Sample text containing PII for redaction testing."""
    return """
    Patient John Doe (MRN: ABC12345678) was admitted on 01/15/1980.
    Contact number: 555-123-4567. Email: patient@example.com.
    SSN: 123-45-6789.
    """


@pytest.fixture
def sample_evaluation_question() -> dict:
    """Sample evaluation question with ground truth."""
    return {
        "id": "q001",
        "query": "What is the post-operative pain protocol for knee replacement?",
        "ground_truth": "Administer acetaminophen 1000mg every 6 hours, combined with oxycodone 5-10mg every 4-6 hours as needed.",
        "expected_sources": ["Post-Op Pain Management Protocol v3.2"],
        "category": "pain_management",
    }


# ============================================
# Marker Configuration
# ============================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "requires_ollama: test requires Ollama service")
    config.addinivalue_line("markers", "requires_db: test requires database connection")
    config.addinivalue_line("markers", "requires_redis: test requires Redis connection")
