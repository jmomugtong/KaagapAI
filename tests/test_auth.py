"""
Tests for MedQuery Authentication (Phase 8)
"""

import pytest

from src.api.auth import (
    create_access_token,
    hash_password,
    verify_password,
    verify_token,
)


class TestJWTAuth:
    """Tests for JWT token creation and verification."""

    @pytest.mark.unit
    def test_create_token_returns_string(self):
        token = create_access_token({"sub": "user123", "hospital_id": "hosp1"})
        assert isinstance(token, str)
        assert len(token) > 0

    @pytest.mark.unit
    def test_verify_valid_token(self):
        token = create_access_token({"sub": "user123"})
        payload = verify_token(token)
        assert payload["sub"] == "user123"

    @pytest.mark.unit
    def test_verify_expired_token(self):
        from datetime import timedelta

        from fastapi import HTTPException

        token = create_access_token(
            {"sub": "user123"}, expires_delta=timedelta(seconds=-1)
        )
        with pytest.raises(HTTPException) as exc_info:
            verify_token(token)
        assert exc_info.value.status_code == 401

    @pytest.mark.unit
    def test_verify_invalid_token(self):
        from fastapi import HTTPException

        with pytest.raises(HTTPException):
            verify_token("invalid.token.string")

    @pytest.mark.unit
    def test_token_contains_expiry(self):
        token = create_access_token({"sub": "user123"})
        payload = verify_token(token)
        assert "exp" in payload


class TestPasswordHashing:
    """Tests for bcrypt password hashing utilities."""

    @pytest.mark.unit
    def test_hash_password_returns_string(self):
        hashed = hash_password("mysecret")
        assert isinstance(hashed, str)
        assert hashed != "mysecret"

    @pytest.mark.unit
    def test_verify_password_correct(self):
        hashed = hash_password("mysecret")
        assert verify_password("mysecret", hashed) is True

    @pytest.mark.unit
    def test_verify_password_wrong(self):
        hashed = hash_password("mysecret")
        assert verify_password("wrongpassword", hashed) is False

    @pytest.mark.unit
    def test_hash_is_unique_per_call(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        # bcrypt uses random salt, so hashes should differ
        assert h1 != h2
        # But both should verify
        assert verify_password("same", h1)
        assert verify_password("same", h2)


class TestAuthEndpoints:
    """Tests for /api/v1/auth/register and /api/v1/auth/login endpoints."""

    @pytest.mark.unit
    def test_register_success(self, client, mocker):
        """Register a new user returns a token."""
        mock_session = mocker.AsyncMock()
        # scalar_one_or_none returns None → no duplicate
        mock_result = mocker.MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result
        mock_session.flush = mocker.AsyncMock()
        mock_session.commit = mocker.AsyncMock()

        # After refresh, user should have an id
        async def fake_refresh(user):
            user.id = 1

        mock_session.refresh = mocker.AsyncMock(side_effect=fake_refresh)

        mock_ctx = mocker.AsyncMock()
        mock_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = mocker.AsyncMock(return_value=False)
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=mock_ctx)

        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "StrongPass123!",
                "full_name": "New User",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        # Verify the token is valid
        payload = verify_token(data["access_token"])
        assert payload["sub"] == "newuser@example.com"

    @pytest.mark.unit
    def test_register_duplicate_email(self, client, mocker):
        """Registering with an existing email returns 409."""
        from src.db.models import User

        existing_user = User(
            id=1,
            email="taken@example.com",
            hashed_password="hashed",
            full_name="Existing",
        )

        mock_session = mocker.AsyncMock()
        mock_result = mocker.MagicMock()
        mock_result.scalar_one_or_none.return_value = existing_user
        mock_session.execute.return_value = mock_result

        mock_ctx = mocker.AsyncMock()
        mock_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = mocker.AsyncMock(return_value=False)
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=mock_ctx)

        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "taken@example.com",
                "password": "StrongPass123!",
                "full_name": "Duplicate",
            },
        )

        assert response.status_code == 409
        assert "already registered" in response.json()["detail"]

    @pytest.mark.unit
    def test_login_success(self, client, mocker):
        """Login with correct credentials returns a token."""
        from src.db.models import User

        hashed = hash_password("correct_password")
        user = User(
            id=42,
            email="login@example.com",
            hashed_password=hashed,
            full_name="Login User",
        )

        mock_session = mocker.AsyncMock()
        mock_result = mocker.MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        mock_ctx = mocker.AsyncMock()
        mock_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = mocker.AsyncMock(return_value=False)
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=mock_ctx)

        response = client.post(
            "/api/v1/auth/login",
            json={"email": "login@example.com", "password": "correct_password"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        payload = verify_token(data["access_token"])
        assert payload["sub"] == "login@example.com"
        assert payload["user_id"] == 42

    @pytest.mark.unit
    def test_login_wrong_password(self, client, mocker):
        """Login with wrong password returns 401."""
        from src.db.models import User

        hashed = hash_password("right_password")
        user = User(
            id=1,
            email="user@example.com",
            hashed_password=hashed,
            full_name="Test",
        )

        mock_session = mocker.AsyncMock()
        mock_result = mocker.MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute.return_value = mock_result

        mock_ctx = mocker.AsyncMock()
        mock_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = mocker.AsyncMock(return_value=False)
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=mock_ctx)

        response = client.post(
            "/api/v1/auth/login",
            json={"email": "user@example.com", "password": "wrong_password"},
        )

        assert response.status_code == 401
        assert "Invalid email or password" in response.json()["detail"]

    @pytest.mark.unit
    def test_login_nonexistent_user(self, client, mocker):
        """Login with nonexistent email returns 401."""
        mock_session = mocker.AsyncMock()
        mock_result = mocker.MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        mock_ctx = mocker.AsyncMock()
        mock_ctx.__aenter__ = mocker.AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = mocker.AsyncMock(return_value=False)
        mocker.patch("src.db.postgres.AsyncSessionLocal", return_value=mock_ctx)

        response = client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@example.com", "password": "whatever"},
        )

        assert response.status_code == 401

    @pytest.mark.unit
    def test_protected_route_with_token(self, client, mocker):
        """Query endpoint accepts and processes Bearer token without error."""
        token = create_access_token({"sub": "authed@example.com", "user_id": 1})

        # Mock the pipeline to avoid needing real services
        mock_result = mocker.MagicMock()
        mock_result.answer = "Test answer"
        mock_result.confidence = 0.9
        mock_result.citations = []
        mock_result.retrieved_chunks = []
        mock_result.query_id = "q_test"
        mock_result.processing_time_ms = 100
        mock_result.hallucination_flagged = False
        mock_result.cached = False

        mock_pipeline = mocker.MagicMock()
        mock_pipeline.run = mocker.AsyncMock(return_value=mock_result)
        mocker.patch(
            "src.pipelines.classical.ClassicalPipeline", return_value=mock_pipeline
        )

        response = client.post(
            "/api/v1/query",
            json={"question": "What is the pain management protocol?"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["answer"] == "Test answer"

    @pytest.mark.unit
    def test_protected_route_without_token(self, client, mocker):
        """Query endpoint still works without a token (optional auth)."""
        mock_result = mocker.MagicMock()
        mock_result.answer = "Test answer"
        mock_result.confidence = 0.9
        mock_result.citations = []
        mock_result.retrieved_chunks = []
        mock_result.query_id = "q_test"
        mock_result.processing_time_ms = 100
        mock_result.hallucination_flagged = False
        mock_result.cached = False

        mock_pipeline = mocker.MagicMock()
        mock_pipeline.run = mocker.AsyncMock(return_value=mock_result)
        mocker.patch(
            "src.pipelines.classical.ClassicalPipeline", return_value=mock_pipeline
        )

        response = client.post(
            "/api/v1/query",
            json={"question": "What is the pain management protocol?"},
        )

        assert response.status_code == 200
        assert response.json()["answer"] == "Test answer"
