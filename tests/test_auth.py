"""
Tests for MedQuery Authentication (Phase 8)
"""

import pytest

from src.api.auth import create_access_token, verify_token


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
