"""
Input Validation for MedQuery (Phase 7)

Validates and sanitizes user input to prevent:
- SQL injection
- XSS / script injection
- Malicious patterns
- Invalid query lengths
"""

import logging
import re

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# Dangerous patterns
SQL_INJECTION_PATTERNS = [
    re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|UNION|TRUNCATE)\b)",
        re.IGNORECASE,
    ),
    re.compile(r"(--|;|/\*|\*/|@@|char\(|nchar\(|varchar\(|exec\()", re.IGNORECASE),
    re.compile(r"(\bOR\b\s+\d+\s*=\s*\d+)", re.IGNORECASE),
    re.compile(r"('\s*(OR|AND)\s+')", re.IGNORECASE),
]

XSS_PATTERNS = [
    re.compile(r"<\s*script", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),
    re.compile(r"<\s*iframe", re.IGNORECASE),
    re.compile(r"<\s*object", re.IGNORECASE),
    re.compile(r"<\s*embed", re.IGNORECASE),
]

MIN_QUERY_LENGTH = 10
MAX_QUERY_LENGTH = 500


class QueryRequest(BaseModel):
    """Validated query request model."""

    question: str
    max_results: int = 5
    confidence_threshold: float = 0.70

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        v = v.strip()
        if len(v) < MIN_QUERY_LENGTH:
            raise ValueError(f"Query must be at least {MIN_QUERY_LENGTH} characters")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query must be at most {MAX_QUERY_LENGTH} characters")
        return v

    @field_validator("max_results")
    @classmethod
    def validate_max_results(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError("max_results must be between 1 and 20")
        return v

    @field_validator("confidence_threshold")
    @classmethod
    def validate_confidence_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v


class UploadRequest(BaseModel):
    """Validated upload metadata."""

    document_type: str = "protocol"
    metadata: str = "{}"

    @field_validator("document_type")
    @classmethod
    def validate_document_type(cls, v: str) -> str:
        allowed = {"protocol", "guideline", "reference"}
        if v not in allowed:
            raise ValueError(f"document_type must be one of: {allowed}")
        return v


class InputValidator:
    """Validates input against injection patterns."""

    def check_sql_injection(self, text: str) -> bool:
        """Return True if SQL injection pattern detected."""
        for pattern in SQL_INJECTION_PATTERNS:
            if pattern.search(text):
                logger.warning("SQL injection pattern detected: %s", text[:100])
                return True
        return False

    def check_xss(self, text: str) -> bool:
        """Return True if XSS pattern detected."""
        for pattern in XSS_PATTERNS:
            if pattern.search(text):
                logger.warning("XSS pattern detected: %s", text[:100])
                return True
        return False

    def is_safe(self, text: str) -> bool:
        """Return True if input passes all safety checks."""
        return not self.check_sql_injection(text) and not self.check_xss(text)

    def sanitize(self, text: str) -> str:
        """Strip potentially dangerous characters from input."""
        # Remove null bytes
        text = text.replace("\x00", "")
        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text.strip()
