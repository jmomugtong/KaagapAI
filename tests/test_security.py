"""
Tests for MedQuery Security Layer (Phase 7)

Tests for PII redaction, input validation, and rate limiting.
"""

import pytest
from pydantic import ValidationError

from src.security.input_validation import InputValidator, QueryRequest
from src.security.pii_redaction import PIIRedactor
from src.security.rate_limiter import RATE_LIMIT, RateLimiter

# ============================================
# PII Redaction Tests
# ============================================


class TestPIIRedaction:
    """Tests for PII detection and redaction."""

    @pytest.mark.unit
    def test_redacts_ssn_dashed(self):
        redactor = PIIRedactor()
        text = "SSN: 123-45-6789"
        result = redactor.redact(text)
        assert "123-45-6789" not in result
        assert "[SSN]" in result

    @pytest.mark.unit
    def test_redacts_email(self):
        redactor = PIIRedactor()
        text = "Email: patient@example.com"
        result = redactor.redact(text)
        assert "patient@example.com" not in result
        assert "[EMAIL]" in result

    @pytest.mark.unit
    def test_redacts_phone_dashed(self):
        redactor = PIIRedactor()
        text = "Phone: 555-123-4567"
        result = redactor.redact(text)
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    @pytest.mark.unit
    def test_redacts_phone_parenthesized(self):
        redactor = PIIRedactor()
        text = "Call (555) 123-4567"
        result = redactor.redact(text)
        assert "(555) 123-4567" not in result
        assert "[PHONE]" in result

    @pytest.mark.unit
    def test_redacts_dob(self):
        redactor = PIIRedactor()
        text = "DOB: 01/15/1980"
        result = redactor.redact(text)
        assert "01/15/1980" not in result
        assert "[DOB]" in result

    @pytest.mark.unit
    def test_redacts_mrn(self):
        redactor = PIIRedactor()
        text = "MRN: ABC12345678"
        result = redactor.redact(text)
        assert "ABC12345678" not in result
        assert "[MRN]" in result

    @pytest.mark.unit
    def test_redacts_patient_name(self):
        redactor = PIIRedactor()
        text = "Patient John Doe was admitted"
        result = redactor.redact(text)
        assert "John Doe" not in result
        assert "[PATIENT_NAME]" in result

    @pytest.mark.unit
    def test_redacts_multiple_pii(self, sample_pii_text):
        redactor = PIIRedactor()
        result = redactor.redact(sample_pii_text)
        assert "John Doe" not in result
        assert "ABC12345678" not in result
        assert "555-123-4567" not in result
        assert "patient@example.com" not in result
        assert "123-45-6789" not in result

    @pytest.mark.unit
    def test_empty_text(self):
        redactor = PIIRedactor()
        assert redactor.redact("") == ""

    @pytest.mark.unit
    def test_no_pii(self):
        redactor = PIIRedactor()
        text = "Administer acetaminophen 1000mg every 6 hours."
        assert redactor.redact(text) == text

    @pytest.mark.unit
    def test_detect_returns_findings(self):
        redactor = PIIRedactor()
        text = "Email: test@example.com"
        findings = redactor.detect(text)
        assert len(findings) > 0
        assert findings[0]["type"] == "[EMAIL]"

    @pytest.mark.unit
    def test_has_pii_true(self):
        redactor = PIIRedactor()
        assert redactor.has_pii("SSN: 123-45-6789") is True

    @pytest.mark.unit
    def test_has_pii_false(self):
        redactor = PIIRedactor()
        assert redactor.has_pii("No PII here.") is False


# ============================================
# Input Validation Tests
# ============================================


class TestInputValidation:
    """Tests for input validation and sanitization."""

    @pytest.mark.unit
    def test_valid_query(self):
        q = QueryRequest(question="What is the protocol for knee replacement?")
        assert q.question == "What is the protocol for knee replacement?"

    @pytest.mark.unit
    def test_query_too_short(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="Hi")

    @pytest.mark.unit
    def test_query_too_long(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="x" * 501)

    @pytest.mark.unit
    def test_max_results_valid(self):
        q = QueryRequest(question="What is the protocol for pain?", max_results=10)
        assert q.max_results == 10

    @pytest.mark.unit
    def test_max_results_out_of_range(self):
        with pytest.raises(ValidationError):
            QueryRequest(question="What is the protocol for pain?", max_results=50)

    @pytest.mark.unit
    def test_confidence_threshold_valid(self):
        q = QueryRequest(
            question="What is the protocol for pain?",
            confidence_threshold=0.85,
        )
        assert q.confidence_threshold == 0.85

    @pytest.mark.unit
    def test_confidence_threshold_out_of_range(self):
        with pytest.raises(ValidationError):
            QueryRequest(
                question="What is the protocol for pain?",
                confidence_threshold=1.5,
            )


class TestInputValidator:
    """Tests for SQL injection and XSS detection."""

    @pytest.mark.unit
    def test_detects_sql_injection(self):
        validator = InputValidator()
        assert validator.check_sql_injection("'; DROP TABLE users; --") is True

    @pytest.mark.unit
    def test_detects_sql_union(self):
        validator = InputValidator()
        assert validator.check_sql_injection("UNION SELECT * FROM passwords") is True

    @pytest.mark.unit
    def test_safe_clinical_query(self):
        validator = InputValidator()
        assert (
            validator.check_sql_injection("What is the post-operative pain protocol?")
            is False
        )

    @pytest.mark.unit
    def test_detects_xss_script(self):
        validator = InputValidator()
        assert validator.check_xss("<script>alert('xss')</script>") is True

    @pytest.mark.unit
    def test_detects_xss_event_handler(self):
        validator = InputValidator()
        assert validator.check_xss("onerror=alert(1)") is True

    @pytest.mark.unit
    def test_safe_text_no_xss(self):
        validator = InputValidator()
        assert validator.check_xss("Normal clinical question here") is False

    @pytest.mark.unit
    def test_is_safe_combined(self):
        validator = InputValidator()
        assert validator.is_safe("What is the treatment for hypertension?") is True
        assert validator.is_safe("'; DROP TABLE --") is False

    @pytest.mark.unit
    def test_sanitize_null_bytes(self):
        validator = InputValidator()
        assert "\x00" not in validator.sanitize("hello\x00world")

    @pytest.mark.unit
    def test_sanitize_control_chars(self):
        validator = InputValidator()
        result = validator.sanitize("test\x01\x02\x03text")
        assert result == "testtext"


# ============================================
# Rate Limiter Tests
# ============================================


class TestRateLimiter:
    """Tests for rate limiting."""

    @pytest.mark.unit
    def test_allows_under_limit(self):
        limiter = RateLimiter()
        allowed, retry_after = limiter.is_allowed("user:1")
        assert allowed is True
        assert retry_after == 0

    @pytest.mark.unit
    def test_blocks_over_limit(self):
        limiter = RateLimiter()
        for _ in range(RATE_LIMIT):
            limiter.is_allowed("user:2")
        allowed, retry_after = limiter.is_allowed("user:2")
        assert allowed is False
        assert retry_after > 0

    @pytest.mark.unit
    def test_different_users_independent(self):
        limiter = RateLimiter()
        for _ in range(RATE_LIMIT):
            limiter.is_allowed("user:3")
        # user:3 is blocked
        assert limiter.is_allowed("user:3")[0] is False
        # user:4 is still allowed
        assert limiter.is_allowed("user:4")[0] is True

    @pytest.mark.unit
    def test_get_remaining(self):
        limiter = RateLimiter()
        assert limiter.get_remaining("user:5") == RATE_LIMIT
        limiter.is_allowed("user:5")
        assert limiter.get_remaining("user:5") == RATE_LIMIT - 1
