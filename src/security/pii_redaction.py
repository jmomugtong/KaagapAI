"""
PII Redaction for MedQuery (Phase 7)

Detects and redacts personally identifiable information from text.
Targets >95% accuracy on common PII patterns found in clinical documents.

Redaction tokens:
- [PATIENT_NAME], [MRN], [DOB], [PHONE], [SSN], [EMAIL]
"""

import logging
import re

logger = logging.getLogger(__name__)

# Pattern definitions with named groups
PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # SSN: 123-45-6789 or 123456789 (9 digits)
    (
        "[SSN]",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "[SSN]",
        re.compile(r"\b\d{9}\b"),
    ),
    # Email addresses
    (
        "[EMAIL]",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    ),
    # Phone numbers: (555) 123-4567, 555-123-4567, 555.123.4567
    (
        "[PHONE]",
        re.compile(r"\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),
    ),
    # Dates of birth: MM/DD/YYYY, YYYY-MM-DD, MM-DD-YYYY
    (
        "[DOB]",
        re.compile(r"\b(?:\d{2}[/\-]\d{2}[/\-]\d{4}|\d{4}[/\-]\d{2}[/\-]\d{2})\b"),
    ),
    # Medical Record Numbers: alphanumeric 8-12 chars (e.g., ABC12345678)
    (
        "[MRN]",
        re.compile(r"\b(?:MRN[:\s]*)?[A-Z]{2,4}\d{6,10}\b", re.IGNORECASE),
    ),
    # Patient names following common prefixes
    (
        "[PATIENT_NAME]",
        re.compile(
            r"(?:(?:[Pp]atient|[Mm]r\.?|[Mm]rs\.?|[Mm]s\.?|[Dd]r\.?)\s+)"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
        ),
    ),
]


class PIIRedactor:
    """Detects and redacts PII from text."""

    def __init__(self) -> None:
        self._patterns = PII_PATTERNS

    def redact(self, text: str) -> str:
        """
        Scan text and replace all detected PII with redaction tokens.

        Args:
            text: Input text potentially containing PII.

        Returns:
            Text with PII replaced by tokens like [PATIENT_NAME], [SSN], etc.
        """
        if not text:
            return text

        redacted = text
        for token, pattern in self._patterns:
            redacted = pattern.sub(token, redacted)

        return redacted

    def detect(self, text: str) -> list[dict[str, str]]:
        """
        Detect PII in text without redacting.

        Returns list of dicts with 'type', 'value', 'start', 'end'.
        """
        if not text:
            return []

        findings: list[dict[str, str]] = []
        for token, pattern in self._patterns:
            for match in pattern.finditer(text):
                findings.append(
                    {
                        "type": token,
                        "value": match.group(),
                        "start": str(match.start()),
                        "end": str(match.end()),
                    }
                )
        return findings

    def has_pii(self, text: str) -> bool:
        """Check if text contains any PII."""
        return len(self.detect(text)) > 0
