"""PII scrubbing and regex redaction for RoboUI intermediate text streams."""

import re
from typing import Tuple, List

# Compiled regex patterns and their replacement tokens
PII_RULES = [
    (
        "SECRET_KEY",
        re.compile(
            r"(?:sk-(?:live|test)-[A-Za-z0-9_\-]{16,}|sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{36}|AKIA[0-9A-Z]{16})",
            re.IGNORECASE,
        ),
        "[REDACTED_SECRET]",
    ),
    (
        "EMAIL",
        re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            re.IGNORECASE,
        ),
        "[REDACTED_EMAIL]",
    ),
    (
        "PHONE",
        re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        ),
        "[REDACTED_PHONE]",
    ),
    (
        "IP_ADDRESS",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        ),
        "[REDACTED_IP]",
    ),
]


def sanitize_pii(text: str) -> Tuple[str, List[str]]:
    """Scrub PII and confidential tokens from input text.

    Returns:
        Tuple of (sanitized_text, detected_categories).
    """
    if not text:
        return "", []

    sanitized = text
    detected_categories: List[str] = []

    for category, pattern, replacement in PII_RULES:
        if pattern.search(sanitized):
            sanitized = pattern.sub(replacement, sanitized)
            if category not in detected_categories:
                detected_categories.append(category)

    return sanitized, detected_categories
