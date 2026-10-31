from roboui.sanitizer import sanitize_pii

def test_sanitize_emails():
    text = "Contact me at alice.smith@company.org or support@domain.io for help."
    clean, matches = sanitize_pii(text)
    assert "[REDACTED_EMAIL]" in clean
    assert "alice.smith@company.org" not in clean
    assert "support@domain.io" not in clean
    assert "EMAIL" in matches

def test_sanitize_api_keys():
    text = "Use key sk-live-abcdef1234567890abcdef1234567890 or ghp_abcdefghijklmnopqrstuvwxyz1234567890."
    clean, matches = sanitize_pii(text)
    assert "[REDACTED_SECRET]" in clean
    assert "sk-live-" not in clean
    assert "ghp_" not in clean
    assert "SECRET_KEY" in matches

def test_sanitize_phone_and_ip():
    text = "Server at 192.168.1.100 call (555) 123-4567"
    clean, matches = sanitize_pii(text)
    assert "[REDACTED_IP]" in clean
    assert "[REDACTED_PHONE]" in clean
    assert "192.168.1.100" not in clean
    assert "(555) 123-4567" not in clean
    assert set(matches) == {"IP_ADDRESS", "PHONE"}

def test_sanitize_clean_text():
    text = "Simple navbar with Home, About, and Contact links."
    clean, matches = sanitize_pii(text)
    assert clean == text
    assert matches == []
