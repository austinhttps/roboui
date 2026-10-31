from roboui.extractor import extract_html, wrap_in_airgap_sandbox

def test_extract_markdown_fenced_html():
    raw = "Here is your site:\n```html\n<!DOCTYPE html><html><body><h1>Hello</h1></body></html>\n```\nEnjoy!"
    extracted = extract_html(raw)
    assert extracted.startswith("<!DOCTYPE html>")
    assert "```" not in extracted
    assert "Here is your site" not in extracted

def test_extract_unclosed_fence():
    raw = "```html\n<div class='container'><p>Truncated UI</p>"
    extracted = extract_html(raw)
    assert "<div class='container'><p>Truncated UI</p>" in extracted
    assert "```" not in extracted

def test_extract_raw_html_fallback():
    raw = "I designed this: <div><span>Direct tag</span></div>."
    extracted = extract_html(raw)
    assert "<div><span>Direct tag</span></div>" in extracted

def test_airgap_sandbox_wrapper():
    html = "<div class='bg-blue-500'>Content</div>"
    wrapped = wrap_in_airgap_sandbox(html, inline_css=".bg-blue-500 { color: blue; }")
    assert "<meta http-equiv=\"Content-Security-Policy\"" in wrapped
    assert "default-src 'none';" in wrapped
    assert "style-src 'unsafe-inline';" in wrapped
    assert "script-src 'unsafe-inline';" in wrapped
    assert "<style>.bg-blue-500 { color: blue; }</style>" in wrapped
