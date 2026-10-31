"""Code extraction pipeline and air-gapped sandboxing wrapper for RoboUI."""

import os
import re
from pathlib import Path

# Paths to static offline assets
STATIC_DIR = Path(__file__).parent / "static"
TAILWIND_PATH = STATIC_DIR / "tailwind.min.css"

# Precompiled regex patterns for extraction
FENCED_BLOCK_PATTERN = re.compile(r"```(?:html)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)
UNCLOSED_FENCE_PATTERN = re.compile(r"```(?:html)?\s*\n?(.*)", re.DOTALL | re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"(<(?:!DOCTYPE|html|div|main|section|header|body)[\s>].*)", re.DOTALL | re.IGNORECASE)


def load_offline_css() -> str:
    """Read the bundled offline CSS reset / utility file."""
    if TAILWIND_PATH.exists():
        return TAILWIND_PATH.read_text(encoding="utf-8")
    return "/* offline reset */ body { font-family: sans-serif; background: #0f172a; color: #fff; padding: 1rem; }"


def _auto_close_tags(html: str) -> str:
    """Auto-close unclosed structural tags caused by token truncation."""
    repaired = html.strip()
    
    # Auto-close common container tags
    for tag in ["div", "section", "main", "body", "html"]:
        open_count = len(re.findall(rf"<{tag}[\s>]", repaired, re.IGNORECASE))
        close_count = len(re.findall(rf"</{tag}>", repaired, re.IGNORECASE))
        if open_count > close_count:
            repaired += ("\n" + f"</{tag}>" * (open_count - close_count))

    return repaired


def extract_html(llm_output: str) -> str:
    """Extract and sanitize raw HTML from LLM completions.

    Handles complete markdown fences, unclosed fences, and raw HTML fragments.
    """
    if not llm_output or not llm_output.strip():
        return ""

    text = llm_output.strip()

    # 1. Closed markdown code block
    fenced_match = FENCED_BLOCK_PATTERN.search(text)
    if fenced_match:
        extracted = fenced_match.group(1).strip()
        return _auto_close_tags(extracted)

    # 2. Unclosed markdown code block (context truncation)
    unclosed_match = UNCLOSED_FENCE_PATTERN.search(text)
    if unclosed_match:
        extracted = unclosed_match.group(1).strip()
        extracted = re.sub(r"```+$", "", extracted).strip()
        return _auto_close_tags(extracted)

    # 3. Direct HTML tags in text
    tag_match = HTML_TAG_PATTERN.search(text)
    if tag_match:
        extracted = tag_match.group(1).strip()
        last_gt = extracted.rfind(">")
        if last_gt != -1:
            extracted = extracted[:last_gt + 1]
        return _auto_close_tags(extracted)

    return _auto_close_tags(text)


def wrap_in_airgap_sandbox(html_content: str, inline_css: str | None = None) -> str:
    """Wrap HTML in an air-gapped container with strict CSP and inlined CSS.
    
    Idempotent: if CSP is already injected, returns the existing HTML.
    """
    clean_html = html_content.strip()

    # Idempotency check: don't double inject CSP or styles
    if 'http-equiv="Content-Security-Policy"' in clean_html:
        return clean_html

    css_payload = inline_css if inline_css is not None else load_offline_css()
    csp_meta = (
        '<meta http-equiv="Content-Security-Policy" '
        'content="default-src \'none\'; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'; img-src \'self\' data:;">'
    )
    style_block = f"<style>{css_payload}</style>" if css_payload else ""

    # If already a full document, inject into <head>
    if "<head>" in clean_html.lower():
        head_pattern = re.compile(r"(<head[^>]*>)", re.IGNORECASE)
        return head_pattern.sub(rf"\1\n    {csp_meta}\n    {style_block}", clean_html, count=1)
    elif "<html" in clean_html.lower():
        html_pattern = re.compile(r"(<html[^>]*>)", re.IGNORECASE)
        return html_pattern.sub(rf"\1\n<head>\n    {csp_meta}\n    {style_block}\n</head>", clean_html, count=1)

    # Wrap snippet into full HTML document
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    {csp_meta}
    {style_block}
</head>
<body class="bg-slate-900 text-white min-h-screen">
{clean_html}
</body>
</html>"""
