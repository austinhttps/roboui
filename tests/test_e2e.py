import io
from PIL import Image
from roboui.config import RoboUIConfig
from roboui.engine import get_engine
from roboui.app import build_iframe_html, validate_uploaded_image

def test_e2e_pipeline_flow():
    # 1. Initialize configuration in Mock mode
    config = RoboUIConfig(mock_mode=True, max_context_tokens=2048)
    engine = get_engine(config)
    
    # 2. Generate a valid mock sketch image
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="lightblue").save(buf, format="PNG")
    sketch_bytes = buf.getvalue()
    valid, err = validate_uploaded_image(sketch_bytes)
    assert valid is True

    # 3. Supply notes with high-entropy sensitive keys and emails
    notes = "Build a sign-up card for dev@company.com with token sk-live-123456789012345678901234567890"
    
    # 4. Execute generation pipeline
    result = engine.generate_ui(sketch_bytes, custom_notes=notes)
    
    # 5. Verify PII scrubbing occurred
    assert "dev@company.com" not in result.semantic_description
    assert "sk-live-" not in result.semantic_description
    assert "EMAIL" in result.redacted_pii
    assert "SECRET_KEY" in result.redacted_pii
    assert result.sanitized_pii_count == 2
    
    # 6. Verify HTML structure & sandbox wrapper
    assert "<!DOCTYPE html>" in result.html_code
    iframe_view = build_iframe_html(result.html_code)
    assert "Content-Security-Policy" in iframe_view
    assert "default-src 'none'" in iframe_view
    assert result.tokens_used > 0
    assert result.latency_ms >= 0.0
