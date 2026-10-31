import io
from PIL import Image
from roboui.app import build_iframe_html, validate_uploaded_image

def test_build_iframe_html():
    raw_html = "<button class='bg-blue-500'>Click</button>"
    iframe_code = build_iframe_html(raw_html)
    assert "<button" in iframe_code
    assert "Content-Security-Policy" in iframe_code
    assert "default-src 'none'" in iframe_code

def test_validate_image_valid():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color="white").save(buf, format="PNG")
    valid_bytes = buf.getvalue()

    valid, err = validate_uploaded_image(valid_bytes)
    assert valid is True
    assert err is None

def test_validate_image_invalid():
    valid, err = validate_uploaded_image(b"not an image file")
    assert valid is False
    assert "Invalid image format" in err
