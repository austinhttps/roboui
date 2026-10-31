import pytest
from unittest.mock import MagicMock
from roboui.config import RoboUIConfig
from roboui.engine import (
    GemmaVisionService,
    ModelFileNotFoundError,
    OutOfMemoryInferenceError,
)

def test_gemma_vision_service_strictly_raises_on_missing_model():
    config = RoboUIConfig(mock_mode=False, model_path="models/missing-model.gguf")
    with pytest.raises(ModelFileNotFoundError) as exc_info:
        GemmaVisionService(config, fallback_to_mock=False)
    assert "models/missing-model.gguf" in str(exc_info.value)

def test_gemma_vision_service_multimodal_payload_and_regex_extraction():
    config = RoboUIConfig(mock_mode=False, model_path="models/dummy.gguf")
    service = GemmaVisionService.__new__(GemmaVisionService)
    service.config = config
    service.fallback_to_mock = False
    service.is_degraded = False
    service._mock_delegate = None

    # Mock Llama model with create_chat_completion
    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": "```html\n<div class='hero'><button>Launch</button></div>\n```"
                }
            }
        ],
        "usage": {"total_tokens": 142},
    }
    service.model = mock_model

    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    result = service.generate_ui(image_bytes=fake_png, custom_notes="Add modern CTA button")

    # Verify create_chat_completion was invoked
    assert mock_model.create_chat_completion.called
    call_args = mock_model.create_chat_completion.call_args[1]
    messages = call_args["messages"]
    
    # Verify system prompt exists
    assert any(m["role"] == "system" for m in messages)
    
    # Verify user message contains base64 image data URI
    user_msg = next(m for m in messages if m["role"] == "user")
    content_list = user_msg["content"]
    image_part = next(c for c in content_list if c["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    # Verify output code was extracted via regex (no ```html fences)
    assert "<button>Launch</button>" in result.html_code
    assert "```" not in result.html_code
    assert "Content-Security-Policy" in result.html_code

def test_gemma_vision_service_raises_oom_without_mock_fallback():
    config = RoboUIConfig(mock_mode=False, model_path="models/dummy.gguf")
    service = GemmaVisionService.__new__(GemmaVisionService)
    service.config = config
    service.fallback_to_mock = False
    service.is_degraded = False
    service._mock_delegate = None

    mock_model = MagicMock()
    mock_model.create_chat_completion.side_effect = MemoryError("CUDA out of memory")
    service.model = mock_model

    fake_png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    with pytest.raises(OutOfMemoryInferenceError) as exc_info:
        service.generate_ui(image_bytes=fake_png, custom_notes="Big model")
    assert "VRAM/RAM exhaustion" in str(exc_info.value)
