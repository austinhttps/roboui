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
    system_msg = next(m for m in messages if m["role"] == "system")
    assert "strict eye for 1:1 design translation" in system_msg["content"]
    assert "Tailwind CSS grid, flexbox, and relative fractional width classes" in system_msg["content"]
    assert "rounded-xl" in system_msg["content"]
    assert "hover:bg-blue-600" in system_msg["content"]
    assert "px-6 py-2.5" in system_msg["content"]
    assert "focus:ring-1 focus:ring-blue-500" in system_msg["content"]
    
    # Verify user message contains base64 image data URI
    user_msg = next(m for m in messages if m["role"] == "user")
    content_list = user_msg["content"]
    text_part = next(c for c in content_list if c["type"] == "text")
    assert "Pay strict attention to the widths, aspect ratios, and relative positioning" in text_part["text"]
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


def test_gemma_vision_service_refine_ui_retains_strict_spatial_prompt():
    config = RoboUIConfig(mock_mode=False, model_path="models/dummy.gguf")
    service = GemmaVisionService.__new__(GemmaVisionService)
    service.config = config
    service.fallback_to_mock = False
    service.is_degraded = False
    service._mock_delegate = None

    mock_model = MagicMock()
    mock_model.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": "```html\n<div class='w-1/2 mx-auto'><button>Dark Button</button></div>\n```"
                }
            }
        ],
        "usage": {"total_tokens": 120},
    }
    service.model = mock_model

    existing = "<div class='w-1/2 mx-auto'><button>Light Button</button></div>"
    result = service.refine_ui(existing_code=existing, refinement_instruction="Make button dark mode")

    assert mock_model.create_chat_completion.called
    call_args = mock_model.create_chat_completion.call_args[1]
    messages = call_args["messages"]

    system_msg = next(m for m in messages if m["role"] == "system")
    assert "strict eye for 1:1 design translation" in system_msg["content"]
    assert "Analyze the relative widths, heights, alignment, and spatial positioning" in system_msg["content"]
    assert "Tailwind CSS grid, flexbox, and relative fractional width classes" in system_msg["content"]
    assert "w-1/2" in system_msg["content"]
    assert "rounded-xl" in system_msg["content"]
    assert "hover:bg-blue-600" in system_msg["content"]
    assert "px-6 py-2.5" in system_msg["content"]
    assert "focus:ring-1 focus:ring-blue-500" in system_msg["content"]

    # Verify extracted code is clean HTML without ``` markdown tags
    assert "```" not in result.html_code
    assert "Dark Button" in result.html_code

