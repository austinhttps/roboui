import pytest
from roboui.config import RoboUIConfig
from roboui.engine import (
    GemmaVisionService,
    ModelFileNotFoundError,
    GenerationResult,
)

def test_gemma_vision_service_env_parsing(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_PATH", "models/custom-gemma-byom.gguf")
    monkeypatch.setenv("ROBOUI_GGUF_PATH", "models/custom-gemma-byom.gguf")
    monkeypatch.setenv("VISION_PROJECTOR_PATH", "models/custom-mmproj.gguf")
    monkeypatch.setenv("ROBOUI_MMPROJ_PATH", "models/custom-mmproj.gguf")
    monkeypatch.setenv("ROBOUI_N_CTX", "1024")
    monkeypatch.setenv("ROBOUI_N_GPU_LAYERS", "16")
    
    config = RoboUIConfig.from_env()
    assert config.model_path == "models/custom-gemma-byom.gguf"
    assert config.mmproj_path == "models/custom-mmproj.gguf"
    assert config.max_context_tokens == 1024
    assert config.n_gpu_layers == 16

def test_gemma_vision_service_missing_model_raises_error():
    config = RoboUIConfig(
        mock_mode=False,
        model_path="models/nonexistent-gemma.gguf",
    )
    with pytest.raises(ModelFileNotFoundError) as exc_info:
        GemmaVisionService(config, fallback_to_mock=False)
    assert "models/nonexistent-gemma.gguf" in str(exc_info.value)

def test_gemma_vision_service_missing_model_graceful_fallback():
    config = RoboUIConfig(
        mock_mode=False,
        model_path="models/nonexistent-gemma.gguf",
    )
    # With fallback_to_mock=True, missing weights should gracefully degrade without crashing
    service = GemmaVisionService(config, fallback_to_mock=True)
    assert service.is_degraded is True
    result = service.generate_ui(image_bytes=b"fake-image-bytes", custom_notes="Add dark theme")
    assert isinstance(result, GenerationResult)
    assert "<!DOCTYPE html>" in result.html_code

def test_gemma_vision_service_oom_handling():
    config = RoboUIConfig(
        mock_mode=False,
        model_path="models/dummy.gguf",
        n_gpu_layers=32,
    )
    service = GemmaVisionService.__new__(GemmaVisionService)
    service.config = config
    service.is_degraded = False
    
    handled = service._handle_vram_failure(MemoryError("CUDA out of memory during visual token projection"))
    assert handled is True
    assert service.config.n_gpu_layers == 0


def test_optimize_sketch_image():
    from PIL import Image
    import io
    from roboui.engine import optimize_sketch_image

    img = Image.new("RGB", (1200, 1600), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    original_bytes = buf.getvalue()

    optimized = optimize_sketch_image(original_bytes, max_dim=512)
    with Image.open(io.BytesIO(optimized)) as out_img:
        assert max(out_img.size) <= 512
        assert out_img.format == "JPEG"

