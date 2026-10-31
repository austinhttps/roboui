import pytest
from roboui.config import RoboUIConfig
from roboui.engine import (
    get_engine,
    GenerationResult,
    ModelFileNotFoundError,
    LlamaCppInferenceEngine,
)

def test_mock_engine_generation():
    config = RoboUIConfig(mock_mode=True)
    engine = get_engine(config)
    result = engine.generate_ui(image_bytes=b"fake-sketch", custom_notes="Add modern pricing table")
    assert isinstance(result, GenerationResult)
    assert "<!DOCTYPE html>" in result.html_code
    assert result.tokens_used > 0
    assert result.latency_ms >= 0
    assert result.sanitized_pii_count >= 0
    assert len(result.semantic_description) > 0

def test_llama_engine_missing_weights_raises_actionable_error():
    config = RoboUIConfig(mock_mode=False, model_path="models/nonexistent-model.gguf")
    with pytest.raises(ModelFileNotFoundError) as exc_info:
        LlamaCppInferenceEngine(config)
    assert "models/nonexistent-model.gguf" in str(exc_info.value)
    assert "Download GGUF weights or run in mock mode" in str(exc_info.value)

def test_vram_exhaustion_fallback():
    config = RoboUIConfig(mock_mode=False, model_path="models/dummy.gguf", n_gpu_layers=33)
    engine = LlamaCppInferenceEngine.__new__(LlamaCppInferenceEngine)
    engine.config = config
    fallback_occurred = engine._handle_vram_failure(RuntimeError("CUDA out of memory"))
    assert fallback_occurred is True
    assert engine.config.n_gpu_layers == 0
