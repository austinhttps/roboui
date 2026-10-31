import os
import pytest
from roboui.config import RoboUIConfig

def test_default_config():
    config = RoboUIConfig()
    assert config.mock_mode is True
    assert config.max_context_tokens == 2048
    assert config.n_gpu_layers == 0
    assert config.model_path.endswith(".gguf")
    assert config.inlined_tailwind is True

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("ROBOUI_MOCK_MODE", "false")
    monkeypatch.setenv("ROBOUI_MAX_CONTEXT_TOKENS", "4096")
    monkeypatch.setenv("ROBOUI_N_GPU_LAYERS", "33")
    config = RoboUIConfig.from_env()
    assert config.mock_mode is False
    assert config.max_context_tokens == 4096
    assert config.n_gpu_layers == 33
