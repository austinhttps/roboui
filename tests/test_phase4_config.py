import os
from roboui.config import RoboUIConfig

def test_phase4_config_paths(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_PATH", r"C:\Users\Austi\gemma-4-E2B-it-UD-Q2_K_XL.gguf")
    monkeypatch.setenv("VISION_PROJECTOR_PATH", r"C:\Users\Austi\mmproj-F16.gguf")
    
    config = RoboUIConfig.from_env()
    assert config.model_path == r"C:\Users\Austi\gemma-4-E2B-it-UD-Q2_K_XL.gguf"
    assert config.mmproj_path == r"C:\Users\Austi\mmproj-F16.gguf"


def test_cpu_threads_config(monkeypatch):
    monkeypatch.setenv("ROBOUI_N_THREADS", "8")
    config = RoboUIConfig.from_env()
    assert config.n_threads == 8

