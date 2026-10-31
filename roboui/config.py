"""Runtime configuration for RoboUI."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load optional .env file
load_dotenv()


def _parse_bool(value: Optional[str], default: bool) -> bool:
    """Parse common truthy string values."""
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class RoboUIConfig:
    """Settings controlling inference engine, hardware resources, and sandbox policies."""

    mock_mode: bool = True
    max_context_tokens: int = 2048
    n_gpu_layers: int = 0
    n_threads: Optional[int] = None
    model_path: str = r"C:\Users\Austi\gemma-4-E2B-it-UD-Q2_K_XL.gguf"
    mmproj_path: Optional[str] = r"C:\Users\Austi\mmproj-F16.gguf"
    inlined_tailwind: bool = True
    enforce_csp: bool = True
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RoboUIConfig":
        """Instantiate configuration parsed from environment variables."""
        mock_mode = _parse_bool(os.getenv("ROBOUI_MOCK_MODE"), default=True)

        # Context window throttling (support ROBOUI_N_CTX and ROBOUI_MAX_CONTEXT_TOKENS)
        max_ctx_val = os.getenv("ROBOUI_N_CTX") or os.getenv("ROBOUI_MAX_CONTEXT_TOKENS", "2048")
        max_context_tokens = int(max_ctx_val.strip())

        # GPU layers
        gpu_layers_val = os.getenv("ROBOUI_N_GPU_LAYERS", "0").strip()
        n_gpu_layers = int(gpu_layers_val)

        # CPU threads allocation (default: physical cores / 8 if detected)
        threads_env = os.getenv("ROBOUI_N_THREADS")
        n_threads = int(threads_env.strip()) if threads_env else max(1, (os.cpu_count() or 4) // 2)

        # Model path: explicitly check LLM_MODEL_PATH, ROBOUI_GGUF_PATH, and ROBOUI_MODEL_PATH
        model_path = (
            os.getenv("LLM_MODEL_PATH")
            or os.getenv("ROBOUI_GGUF_PATH")
            or os.getenv("ROBOUI_MODEL_PATH")
            or r"C:\Users\Austi\gemma-4-E2B-it-UD-Q2_K_XL.gguf"
        ).strip()

        # Multimodal vision projector path: explicitly check VISION_PROJECTOR_PATH, ROBOUI_MMPROJ_PATH
        mmproj_env = (
            os.getenv("VISION_PROJECTOR_PATH")
            or os.getenv("ROBOUI_MMPROJ_PATH")
            or r"C:\Users\Austi\mmproj-F16.gguf"
        )
        mmproj_path = mmproj_env.strip() if mmproj_env else None

        inlined_tailwind = _parse_bool(os.getenv("ROBOUI_INLINED_TAILWIND"), default=True)
        enforce_csp = _parse_bool(os.getenv("ROBOUI_ENFORCE_CSP"), default=True)

        api_key = os.getenv("ROBOUI_FALLBACK_API_KEY")
        api_base_url = os.getenv("ROBOUI_FALLBACK_API_BASE_URL")

        return cls(
            mock_mode=mock_mode,
            max_context_tokens=max_context_tokens,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            model_path=model_path,
            mmproj_path=mmproj_path,
            inlined_tailwind=inlined_tailwind,
            enforce_csp=enforce_csp,
            api_key=api_key,
            api_base_url=api_base_url,
        )
