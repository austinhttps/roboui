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
    model_path: str = "models/gemma-2b-instruct-q4.gguf"
    inlined_tailwind: bool = True
    enforce_csp: bool = True
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RoboUIConfig":
        """Instantiate configuration parsed from environment variables."""
        mock_mode = _parse_bool(os.getenv("ROBOUI_MOCK_MODE"), default=True)

        max_ctx_str = os.getenv("ROBOUI_MAX_CONTEXT_TOKENS", "2048").strip()
        max_context_tokens = int(max_ctx_str)

        gpu_layers_str = os.getenv("ROBOUI_N_GPU_LAYERS", "0").strip()
        n_gpu_layers = int(gpu_layers_str)

        model_path = os.getenv("ROBOUI_MODEL_PATH", "models/gemma-2b-instruct-q4.gguf").strip()
        inlined_tailwind = _parse_bool(os.getenv("ROBOUI_INLINED_TAILWIND"), default=True)
        enforce_csp = _parse_bool(os.getenv("ROBOUI_ENFORCE_CSP"), default=True)

        api_key = os.getenv("ROBOUI_FALLBACK_API_KEY")
        api_base_url = os.getenv("ROBOUI_FALLBACK_API_BASE_URL")

        return cls(
            mock_mode=mock_mode,
            max_context_tokens=max_context_tokens,
            n_gpu_layers=n_gpu_layers,
            model_path=model_path,
            inlined_tailwind=inlined_tailwind,
            enforce_csp=enforce_csp,
            api_key=api_key,
            api_base_url=api_base_url,
        )
