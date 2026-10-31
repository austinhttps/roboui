"""Runtime configuration for RoboUI."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load optional .env file
load_dotenv()

@dataclass
class RoboUIConfig:
    """Settings controlling inference engine, hardware resources, and sandbox policies."""

    mock_mode: bool = True
    max_context_tokens: int = 2048
    n_gpu_layers: int = 0
    model_path: str = "models/gemma-2b-instruct-q4.gguf"
    inlined_tailwind: bool = True
    enforce_csp: bool = True

    @classmethod
    def from_env(cls) -> "RoboUIConfig":
        """Instantiate configuration parsed from environment variables."""
        mock_mode_str = os.getenv("ROBOUI_MOCK_MODE", "true").strip().lower()
        mock_mode = mock_mode_str in ("1", "true", "yes", "on")

        max_ctx_str = os.getenv("ROBOUI_MAX_CONTEXT_TOKENS", "2048").strip()
        max_context_tokens = int(max_ctx_str)

        gpu_layers_str = os.getenv("ROBOUI_N_GPU_LAYERS", "0").strip()
        n_gpu_layers = int(gpu_layers_str)

        model_path = os.getenv("ROBOUI_MODEL_PATH", "models/gemma-2b-instruct-q4.gguf").strip()

        inlined_tailwind_str = os.getenv("ROBOUI_INLINED_TAILWIND", "true").strip().lower()
        inlined_tailwind = inlined_tailwind_str in ("1", "true", "yes", "on")

        enforce_csp_str = os.getenv("ROBOUI_ENFORCE_CSP", "true").strip().lower()
        enforce_csp = enforce_csp_str in ("1", "true", "yes", "on")

        return cls(
            mock_mode=mock_mode,
            max_context_tokens=max_context_tokens,
            n_gpu_layers=n_gpu_layers,
            model_path=model_path,
            inlined_tailwind=inlined_tailwind,
            enforce_csp=enforce_csp,
        )
