"""Streamlit application for RoboUI: Local Vision Inference & Streamlit UI Integration."""

import io
import sys
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image
import streamlit as st

# Ensure adjacent modules in the same directory can always be resolved
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from config import RoboUIConfig
    from engine import get_engine, GenerationResult, ModelFileNotFoundError, OutOfMemoryInferenceError
    from extractor import wrap_in_airgap_sandbox
except ImportError:
    from roboui.config import RoboUIConfig
    from roboui.engine import get_engine, GenerationResult, ModelFileNotFoundError, OutOfMemoryInferenceError
    from roboui.extractor import wrap_in_airgap_sandbox


def validate_uploaded_image(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """Validate that uploaded bytes represent a valid PNG or JPEG image file."""
    if not file_bytes:
        return False, "Empty file received."
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
        return True, None
    except Exception as exc:
        return False, f"Invalid image format: {exc}. Please upload a valid PNG or JPEG image."


def build_iframe_html(html_content: str) -> str:
    """Format and wrap HTML content inside the air-gapped sandbox wrapper."""
    return wrap_in_airgap_sandbox(html_content)


def main():
    st.set_page_config(
        page_title="RoboUI: Local Vision Sketch-to-Code",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state keys
    if "generation_result" not in st.session_state:
        st.session_state["generation_result"] = None
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state["uploaded_image_bytes"] = None

    # Sidebar: Model and Hardware Configuration (BYOM & VRAM limits)
    st.sidebar.title("⚙️ Local Vision Hardware")
    mock_mode = st.sidebar.toggle(
        "Enable Mock Mode (Testing Only)",
        value=False,
        help="Run without local GGUF weights for dry testing.",
    )

    max_context = st.sidebar.slider(
        "Context Window Throttling (n_ctx)",
        min_value=512,
        max_value=8192,
        value=2048,
        step=256,
        help="Strict upper bound on context tokens to prevent VRAM exhaustion.",
    )

    n_gpu_layers = st.sidebar.slider(
        "GPU Offload Layers (n_gpu_layers)",
        min_value=0,
        max_value=64,
        value=0,
        step=1,
        help="Number of layers to offload to GPU VRAM via llama.cpp (0 for CPU only).",
    )

    default_config = RoboUIConfig.from_env()

    model_path = st.sidebar.text_input(
        "BYOM Model Path (.gguf)",
        value=default_config.model_path,
        help="Path to local quantized GGUF weights.",
    )

    mmproj_path = st.sidebar.text_input(
        "Vision Projector (.gguf)",
        value=default_config.mmproj_path or "",
        help="Path to multimodal CLIP/SigLIP vision projector.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("🛡️ **Air-Gapped Sandbox:** Active (`default-src 'none'`)")
    st.sidebar.caption("🔒 **PII Scrubbing:** Active on intermediate text streams")
    st.sidebar.caption("🎨 **Tailwind CSS:** Inlined offline without external CDNs")

    # Header
    st.title("⚡ RoboUI: Local Vision Image-to-Code Web Generator")
    st.markdown(
        "Upload a hand-drawn UI sketch (.png, .jpg) to generate clean HTML/Tailwind/JS code "
        "using local Gemma multimodal vision and LangChain."
    )

    # Top Section: File Uploader and Prompts
    upload_col, prompt_col = st.columns([1, 1], gap="medium")

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload UI Sketch (.png, .jpg)",
            type=["png", "jpg", "jpeg"],
            help="Upload your hand-drawn wireframe or UI design sketch.",
        )

        image_bytes = None
        if uploaded_file is not None:
            raw_bytes = uploaded_file.read()
            is_valid, err_msg = validate_uploaded_image(raw_bytes)
            if is_valid:
                image_bytes = raw_bytes
                st.session_state["uploaded_image_bytes"] = image_bytes
                st.image(image_bytes, caption="Uploaded Wireframe Sketch", use_container_width=True)
            else:
                st.error(err_msg)

    with prompt_col:
        custom_notes = st.text_area(
            "Design Specifications & Notes",
            value="Build a responsive SaaS dashboard with dark theme and interactive analytics card.",
            height=120,
            help="Additional hints for layout, color themes, or components. Tested for PII scrubbing.",
        )

        generate_clicked = st.button("🚀 Synthesize Web Application", type="primary", use_container_width=True)

    config = RoboUIConfig(
        mock_mode=mock_mode,
        max_context_tokens=max_context,
        n_gpu_layers=n_gpu_layers,
        model_path=model_path,
        mmproj_path=mmproj_path.strip() if mmproj_path.strip() else None,
        inlined_tailwind=True,
        enforce_csp=True,
    )

    # Trigger Real Generation with Progress Spinner and Error Boundaries
    if generate_clicked:
        with st.spinner("Analyzing sketch with Gemma vision and synthesizing code..."):
            try:
                engine = get_engine(config, fallback_to_mock=False)
                active_image = image_bytes or st.session_state.get("uploaded_image_bytes")
                result = engine.generate_ui(image_bytes=active_image, custom_notes=custom_notes)
                st.session_state["generation_result"] = result
                st.success("Web UI generated successfully!")
            except ModelFileNotFoundError as exc:
                st.error(f"❌ Local GGUF Model File Not Found: {exc}")
                st.info("💡 To run on-device inference, place your .gguf model weights in the models/ directory or specify the exact path in the sidebar. You can also toggle 'Enable Mock Mode (Testing Only)' in the sidebar for offline testing.")
            except (OutOfMemoryInferenceError, MemoryError) as oom_exc:
                st.error(f"⚠️ GPU / VRAM Out Of Memory: {oom_exc}")
                st.info("💡 Try reducing Context Window Throttling (n_ctx) or reducing GPU Offload Layers (n_gpu_layers) in the sidebar.")
            except Exception as exc:
                st.error(f"❌ Inference Failure: {exc}")

    # Output Section: Two-Column Layout (Code on left, Live Preview on right)
    result: Optional[GenerationResult] = st.session_state.get("generation_result")

    if result:
        st.markdown("---")
        code_col, preview_col = st.columns([1, 1], gap="large")

        # Left Column: Raw Extracted Code in Expandable Text Area
        with code_col:
            st.subheader("📄 Extracted Code")
            with st.expander("Expand Raw Extracted Code", expanded=True):
                st.text_area(
                    label="HTML / Tailwind / Vanilla JavaScript",
                    value=result.html_code,
                    height=460,
                    help="Clean code extracted from markdown code fences.",
                )

            st.download_button(
                label="💾 Download Standalone HTML App",
                data=result.html_code,
                file_name="roboui-generated-app.html",
                mime="text/html",
                use_container_width=True,
            )

            # Metadata & PII summary
            with st.expander("Security & Performance Audit", expanded=False):
                st.caption(f"**Tokens Synthesized:** {result.tokens_used}")
                st.caption(f"**Latency:** {result.latency_ms:.1f} ms")
                st.caption(f"**PII Scrubbed:** {result.sanitized_pii_count} items")
                if result.redacted_pii:
                    st.warning(f"Redacted elements: {', '.join(result.redacted_pii)}")

        # Right Column: Live Website Preview via streamlit.components.v1.html
        with preview_col:
            st.subheader("🌐 Live Website Preview")
            sandboxed_html = build_iframe_html(result.html_code)
            st.components.v1.html(sandboxed_html, height=520, scrolling=True)


if __name__ == "__main__":
    main()
