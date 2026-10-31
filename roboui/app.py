"""Streamlit application for RoboUI: On-Device Image-to-Code Web Generator."""

import io
from typing import Optional, Tuple
from PIL import Image
import streamlit as st

from roboui.config import RoboUIConfig
from roboui.engine import get_engine, GenerationResult, ModelFileNotFoundError
from roboui.extractor import wrap_in_airgap_sandbox


def validate_uploaded_image(file_bytes: bytes) -> Tuple[bool, Optional[str]]:
    """Validate that uploaded bytes represent a valid image file."""
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
        page_title="RoboUI: On-Device Image-to-Code Generator",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state keys
    if "generation_result" not in st.session_state:
        st.session_state["generation_result"] = None
    if "uploaded_image_bytes" not in st.session_state:
        st.session_state["uploaded_image_bytes"] = None

    # Sidebar: Model and Hardware Configuration
    st.sidebar.title("⚙️ Engine & Hardware")
    mock_mode = st.sidebar.toggle("Enable Mock Mode (Zero-Weight)", value=True, help="Run without downloading multi-gigabyte GGUF model files.")
    
    max_context = st.sidebar.slider(
        "Context Window Throttling (tokens)",
        min_value=512,
        max_value=8192,
        value=2048,
        step=256,
        help="Throttles maximum context window to prevent VRAM allocation crashes.",
    )
    
    n_gpu_layers = st.sidebar.slider(
        "GPU Layers (VRAM Offload)",
        min_value=0,
        max_value=64,
        value=0,
        step=1,
        help="0 uses CPU only; >0 offloads layers to local GPU via llama.cpp.",
    )

    model_path = st.sidebar.text_input("Local Model Path", value="models/gemma-2b-instruct-q4.gguf")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ Privacy & Sandboxing")
    st.sidebar.caption("✅ **Process Network Isolation:** Outbound traffic blocked")
    st.sidebar.caption("✅ **Air-Gapped CSP:** `default-src 'none'` enforced")
    st.sidebar.caption("✅ **PII Scrubbing:** Active on intermediate representations")
    st.sidebar.caption("✅ **Tailwind CSS:** Inlined offline without external CDNs")

    # App Header
    st.title("⚡ RoboUI: On-Device Sketch-to-Code Generator")
    st.markdown(
        "Convert hand-drawn interface sketches and wireframes into clean, air-gapped "
        "HTML/Tailwind web applications using local models and LangChain."
    )

    config = RoboUIConfig(
        mock_mode=mock_mode,
        max_context_tokens=max_context,
        n_gpu_layers=n_gpu_layers,
        model_path=model_path,
        inlined_tailwind=True,
        enforce_csp=True,
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Upload Sketch & Style Notes")
        uploaded_file = st.file_uploader("Upload wireframe drawing or sketch", type=["png", "jpg", "jpeg"])
        
        image_bytes = None
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            is_valid, err_msg = validate_uploaded_image(image_bytes)
            if not is_valid:
                st.error(err_msg)
                image_bytes = None
            else:
                st.image(image_bytes, caption="Uploaded UI Sketch", use_container_width=True)
                st.session_state["uploaded_image_bytes"] = image_bytes
        
        custom_notes = st.text_area(
            "Design Specifications & Notes",
            value="Build a responsive SaaS dashboard with dark theme and interactive analytics card.",
            height=100,
            help="Additional hints for layout, color themes, or components. Tested for PII scrubbing.",
        )

        generate_clicked = st.button("🚀 Generate Air-Gapped Web UI", type="primary", use_container_width=True)

        if generate_clicked:
            with st.spinner("Synthesizing UI with on-device pipeline..."):
                try:
                    engine = get_engine(config)
                    result = engine.generate_ui(image_bytes=image_bytes, custom_notes=custom_notes)
                    st.session_state["generation_result"] = result
                    st.success("Web application generated successfully!")
                except ModelFileNotFoundError as exc:
                    st.error(f"Model Error: {exc}")
                except Exception as exc:
                    st.error(f"Inference Failure: {exc}")

    with col2:
        st.subheader("2. Generated Output & Sandbox")
        result: Optional[GenerationResult] = st.session_state.get("generation_result")

        if result:
            tab_preview, tab_code, tab_audit = st.tabs([
                "🌐 Live Air-Gapped Preview",
                "📄 HTML / CSS Source",
                "🔍 Security & Audit",
            ])

            with tab_preview:
                sandboxed_html = build_iframe_html(result.html_code)
                st.components.v1.html(sandboxed_html, height=520, scrolling=True)
                st.download_button(
                    label="💾 Download Standalone HTML App",
                    data=result.html_code,
                    file_name="roboui-generated-app.html",
                    mime="text/html",
                    use_container_width=True,
                )

            with tab_code:
                st.code(result.html_code, language="html")

            with tab_audit:
                m1, m2, m3 = st.columns(3)
                m1.metric("Latency", f"{result.latency_ms:.1f} ms")
                m2.metric("Tokens Synthesized", result.tokens_used)
                m3.metric("PII Scrubbed", result.sanitized_pii_count)

                st.markdown("#### Intermediate Semantic Representation")
                st.info(result.semantic_description)

                if result.redacted_pii:
                    st.warning(f"Redacted sensitive items: {', '.join(result.redacted_pii)}")
                else:
                    st.success("No sensitive PII patterns detected in prompt or layout.")
        else:
            st.info("Upload a sketch or click **Generate Air-Gapped Web UI** to view live results.")


if __name__ == "__main__":
    main()
