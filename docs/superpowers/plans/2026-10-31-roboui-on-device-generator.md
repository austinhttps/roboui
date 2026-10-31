# RoboUI: On-Device Image-to-Code Web Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build "RoboUI", an on-device Python application that transforms hand-drawn UI sketches into runnable HTML/Tailwind/JS web applications using LangChain orchestration, local GGUF quantization (llama.cpp) with a zero-weight mock fallback, deterministic PII scrubbing, and an air-gapped iframe sandbox in Streamlit.

**Architecture:** A two-stage chained LangChain pipeline: Stage 1 analyzes the visual UI sketch to extract an Intermediate Semantic Representation (layout, components, text); PII regex scrubbing masks credentials and personal data; Stage 2 synthesizes self-contained HTML/CSS/JS. Robust multi-strategy regex parsing handles unclosed markdown fences, and an air-gapped Streamlit iframe renders the output without outbound network leaks by inlining a local CSS reset/Tailwind stylesheet.

**Tech Stack:** Python 3.12, LangChain (`langchain-core`), llama-cpp-python (with mock fallback), Streamlit, Pillow, pytest, python-dotenv.

**Spec:** `PROMPT.md`, `GLOSSARY.md`, and `docs/adr/0001-on-device-vision-to-code-architecture.md`.

## Global Constraints

- Never download multi-gigabyte weights during automated build and test passes; default to deterministic Mock Inference Engine.
- Output code must strictly consist of HTML, CSS (Tailwind compatible), and vanilla JavaScript.
- All outbound network calls must be blocked in the air-gapped sandbox (`sandbox="allow-scripts"` on iframe with strict Content-Security-Policy).
- Local styling must be inlined from `roboui/static/tailwind.min.css` so air-gapped execution never attempts external CDN fetches.
- PII scrubbing must mask API keys, bearer tokens, emails, phone numbers, and IP addresses before code synthesis.
- Every commit must be atomic, structured, and verified with passing pytest test cases.

## Review Focus

- Unclosed markdown code fences (` ```html ` without terminating ` ``` ` due to context truncation).
- High-entropy API key patterns and JWTs appearing in user sketches or intermediate text.
- Malicious `<script>` tags attempting network exfiltration inside the generated iframe.
- Memory/VRAM allocation failures gracefully falling back to CPU or mock mode with structured logging.
- Missing model weights prompting clear user instructions rather than uncaught exceptions.

---

### Task 1: Project Scaffolding, Dependencies, and Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `roboui/__init__.py`
- Create: `roboui/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `RoboUIConfig` dataclass managing model paths, context window limits, VRAM offload layers, mock mode toggle, and sandbox policies.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with ModuleNotFoundError: No module named 'roboui'

- [ ] **Step 3: Implement `roboui/config.py` and dependency manifests**

Define `RoboUIConfig` dataclass with `from_env()` method loading environment variables via `os.environ` and `python-dotenv`.
Define `requirements.txt`:
```
streamlit>=1.35.0
langchain-core>=0.2.0
pillow>=10.0.0
pytest>=8.0.0
python-dotenv>=1.0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt .env.example .gitignore roboui/ tests/test_config.py
git commit -m "feat: configure project dependencies and runtime settings"
```

---

### Task 2: PII Redaction & Regex Filtering

**Files:**
- Create: `roboui/sanitizer.py`
- Test: `tests/test_sanitizer.py`

**Interfaces:**
- Consumes: Raw text strings from intermediate vision description or user prompts.
- Produces: `sanitize_pii(text: str) -> tuple[str, list[str]]` returning sanitized text and deduplicated list of detected PII types.

**Taxonomy Mapping Table:**
- `EMAIL` -> `[REDACTED_EMAIL]`
- `SECRET_KEY` -> `[REDACTED_SECRET]` (matches `sk-[A-Za-z0-9_-]{20,}`, `ghp_[A-Za-z0-9]{36}`, `AKIA[0-9A-Z]{16}`, generic high-entropy API tokens)
- `PHONE` -> `[REDACTED_PHONE]` (matches standard US and international formats: `+1-XXX-XXX-XXXX`, `(XXX) XXX-XXXX`, etc.)
- `IP_ADDRESS` -> `[REDACTED_IP]` (matches standard IPv4 dotted quads)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sanitizer.py
from roboui.sanitizer import sanitize_pii

def test_sanitize_emails():
    text = "Contact me at alice.smith@company.org or support@domain.io for help."
    clean, matches = sanitize_pii(text)
    assert "[REDACTED_EMAIL]" in clean
    assert "alice.smith@company.org" not in clean
    assert "support@domain.io" not in clean
    assert "EMAIL" in matches

def test_sanitize_api_keys():
    text = "Use key sk-live-abcdef1234567890abcdef1234567890 or ghp_abcdefghijklmnopqrstuvwxyz1234567890."
    clean, matches = sanitize_pii(text)
    assert "[REDACTED_SECRET]" in clean
    assert "sk-live-" not in clean
    assert "ghp_" not in clean
    assert "SECRET_KEY" in matches

def test_sanitize_phone_and_ip():
    text = "Server at 192.168.1.100 call (555) 123-4567"
    clean, matches = sanitize_pii(text)
    assert "[REDACTED_IP]" in clean
    assert "[REDACTED_PHONE]" in clean
    assert "192.168.1.100" not in clean
    assert "(555) 123-4567" not in clean
    assert set(matches) == {"IP_ADDRESS", "PHONE"}

def test_sanitize_clean_text():
    text = "Simple navbar with Home, About, and Contact links."
    clean, matches = sanitize_pii(text)
    assert clean == text
    assert matches == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sanitizer.py -v`
Expected: FAIL with ModuleNotFoundError or ImportError

- [ ] **Step 3: Implement `sanitize_pii` in `roboui/sanitizer.py`**

Compile regex patterns using standard library `re`. Loop through patterns, replace occurrences, and record deduplicated category names in order of discovery.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sanitizer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add roboui/sanitizer.py tests/test_sanitizer.py
git commit -m "feat: implement deterministic PII scrubbing regex engine"
```

---

### Task 3: Robust Code Extraction & Air-Gapped Sandboxing

**Files:**
- Create: `roboui/extractor.py`
- Create: `roboui/static/tailwind.min.css`
- Test: `tests/test_extractor.py`

**Interfaces:**
- Consumes: Raw LLM output strings (potentially with markdown code blocks or truncated tags).
- Produces: `extract_html(llm_output: str) -> str` and `wrap_in_airgap_sandbox(html_content: str, inline_css: str | None = None) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extractor.py
from roboui.extractor import extract_html, wrap_in_airgap_sandbox

def test_extract_markdown_fenced_html():
    raw = "Here is your site:\n```html\n<!DOCTYPE html><html><body><h1>Hello</h1></body></html>\n```\nEnjoy!"
    extracted = extract_html(raw)
    assert extracted.startswith("<!DOCTYPE html>")
    assert "```" not in extracted
    assert "Here is your site" not in extracted

def test_extract_unclosed_fence():
    raw = "```html\n<div class='container'><p>Truncated UI</p>"
    extracted = extract_html(raw)
    assert "<div class='container'><p>Truncated UI</p>" in extracted
    assert "```" not in extracted

def test_extract_raw_html_fallback():
    raw = "I designed this: <div><span>Direct tag</span></div>."
    extracted = extract_html(raw)
    assert "<div><span>Direct tag</span></div>" in extracted

def test_airgap_sandbox_wrapper():
    html = "<div class='bg-blue-500'>Content</div>"
    wrapped = wrap_in_airgap_sandbox(html, inline_css=".bg-blue-500 { color: blue; }")
    assert "<meta http-equiv=\"Content-Security-Policy\"" in wrapped
    assert "default-src 'none';" in wrapped
    assert "style-src 'unsafe-inline';" in wrapped
    assert "script-src 'unsafe-inline';" in wrapped
    assert "<style>.bg-blue-500 { color: blue; }</style>" in wrapped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_extractor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `roboui/extractor.py` and `roboui/static/tailwind.min.css`**

Implement multi-strategy extractor:
1. Regex matching complete ` ```(?:html)?\s*(.*?)\s*``` `.
2. Regex matching unclosed prefix ` ```(?:html)?\s*(.*) ` for truncated completions.
3. Fallback searching for first `<` and last `>` with HTML tags.
4. Auto-closing unbalanced common container tags (`<div>`, `<section>`, `<body>`, `<html>`) if truncated.
5. Sandboxing wrapper injecting strict offline Content Security Policy (CSP) blocking external requests (`default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'self' data:;`) and inlining local CSS.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add roboui/extractor.py roboui/static/tailwind.min.css tests/test_extractor.py
git commit -m "feat: implement robust code extraction and CSP airgap wrapper"
```

---

### Task 4: LangChain Prompt Templates, Chaining, & Context Throttling

**Files:**
- Create: `roboui/chains.py`
- Test: `tests/test_chains.py`

**Interfaces:**
- Consumes: Vision sketch descriptions and user style preferences.
- Produces: `get_vision_prompt_template()`, `get_codegen_prompt_template()`, `throttle_context(text: str, max_tokens: int) -> str`, and `build_roboui_pipeline(vision_llm, codegen_llm)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chains.py
from langchain_core.language_models.fake import FakeListLLM
from roboui.chains import (
    get_vision_prompt_template,
    get_codegen_prompt_template,
    throttle_context,
    build_roboui_pipeline,
)

def test_prompt_templates():
    vision_tmpl = get_vision_prompt_template()
    codegen_tmpl = get_codegen_prompt_template()
    assert vision_tmpl.input_variables == ["image_description"]
    assert set(codegen_tmpl.input_variables) == {"semantic_description", "style_notes"}

def test_throttle_context():
    long_text = "word " * 1000
    throttled = throttle_context(long_text, max_tokens=100)
    assert len(throttled.split()) <= 120

def test_build_roboui_pipeline():
    vision_mock = FakeListLLM(responses=["Navbar with logo, Hero header with Title and CTA button."])
    codegen_mock = FakeListLLM(responses=["```html\n<nav>Logo</nav><header><h1>Title</h1><button>CTA</button></header>\n```"])
    pipeline = build_roboui_pipeline(vision_mock, codegen_mock)
    output = pipeline.invoke({
        "image_description": "Hand-drawn wireframe with box on top and button below",
        "style_notes": "Tailwind dark theme",
    })
    assert "semantic_description" in output
    assert "raw_code" in output
    assert "sanitized_description" in output
    assert "<nav>Logo</nav>" in output["raw_code"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chains.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `roboui/chains.py`**

Define prompt templates using LangChain's `PromptTemplate`. Implement `throttle_context` using token count estimation (~4 characters per token). Implement `build_roboui_pipeline` composing vision prompt, vision model, `StrOutputParser()`, `sanitize_pii`, codegen prompt, codegen model, and `StrOutputParser()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chains.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add roboui/chains.py tests/test_chains.py
git commit -m "feat: implement LangChain prompt chaining and context throttling"
```

---

### Task 5: Dual Inference Engine (Quantized Local llama.cpp + Deterministic Mock)

**Files:**
- Create: `roboui/engine.py`
- Test: `tests/test_engine.py`

**Interfaces:**
- Consumes: `RoboUIConfig`, image input, and prompt text.
- Produces: `GenerationResult` dataclass, `BaseInferenceEngine`, `MockInferenceEngine`, and `LlamaCppInferenceEngine` with `get_engine(config) -> BaseInferenceEngine`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_engine.py
import pytest
from roboui.config import RoboUIConfig
from roboui.engine import get_engine, GenerationResult, ModelFileNotFoundError, LlamaCppInferenceEngine

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

def test_vram_exhaustion_fallback(monkeypatch):
    config = RoboUIConfig(mock_mode=False, model_path="tests/fixtures/dummy.gguf", n_gpu_layers=33)
    # Simulate VRAM failure when attempting GPU offload
    engine = LlamaCppInferenceEngine.__new__(LlamaCppInferenceEngine)
    engine.config = config
    fallback_occurred = engine._handle_vram_failure(RuntimeError("CUDA out of memory"))
    assert fallback_occurred is True
    assert engine.config.n_gpu_layers == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `roboui/engine.py`**

Create:
- `GenerationResult` dataclass (`html_code: str`, `semantic_description: str`, `redacted_pii: list[str]`, `tokens_used: int`, `latency_ms: float`, with `@property def sanitized_pii_count(self) -> int: return len(self.redacted_pii)`).
- Custom exception `ModelFileNotFoundError`.
- `MockInferenceEngine`: generates realistic Tailwind dashboard/landing-page HTML based on sketch description, executing the PII scrubbing, code extraction, and CSP wrapping pipeline.
- `LlamaCppInferenceEngine`: lazy-loads `llama_cpp.Llama` with 4-bit quantization, `n_gpu_layers`, `n_ctx`, and structured exception handling for missing files or VRAM exhaustion.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add roboui/engine.py tests/test_engine.py
git commit -m "feat: implement dual inference engines with mock and quantized llama.cpp"
```

---

### Task 6: Streamlit Application & Air-Gapped Preview

**Files:**
- Create: `roboui/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Provides: Web interface with file uploader (PNG, JPG), mock/local engine selector, intermediate representation inspector, live sandboxed `st.components.v1.html` preview, and HTML download button.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py
from roboui.app import build_iframe_html, validate_uploaded_image

def test_build_iframe_html():
    raw_html = "<button class='bg-blue-500'>Click</button>"
    iframe_code = build_iframe_html(raw_html)
    assert "<button" in iframe_code
    assert "Content-Security-Policy" in iframe_code
    assert "default-src 'none'" in iframe_code

def test_validate_image_valid():
    valid, err = validate_uploaded_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
    assert valid is True
    assert err is None

def test_validate_image_invalid():
    valid, err = validate_uploaded_image(b"not an image file")
    assert valid is False
    assert "Invalid image format" in err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `roboui/app.py`**

Implement Streamlit layout with:
- Session state initialization: `st.session_state.generation_result = None`, `st.session_state.uploaded_image_bytes = None`.
- Sidebar controls: Engine selection (Mock vs Local GGUF), Context window slider, VRAM layers slider.
- Main panel: Sketch upload, sketch display, Generate button, Tabs for "Live Preview", "Generated Code", and "Security & PII Audit".
- Safe sandboxed HTML embedding via `st.components.v1.html(build_iframe_html(...), height=600, scrolling=True)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add roboui/app.py tests/test_app.py
git commit -m "feat: scaffold Streamlit web application with sandboxed preview"
```

---

### Task 7: End-to-End Pipeline Integration Test

**Files:**
- Create: `tests/test_e2e.py`
- Test: `tests/test_e2e.py`

**Interfaces:**
- Verifies: Full end-to-end execution of image input -> vision -> intermediate PII scrubbing -> codegen -> extraction -> sandbox wrapping -> HTML validation.

- [ ] **Step 1: Write E2E integration test**

```python
# tests/test_e2e.py
from roboui.config import RoboUIConfig
from roboui.engine import get_engine
from roboui.app import build_iframe_html

def test_e2e_pipeline_flow():
    # 1. Initialize configuration in Mock mode
    config = RoboUIConfig(mock_mode=True, max_context_tokens=2048)
    engine = get_engine(config)
    
    # 2. Simulate user sketch with embedded PII in notes
    sketch_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    notes = "Build a sign-up card for dev@company.com with token sk-live-123456789012345678901234567890"
    
    # 3. Execute generation pipeline
    result = engine.generate_ui(sketch_bytes, custom_notes=notes)
    
    # 4. Verify PII scrubbing occurred
    assert "dev@company.com" not in result.semantic_description
    assert "sk-live-" not in result.semantic_description
    assert "EMAIL" in result.redacted_pii
    assert "SECRET_KEY" in result.redacted_pii
    assert result.sanitized_pii_count == 2
    
    # 5. Verify HTML structure & sandbox wrapper
    assert "<!DOCTYPE html>" in result.html_code
    iframe_view = build_iframe_html(result.html_code)
    assert "Content-Security-Policy" in iframe_view
    assert "default-src 'none'" in iframe_view
```

- [ ] **Step 2: Run all tests with pytest**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: add end-to-end integration test verifying full pipeline"
```

---

### Task 8: Production Documentation, Architecture Diagram, & Antigravity Artifact

**Files:**
- Modify: `README.md`
- Create: `C:\Users\Austi\.gemini\antigravity\brain\aa0d7316-d673-4828-a00f-5e4a60402ead\roboui-verification-report.md`

**Deliverables:**
1. Production-grade `README.md` featuring:
   - System overview & LangChain Mermaid data flow diagram
   - Local model setup guide (GGUF 4-bit Gemma/llama.cpp)
   - Architectural trade-offs: On-Device vs Cloud API
   - Security model & air-gapped sandbox verification
2. Antigravity Artifact featuring:
   - Clean terminal output summary showing build and 100% pytest pass
   - Technical overview of context throttling and token economics
   - 3 high-impact, ready-to-use resume bullet points
3. Quality Gate:
   - Run two-axis `code-review` (Standards and Spec)
   - Run `slopmonster` (`deslop.py`) to eliminate AI buzzwords and ensure a clean 5/5 score
