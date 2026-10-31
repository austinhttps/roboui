# RoboUI Domain Model

RoboUI is an on-device multimodal Python application that converts hand-drawn UI sketches into production-grade HTML/CSS/JS applications using local quantized models, LangChain orchestration, and a privacy-preserving sandbox.

## Language

**UI Sketch**:
A hand-drawn or wireframe visual representation of a web interface uploaded as an image.
_Avoid_: screenshot, mock, design spec

**Visual Token Budget**:
The portion of the context window allocated to multimodal projector tokens during sketch encoding.
_Avoid_: image buffer, vision context

**Intermediate Semantic Representation**:
The structured textual analysis of UI layout, components, and hierarchy emitted by the vision analysis step before code generation.
_Avoid_: draft, design summary, intermediate prompt

**PII Scrubbing**:
The deterministic regex-based masking of credentials, tokens, emails, and sensitive identifiers from intermediate text streams prior to code generation.
_Avoid_: anonymization, censorship, data cleaning

**Air-Gapped Sandbox**:
A browser execution boundary for generated code enforcing `iframe` attribute isolation (`sandbox="allow-scripts"`) without network exfiltration or DOM access to the host app.
_Avoid_: container, virtual machine, jail

**Code Extraction Pipeline**:
The multi-strategy parsing sequence that strips markdown wrappers, repairs truncated tags, and extracts standalone HTML/CSS/JS.
_Avoid_: regex stripper, code cleaner

**Quantized Local Engine**:
The local 4-bit GGUF inference runtime powered by `llama.cpp` for on-device execution without cloud dependencies.
_Avoid_: local LLM, offline model

**Mock Inference Engine**:
The deterministic inference simulator used for unit testing and offline verification without requiring multi-gigabyte weight downloads.
_Avoid_: dummy model, test stub
