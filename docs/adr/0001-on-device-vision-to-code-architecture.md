# 1. On-Device Vision-to-Code Pipeline Architecture

Transforming hand-drawn UI sketches into runnable web applications on consumer hardware requires balancing strict privacy, tight VRAM limits, and LLM output fragility. We decided to split the pipeline into an explicit two-stage LangChain orchestration (Vision Analysis -> Intermediate Semantic Representation -> Code Generation), insert deterministic regex PII scrubbing on the text representation rather than raw image pixels, isolate client-side rendering inside an air-gapped `iframe` sandbox (`sandbox="allow-scripts"`), and provide dual engine backends (Quantized GGUF llama.cpp and a deterministic zero-weight Mock Engine).

### Status
Accepted

### Considered Options
- **End-to-End Single Prompt (Image directly to HTML)**: Rejected due to context window exhaustion (visual tokens + code tokens exceeding 4K/8K consumer limits) and inability to inspect or scrub PII prior to code synthesis.
- **Client-side External CDN Loading**: Rejected as it breaks air-gapped security guarantees and leaks data during offline execution.
- **Mandatory Heavy Weight Downloads in CI/Dev**: Rejected because multi-gigabyte models prevent automated, reproducible unit test loops.

### Consequences
- PII is safely caught in the textual intermediate stream.
- The system can run completely offline without external network calls.
- Automated testing runs in sub-seconds with 100% deterministic coverage using the Mock Inference Engine.
- Real local models can be dropped in smoothly by supplying GGUF weights.
