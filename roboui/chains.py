"""LangChain prompt templates, LCEL chaining, and context throttling for RoboUI."""

import logging
from typing import Any, Dict, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

try:
    from extractor import extract_html, wrap_in_airgap_sandbox
    from sanitizer import sanitize_pii
except ImportError:
    from roboui.extractor import extract_html, wrap_in_airgap_sandbox
    from roboui.sanitizer import sanitize_pii

logger = logging.getLogger("roboui.chains")


class OutOfMemoryInferenceError(RuntimeError):
    """Raised when VRAM or host memory is exhausted during visual token generation."""
    pass


VISION_PROMPT_TEXT = """You are an expert multimodal UI analyzer. Examine this sketch description:
{image_description}

Provide a structured, step-by-step description of all UI elements:
1. Navigation and headers
2. Main sections, cards, and layouts (Flexbox / Grid)
3. Form controls, buttons, and call-to-actions
4. Visual hierarchy and typography
"""

CODEGEN_PROMPT_TEXT = """You are an expert frontend engineer. Generate a clean, responsive single-page web application.

Strict Constraints:
- Use clean semantic HTML5.
- Style with standard modern utility classes (Tailwind CSS compatible).
- Write vanilla JavaScript for interactions.
- Strictly no external API calls, fonts, or tracking scripts.

Sanitized UI Structure:
{semantic_description}

User Style Preferences:
{style_notes}

Provide the complete webpage inside a ```html ... ``` code block.
"""

REFINEMENT_PROMPT_TEXT = """You are an expert frontend developer with a strict eye for 1:1 design translation.
Analyze the relative widths, heights, alignment, and spatial positioning of every element.
You must use Tailwind CSS grid, flexbox, and relative fractional width classes (e.g., `w-1/2`, `w-1/3`, `max-w-md`) to ensure perfect 1:1 design translation.
Aggressively apply modern Tailwind visual polish: rounded corners (`rounded-xl`), drop shadows (`shadow-sm`, `shadow-md`), subtle borders (`border-slate-200`), accessible text contrasts (`text-slate-800`), and smooth interactive states (`hover:bg-blue-600`, `hover:-translate-y-0.5`, `focus:ring-2`, `transition-all`).
Buttons require generous padding (`px-6 py-2.5`, `font-semibold`), text inputs need focus rings (`focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500`), and containers must utilize ample whitespace (`gap-4`, `p-6`).
Carefully apply the refinement instruction while preserving the overall layout, existing components, and Tailwind styling.

Existing Code Context:
```html
{existing_code}
```

Refinement Instruction:
{refinement_instruction}

Output strictly valid HTML, Tailwind CSS, and vanilla JavaScript inside a single ```html ... ``` code block. Never output conversational filler or explanations. Only output code.
"""


def get_vision_prompt_template() -> PromptTemplate:
    """Return PromptTemplate for vision layout analysis."""
    return PromptTemplate(
        template=VISION_PROMPT_TEXT,
        input_variables=["image_description"],
    )


def get_codegen_prompt_template() -> PromptTemplate:
    """Return PromptTemplate for code generation."""
    return PromptTemplate(
        template=CODEGEN_PROMPT_TEXT,
        input_variables=["semantic_description", "style_notes"],
    )


def get_refinement_prompt_template() -> PromptTemplate:
    """Return PromptTemplate for iterative code refinement with memory."""
    return PromptTemplate(
        template=REFINEMENT_PROMPT_TEXT,
        input_variables=["existing_code", "refinement_instruction"],
    )



def throttle_context(text: str, max_tokens: int) -> str:
    """Throttle context length to prevent VRAM memory exhaustion."""
    if not text:
        return ""

    # Approximate ~4 characters per token
    char_limit = max_tokens * 4
    if len(text) <= char_limit:
        return text

    words = text.split()
    trimmed_words = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > char_limit:
            break
        trimmed_words.append(word)
        current_len += len(word) + 1

    return " ".join(trimmed_words) + " ... [TRUNCATED]"


def build_roboui_pipeline(vision_llm: Any, codegen_llm: Any):
    """Build an end-to-end chained LangChain pipeline linking vision analysis and code generation."""
    vision_prompt = get_vision_prompt_template()
    codegen_prompt = get_codegen_prompt_template()
    str_parser = StrOutputParser()

    vision_chain = vision_prompt | vision_llm | str_parser
    codegen_chain = codegen_prompt | codegen_llm | str_parser

    def _execute_pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
        image_desc = inputs.get("image_description", "")
        style_notes = inputs.get("style_notes", "Default modern clean theme")

        # Stage 1: Vision analysis
        semantic_desc = vision_chain.invoke({"image_description": image_desc})
        combined_text = f"{semantic_desc} (Notes: {style_notes})"

        # Stage 2: Intermediate PII scrubbing
        sanitized_desc, redacted_pii = sanitize_pii(combined_text)

        # Stage 3: Code generation from sanitized representation
        raw_code = codegen_chain.invoke({
            "semantic_description": sanitized_desc,
            "style_notes": style_notes,
        })

        return {
            "semantic_description": semantic_desc,
            "sanitized_description": sanitized_desc,
            "redacted_pii": redacted_pii,
            "raw_code": raw_code,
        }

    return RunnableLambda(_execute_pipeline)


def build_refinement_pipeline(codegen_llm: Any):
    """Build LangChain pipeline for conversational iterative code refinement."""
    refinement_prompt = get_refinement_prompt_template()
    str_parser = StrOutputParser()
    chain = refinement_prompt | codegen_llm | str_parser

    def _execute_refinement(inputs: Dict[str, Any]) -> Dict[str, Any]:
        existing_code = inputs.get("existing_code", "")
        instruction = inputs.get("refinement_instruction", "")
        sanitized_instruction, redacted_pii = sanitize_pii(instruction)

        raw_output = chain.invoke({
            "existing_code": existing_code,
            "refinement_instruction": sanitized_instruction,
        })
        clean_html = extract_html(raw_output)
        return {
            "raw_code": raw_output,
            "clean_html": clean_html,
            "redacted_pii": redacted_pii,
            "instruction": sanitized_instruction,
        }

    return RunnableLambda(_execute_refinement)



def execute_langchain_vision_pipeline(
    image_bytes: Optional[bytes],
    custom_notes: str,
    vision_llm: Any,
    codegen_llm: Any,
    enforce_csp: bool = True,
    max_context_tokens: int = 2048,
) -> Dict[str, Any]:
    """Execute LangChain multimodal pipeline, extracting clean HTML/JS and enforcing OOM boundaries."""
    sketch_tag = f"UI sketch payload ({len(image_bytes)} bytes)" if image_bytes else "No image payload"
    throttled_input = throttle_context(f"{sketch_tag}. Notes: {custom_notes}", max_context_tokens)

    try:
        pipeline = build_roboui_pipeline(vision_llm, codegen_llm)
        pipeline_output = pipeline.invoke({
            "image_description": throttled_input,
            "style_notes": custom_notes or "Modern clean dark theme",
        })
        
        # Route raw LLM output through markdown-stripping regex
        raw_code = pipeline_output["raw_code"]
        clean_html = extract_html(raw_code)
        sandboxed_html = wrap_in_airgap_sandbox(clean_html) if enforce_csp else clean_html

        return {
            "semantic_description": pipeline_output["sanitized_description"],
            "redacted_pii": pipeline_output["redacted_pii"],
            "raw_code": raw_code,
            "clean_html": clean_html,
            "sandboxed_html": sandboxed_html,
        }
    except (MemoryError, RuntimeError) as exc:
        logger.error("OOM error in vision pipeline: %s", exc)
        raise OutOfMemoryInferenceError(f"Memory exhausted during vision pipeline execution: {exc}") from exc
