"""LangChain prompt templates, LCEL chaining, and context throttling for RoboUI."""

from typing import Any, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from roboui.sanitizer import sanitize_pii


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

        # Stage 2: Intermediate PII scrubbing
        sanitized_desc, redacted_pii = sanitize_pii(semantic_desc)

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
