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


def test_refinement_pipeline_with_memory():
    from roboui.chains import build_refinement_pipeline, get_refinement_prompt_template

    refine_tmpl = get_refinement_prompt_template()
    assert set(refine_tmpl.input_variables) == {"existing_code", "refinement_instruction"}
    template_str = refine_tmpl.template
    assert "Analyze the relative widths, heights, alignment, and spatial positioning" in template_str
    assert "Tailwind CSS grid, flexbox, and relative fractional width classes" in template_str
    assert "1:1 design translation" in template_str

    codegen_mock = FakeListLLM(responses=["```html\n<nav class='bg-black'>Dark Logo</nav>\n```"])
    refine_pipeline = build_refinement_pipeline(codegen_llm=codegen_mock)
    output = refine_pipeline.invoke({
        "existing_code": "<nav class='bg-white'>Logo</nav>",
        "refinement_instruction": "Make the navigation bar dark mode",
    })
    assert "raw_code" in output
    assert "clean_html" in output
    assert "bg-black" in output["clean_html"]


