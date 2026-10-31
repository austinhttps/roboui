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
