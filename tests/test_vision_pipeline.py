import pytest
from langchain_core.language_models.fake import FakeListLLM
from roboui.chains import execute_langchain_vision_pipeline, OutOfMemoryInferenceError

def test_execute_langchain_vision_pipeline_success():
    vision_mock = FakeListLLM(responses=["Navigation bar with search box, hero card with submit button."])
    codegen_mock = FakeListLLM(responses=["```html\n<div class='p-4'><button>Submit</button></div>\n```"])
    
    result = execute_langchain_vision_pipeline(
        image_bytes=b"fake-image",
        custom_notes="Add email contact: admin@test.com",
        vision_llm=vision_mock,
        codegen_llm=codegen_mock,
        enforce_csp=True,
    )
    
    assert "clean_html" in result
    assert "<button>Submit</button>" in result["clean_html"]
    assert "```" not in result["clean_html"]
    assert "Content-Security-Policy" in result["sandboxed_html"]
    assert "admin@test.com" not in result["semantic_description"]
    assert "EMAIL" in result["redacted_pii"]

def test_execute_langchain_vision_pipeline_oom_boundary():
    class OOMThrowingLLM(FakeListLLM):
        def invoke(self, *args, **kwargs):
            raise MemoryError("CUDA out of memory during visual token projection")

    with pytest.raises(OutOfMemoryInferenceError) as exc_info:
        execute_langchain_vision_pipeline(
            image_bytes=b"large-image",
            custom_notes="Render complex layout",
            vision_llm=OOMThrowingLLM(responses=[]),
            codegen_llm=FakeListLLM(responses=[]),
        )
    assert "Memory exhausted during vision pipeline execution" in str(exc_info.value)
