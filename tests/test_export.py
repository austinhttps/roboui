import io
import zipfile
import pytest
from roboui.exporter import ZipExportService

def test_zip_export_service_creates_valid_archive():
    sample_html = "<!DOCTYPE html><html><body><h1>Hello RoboUI</h1></body></html>"
    zip_bytes = ZipExportService.export_to_zip(sample_html)
    
    assert isinstance(zip_bytes, bytes)
    assert len(zip_bytes) > 0
    
    # Read back the zip file in-memory
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        namelist = zf.namelist()
        assert "index.html" in namelist
        extracted_content = zf.read("index.html").decode("utf-8")
        assert extracted_content == sample_html

def test_zip_export_service_preserves_code_without_markdown_corruption():
    code_with_markdown_edge_cases = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><title>App</title></head>\n"
        "<body>\n"
        "  <div class='markdown-view'>`inline_code` and ```triple_ticks```</div>\n"
        "</body>\n"
        "</html>"
    )
    zip_bytes = ZipExportService.export_to_zip(code_with_markdown_edge_cases, zip_filename="project.zip")
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        extracted = zf.read("index.html").decode("utf-8")
        assert extracted == code_with_markdown_edge_cases
        assert "```triple_ticks```" in extracted
