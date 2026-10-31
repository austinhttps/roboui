from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "roboui" / "app.py")

def test_app_ui_initialization():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception
    # Check that file uploader exists
    assert len(at.file_uploader) > 0
    # Check that generate button exists
    assert len(at.button) > 0

def test_app_ui_generation_flow():
    at = AppTest.from_file(APP_PATH)
    at.run()
    
    # Click the generate button
    at.button[0].click().run()
    assert not at.exception
    
    # Verify session state contains generation result
    assert at.session_state.get("generation_result") is not None
    result = at.session_state["generation_result"]
    assert "<!DOCTYPE html>" in result.html_code
    
    # Verify expandable area and text area / code representation exists
    assert len(at.expander) > 0
