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

def test_app_ui_missing_model_error_surfaced():
    at = AppTest.from_file(APP_PATH)
    at.run()
    # Explicitly set model path to a non-existent path
    at.sidebar.text_input[0].set_value("nonexistent_model.gguf").run()
    # Click generate with non-existent local weights
    at.button[0].click().run()
    assert not at.exception
    # Assert that st.error surfaced the missing model error
    assert len(at.error) > 0
    assert "Local GGUF Model File Not Found" in at.error[0].value

def test_app_ui_mock_toggle_generation_flow():
    at = AppTest.from_file(APP_PATH)
    at.run()
    
    # Toggle Mock Mode on in sidebar
    at.sidebar.toggle[0].set_value(True).run()
    assert at.sidebar.toggle[0].value is True
    
    # Click the generate button
    at.button[0].click().run()
    assert not at.exception
    
    # Verify session state contains generation result
    assert at.session_state.get("generation_result") is not None
    result = at.session_state["generation_result"]
    assert "<!DOCTYPE html>" in result.html_code
    
    # Verify expandable area exists
    assert len(at.expander) > 0

    # Verify download buttons exist (standalone HTML + zip workspace)
    assert len(at.download_button) >= 2
    zip_btn = next((b for b in at.download_button if "Workspace" in b.label), None)
    assert zip_btn is not None

