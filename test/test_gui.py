# test_gui.py
import streamlit.testing.v1 as st_test

def test_gui_behavior():
    app = st_test.AppTest.from_file("gui.py")
    app.run()

    # Test file uploader exists
    assert app.file_uploader[0].label == "📂 Upload CSV"

    # Test that no errors appear before uploading
    assert app.exception == []

    # Test that page title renders
    assert "Congen‑AI" in app.markdown[0].value
