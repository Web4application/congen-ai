
import streamlit.testing.v1 as st_test

def test_gui_load():
    app = st_test.AppTest.from_file("gui.py")
    app.run()
    assert app.file_uploader[0].label.startswith("📂 Upload"), "File uploader missing"
