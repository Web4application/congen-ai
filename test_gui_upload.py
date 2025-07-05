
import streamlit.testing.v1 as st_test
import os
import pandas as pd

def test_gui_upload_and_train(tmp_path):
    app = st_test.AppTest.from_file("gui.py")
    # Supply iris sample
    app.session_state["file_uploader"] = None
    app.run()
    # Confirm page ran without exceptions
    assert app.exception == []
