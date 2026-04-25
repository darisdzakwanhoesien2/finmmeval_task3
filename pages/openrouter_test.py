import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from shared.openrouter_test_page import render_openrouter_test_page

st.set_page_config(
    page_title="OpenRouter Test — Task 3",
    page_icon="🌐",
    layout="wide",
)

render_openrouter_test_page(key_prefix="task3_or")