from __future__ import annotations

import streamlit as st

from utils.data import load_notes_sections, read_notes


st.set_page_config(page_title="Notes Review", page_icon=":memo:")
st.title("Notes Review")
st.caption("Research notes behind the Task 3 trading workflow and implementation")

sections = load_notes_sections()
notes = read_notes()

if not notes:
    st.warning("`notes.md` is missing.")
else:
    st.subheader("Section Navigator")
    for section in sections:
        with st.expander(section["title"], expanded=section["title"].lower().startswith("workflow")):
            st.markdown(section["body"])

    st.subheader("Full Notes")
    st.markdown(notes)
