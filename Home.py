import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Wave project ! 👋")

st.sidebar.success("Select a version above.\n"
                   "Spectators -> direct video\n"
                   "Judges -> accurate scoring")