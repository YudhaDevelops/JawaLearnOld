import streamlit as st
import glob 
import os

st.set_page_config(
    page_title="About Me",
    page_icon="🙍",
)
with open("./assets/style.css") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


with open('./assets/README.md','r', encoding='utf-8') as f:
    readme_content = f.read()

st.markdown(readme_content, unsafe_allow_html=True)

