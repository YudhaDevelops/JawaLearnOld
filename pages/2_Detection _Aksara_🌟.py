import streamlit as st

st.set_page_config(
    page_title="Detection Aksara",
    page_icon="🌟",
)

st.header("🌟 Detection Aksara :sparkles:", divider="rainbow")

import sys
from helpers.object_detection import realtime_video_detection
sys.path.append("helpers")
realtime_video_detection()