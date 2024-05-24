import cv2
import streamlit as st
import time
import tensorflow as tf
import numpy as np

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, VideoProcessorFactory

# Set page config ==================================================
st.set_page_config(
    page_title="Scan AR Aksara",
    page_icon="🌟",
)

# Set css ===========================================================
with open("./assets/style.css") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Set header page ===================================================
st.header("🌟 Scan AR Aksara :sparkles:", divider="rainbow")

# Deteksi ===========================================================
from helper.objek_deteksi import realtime_video_detection

realtime_video_detection()