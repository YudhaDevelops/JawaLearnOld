import cv2
import streamlit as st
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import tensorflow as tf

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, VideoProcessorFactory

import helper.objek_deteksi as objek_deteksi

# Set page config ==================================================
st.set_page_config(
    page_title="Detection Aksara",
    page_icon="🌟",
)

# Set css ===========================================================
with open("./assets/style.css") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Set header page ===================================================
st.header("🌟 Detection Aksara :sparkles:", divider="rainbow")

# Fungsi ===========================================================
PATH_TO_MODEL = "./models/detectObject/model.tflite"
PATH_TO_LABELS = "./models/detectObject/labels.txt"
resW, resH = 640,480
imW, imH = int(resW), int(resH)
min_conf_threshold = 0.5

# realtime_video_detection()