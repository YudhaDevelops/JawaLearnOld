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
import helper.turn as turn


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
from streamlit_webrtc import webrtc_streamer
import av


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    flipped = img[::-1,:,:]

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
