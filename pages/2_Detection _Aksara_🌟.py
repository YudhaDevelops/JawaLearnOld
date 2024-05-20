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

import streamlit as st
import cv2
from tempfile import NamedTemporaryFile

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, VideoProcessorFactory

import helper.turn as turn

def showDetectFrame(st_frame, image):

    # Tampilkan frame video tanpa prediksi
    st_frame.image(image,
                   caption='Detected Video',
                   channels="BGR",
                   )

def play_webcam():
    
    source_webcam = 0
    
    if st.button('Deteksi Secara Langsung'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            stop_button = st.button('Berhenti')
            while (vid_cap.isOpened() and not stop_button):
                success, image = vid_cap.read()
                if success:
                    showDetectFrame(st_frame, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.error("Ada Kesalahan Saat Proses Deteksi: " + str(e))
import av

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        pass

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return img

def live():
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={
            "iceServers": turn.get_ice_servers(),
            "iceTransportPolicy": "relay",
        },
        video_transformer_factory=lambda: VideoTransformer(),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        video_processor_factory=lambda: VideoProcessorFactory(fps=60)
    )
