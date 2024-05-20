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

import threading
from typing import Union

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import (
    VideoProcessorBase,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
)


PATH_TO_MODEL = "../models/detectObject/model.tflite"
PATH_TO_LABELS = "../models/detectObject/labels.txt"


@st.cache_resource
def load_tf_lite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=PATH_TO_MODEL)
        interpreter.allocate_tensors()

        return interpreter
    except ValueError as ve:
        print("Error loading the TensorFlow Lite model:", ve)
        exit()


@st.cache_resource
def load_labels():
    with open(PATH_TO_LABELS, "r") as f:
        labels = [line.strip() for line in f.readlines()]

        return labels


def get_model_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    float_input = input_details[0]["dtype"] == np.float32

    return input_details, output_details, height, width, float_input


class VideoTransformer(VideoTransformerBase):
    frame_lock: threading.Lock
    out_image: Union[np.ndarray, None]

    def __init__(self) -> None:
        self.frame_lock = threading.Lock()
        self.out_image = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        interpreter = load_tf_lite_model()

        labels = load_labels()

        input_mean = 127.5
        input_std = 127.5

        # get model details
        input_details, output_details, height, width, float_input = get_model_details(
            interpreter=interpreter
        )

        out_image = frame.to_ndarray(format="bgr24")

        imH, imW, _ = out_image.shape

        image_resized = cv2.resize(out_image, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[1]["index"])[0]
        classes = interpreter.get_tensor(output_details[3]["index"])[0]
        scores = interpreter.get_tensor(output_details[0]["index"])[0]

        for i in range(len(scores)):
            if (scores[i] > 0.5) and (scores[i] <= 1.0):
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(out_image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])]

                label = "%s: %d%%" % (
                    object_name,
                    int(scores[i] * 100),
                )

                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )

                label_ymin = max(ymin, labelSize[1] + 10)

                cv2.rectangle(
                    out_image,
                    (xmin, label_ymin - labelSize[1] - 10),
                    (xmin + labelSize[0], label_ymin + baseLine - 10),
                    (255, 255, 255),
                    cv2.FILLED,
                )

                cv2.putText(
                    out_image,
                    label,
                    (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

                with self.frame_lock:
                    self.out_image = out_image

        return out_image


def realtime_video_detection():
    info = st.empty()
    info.markdown("First, click on :blue['START'] to use webcam")
    ctx = webrtc_streamer(
        key="object detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": turn.get_ice_servers()},
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_transformer:
        info.markdown("Click on :blue['SNAPSHOT'] to take a picture")
        snap = st.button("SNAPSHOT")
        
realtime_video_detection()
