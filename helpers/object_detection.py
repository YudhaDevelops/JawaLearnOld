# https://www.kaggle.com/code/marquis03/mobilenet-v3-70-dog-breeds-image-classification
# https://www.kaggle.com/code/umairshahpirzada/birds-525-species-image-classification-mobilenet
# https://www.kaggle.com/code/rohandeysarkar/ultimate-image-classification-guide-2020/notebook


# Import packages
import threading
from typing import Union

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import os
import argparse
import sys
import time
from threading import Thread
import importlib.util

from streamlit_webrtc import (
    VideoProcessorBase,
    VideoTransformerBase,
    WebRtcMode,
    webrtc_streamer,
    RTCConfiguration,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

PATH_TO_MODEL = "./models/mobilenetV2.tflite"
PATH_TO_LABELS = "./models/labels_legena.txt"
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

def detect_capture(image):
    interpreter = load_tf_lite_model()
    labels = load_labels()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    float_input = input_details[0]["dtype"] == np.float32

    image_resized = cv2.resize(image, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    input_mean = 127.5
    input_std = 127.5

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
    st.write("Kelas : {} \n"
          "Score : {} \n"
          "Boxes : {}".format(classes,scores,boxes))

    imH, imW, _ = image.shape

    for i in range(len(scores)):
        if(scores[i] > 0.05) and (scores[i] <= 1.0):
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

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
                image,
                (xmin, label_ymin - labelSize[1] - 10),
                (xmin + labelSize[0], label_ymin + baseLine - 10),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                image,
                label,
                (xmin, label_ymin - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
    print("Kelas : {} \n"
          "Score : {} \n"
          "Boxes : {}".format(classes,scores,boxes))
    return image

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
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]["shape"][1]
        width = input_details[0]["shape"][2]
        float_input = input_details[0]["dtype"] == np.float32

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
            if (scores[i] > 0.05) and (scores[i] <= 1.0):
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

        print("Shape : {}".format(out_image.shape))
        return out_image


def realtime_video_detection():
    info = st.empty()
    info.markdown("First, click on :blue['START'] to use webcam")
    ctx = webrtc_streamer(
        key="object_detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx.video_transformer:
        info.markdown("Click on :blue['SNAPSHOT'] to take a picture")
        snap = st.button("SNAPSHOT")
        if snap:
            if ctx.video_transformer.out_image is not None:
                with ctx.video_transformer.frame_lock:
                    out_image = ctx.video_transformer.out_image.copy()

                st.write("Sebelum:")
                st.image(out_image, channels="BGR")
                image = detect_capture(out_image)
                st.write("Sesudah:")
                st.image(image, channels="BGR")



if __name__ == "__main__":
    realtime_video_detection()
