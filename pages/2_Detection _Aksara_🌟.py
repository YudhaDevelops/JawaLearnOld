import cv2
import streamlit as st
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# VideoTransformer class for processing the video frames
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

# Streamlit app
def main():
    st.header("Camera Access with Streamlit")

    # Display the video stream from the camera
    webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

if __name__ == "__main__":
    main()
