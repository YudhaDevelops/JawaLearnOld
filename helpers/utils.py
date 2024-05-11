import base64

import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

def set_background(image_files):
    with open(image_files,"rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image:url(dara:image/png;base64,{b64_encoded});
            background-size:cover;
            }}
        </style>
    """
    st.markdown(style,unsafe_allow_html=True)

