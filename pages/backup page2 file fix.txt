import cv2
import streamlit as st
import numpy as np
import tempfile
import time
import os
import argparse
import sys
import time
from threading import Thread
import importlib.util
import tensorflow as tf

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

# Set layout button =================================================
col_top = st.columns(2)
with col_top[0]:
    # Mendapatkan daftar kamera yang tersedia
    # Biasanya 0 adalah kamera default, kita asumsikan tersedia hingga 4 kamera untuk contoh ini
    available_cameras = ["Camera 1", "Camera 2", "Camera 3", "Camera 4"]
    camera_option = st.selectbox("Pilih nomor kamera:", 
                                 available_cameras,
                                 index=None,
                                 placeholder="Pilih nomor kamera")
with col_top[1]:
    # Membuat tombol "Start"
    start_button = st.button("Start",use_container_width=True)
    

# Variabel untuk menampilkan frame ================================================
frame_placeholder = st.empty()
# Variabel untuk menyimpan status apakah video sedang berjalan atau tidak ==========
video_running = False

# Fungsi untuk mengambil frame dari video ==========================================
def capture_video(camera_option):
    if camera_option == "Camera 1":
        camera_option = 0
    elif camera_option == "Camera 2":
        camera_option = 1
    elif camera_option == "Camera 3":
        camera_option = 2
    elif camera_option == "Camera 4":
        camera_option = 3
        
    cap = cv2.VideoCapture(camera_option)
    global video_running

    if camera_option == None:
        time.sleep(.5)
        msg = st.toast("Anda belum memilih kamera")
        return
    if not cap.isOpened():
        time.sleep(.5)
        msg = st.toast("Kamera tidak tersedia. Silakan pilih kamera lain atau periksa koneksi kamera Anda.")
        return
    
    # Menampilkan tombol "Stop"
    stop_button = st.button("Stop")
    
    st.markdown('<hr class="garis_sendiri"></hr>', unsafe_allow_html=True)
    prev_time = 0
    while cap.isOpened() and video_running:
        ret, frame = cap.read()

        if not ret:
            st.write('Video Capture Has Ended')
            break
        
        # load masalah model dan label
        interpreter = load_tf_lite_model()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        floating_model = input_details[0]['dtype'] == np.float32
        
        labels = load_labels()
        
        input_mean = 127.5
        input_std = 127.5
        # Check output layer name to determine if this model was created with TF2 or TF1,
        # because outputs are ordered differently for TF2 and TF1 models
        outname = output_details[0]['name']

        if ('StatefulPartitionedCall' in outname): # This is a TF2 model
            boxes_idx, classes_idx, scores_idx = 1, 3, 0
        else: # This is a TF1 model
            boxes_idx, classes_idx, scores_idx = 0, 1, 2
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        
        # Perform the actual detection
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()
        
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
        
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame_rgb, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                # print(label)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame_rgb, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame_rgb, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
        # Hitung frame rate
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame_rgb, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        frame_placeholder.image(frame_rgb, use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q') or not video_running:
            break

    cap.release()
    cv2.destroyAllWindows()

# Ketika tombol "Start" diklik ===================================================
if start_button:
    # Mengganti status video_running menjadi True
    video_running = True
    
    # Menyembunyikan tombol "Start"
    start_button = False
    
    # Memanggil fungsi capture_video
    capture_video(camera_option)

# Ketika tombol "Stop" diklik ====================================================
if st.session_state.get("stop_button", False):
    # Mengganti status video_running menjadi False
    video_running = False
