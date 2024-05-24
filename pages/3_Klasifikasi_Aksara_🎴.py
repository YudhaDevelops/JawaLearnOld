import os
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

# declare models
DenseNet121 = False
EfficientNet = False
Inception = False
MobileNetV2 = False
ResNet50 = False
VGG16 = False
Xception = False

st.set_page_config(page_title="Predict Aksara", page_icon="🎴")

with open("./assets/style.css") as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def classify(image, model, class_names):
    try:
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        prediction = model.predict(data)
        return prediction
    except Exception as e:
        st.error(f"Error in classify function: {e}")
        return None

with open("./models/label_best_model2.txt", "r") as f:
    class_name = [a.strip() for a in f.readlines()]

def load_my_model(name):
    path_model = "./models/modelA"
    model_files = [file for file in os.listdir(path_model) if file.endswith(".h5")]
    st.write(f"Model files found: {model_files}")
    for file in model_files:
        if name in file:
            model_path = os.path.join(path_model, file)
            st.write(f"Loading model from {model_path}")
            try:
                model_keras = load_model(model_path)
                return model_keras
            except Exception as e:
                st.error(f"Error loading model {model_path}: {e}")
                return None
    st.error(f"Model {name} not found.")
    return None

st.header("🎴 Predict Aksara :sparkles:")
uploaded_file = st.file_uploader("Choose a file", type=["jpeg", "png", "jpg"])

st.markdown('<hr class="garis_sendiri"></hr>', unsafe_allow_html=True)

def my_predict(uploaded_file, models):
    predict_arr = []
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        if len(models) > 0:
            for idx, model in enumerate(models):
                model_keras = load_my_model(model)
                if model_keras is None:
                    continue
                try:
                    prediction = classify(image, model_keras, class_name)
                    if prediction is None:
                        continue
                    index = np.argmax(prediction)
                    predict_class = class_name[index]
                    confidence_score = prediction[0][index]
                    nama_model = "Model : " + model.capitalize()
                    skor_dua_angka = "{:.2f}".format(confidence_score)
                    predict_arr.append([nama_model, predict_class, skor_dua_angka])
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    continue
    return predict_arr

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col_tampil = st.columns(2)
    with col_tampil[0]:
        st.image(image, width=300)
    with col_tampil[1]:
        pilih_model = st.multiselect('Pilih Model', 
                                     options=['DenseNet121', 'EfficientNet', 'Inception', 'MobileNetV2', 'ResNet50', 'VGG16', 'Xception'],
                                     placeholder="Pilih model yang mau anda pakai"
                                     )
    st.markdown('<hr class="garis_sendiri"></hr>', unsafe_allow_html=True)
    pilih_model_lower = [model.lower() for model in pilih_model]
    arr_pred = my_predict(uploaded_file, pilih_model_lower)
    if len(arr_pred) > 0:
        st.markdown('<h3 class="hasil_predict">Hasil Prediksi</h3>', unsafe_allow_html=True)
        for i in range(len(arr_pred)):
            st.write(arr_pred[i][0], "| Memprediksi : ", arr_pred[i][1], "| Skor : ", arr_pred[i][2])
        st.markdown('<hr class="garis_sendiri"></hr>', unsafe_allow_html=True)
    else:
        st.write("No predictions were made.")
