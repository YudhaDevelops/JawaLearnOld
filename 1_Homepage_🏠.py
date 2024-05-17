import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="JawaLearn",
    page_icon="🏠",
)
st.header("🏠 Home :sparkles:", divider="rainbow")

col_deskripsi = st.columns(1)
col_fitur = st.columns(2)

with col_deskripsi[0]:
    "## Aplikasi JawaLearn"
    st.markdown('<div style="text-align: justify;">'
                'Merupakan aplikasi yang dibangun dengan menggunakan pembelajaran mesin yaitu CNN (Convolutional Neural Network), '
                'serta menggunakan data gambar sebagai alat belajar si mesin. Dengan melatih mesin yang berbeda '
                'diharapkan dapat membantu anda dalam memahami bentuk aksara Jawa. \n'
                'Aplikasi ini adalah hasil dari riset saya yang berjudul PENERAPAN TEKNOLOGI AUGMENTED REALITY MENGGUNAKAN CNN (CONVOLUTIONAL NEURAL NETWORK) DALAM PEMBELAJARAN AKSARA JAWA'
                '</div>', unsafe_allow_html=True)

    "## Tujuan Dibangun nya Aplikasi"
    tujuan_text = ("Aplikasi ini dibangun dengan tujuan : \n"
                   "1. Memudahkan anda untuk belajar aksara jawa \n"
                   "2. Membuat aplikasi Augmented Reality dengan menggabungkan teknologi AI yang dapat mempermudah belajar mengajar aksara Jawa. \n"
                   "3. Membuktikan bahwa teknologi Augmented Reality dapat digabungkan bersama dengan menggunakan algoritma Convolutional Neural Network (CNN). \n"
                   "4. Memberikan kontribusi dalam mengatasi kesulitan siswa dalam mempelajari aksara Jawa menggunakan teknologi Augmented Reality\n")
    st.markdown(tujuan_text, unsafe_allow_html=True)

    "## Fitur-Fitur Aplikasi JawaLearn\n"

with col_fitur[0]:
    "## Detection Aksara"
    st.markdown('<div style="text-align: justify;">'
                'Fitur ini dapat digunakan untuk melakukan translitrasi aksara Jawa LEGENA dengan deteksi '
                'secara realtime dengan menggunakan camera yang tersedia, dengan fitur ini memberikan arti '
                'pada aksara yang di letakan di depan kamera'
                '</div>', unsafe_allow_html=True)
    st.link_button("Coba Sekarang :sparkles:", "/Detection_Aksara_🌟")
    
with col_fitur[1]:
    "## Klasifikasi Aksara"
    st.markdown('<div style="text-align: justify;">'
                'fitur ini dapat digunakan untuk melakukan transliterasi aksara Jawa LEGENA dengan deteksi '
                'berdasarkan gambar tulisan aksara yang di masukkan baik daci kamera atau penyimpanan pada '
                'device anda</div>',unsafe_allow_html=True)
    st.link_button("Coba Sekarang :sparkles:", "/Klasifikasi_Aksara_🎴")


st.header("", divider="rainbow")
