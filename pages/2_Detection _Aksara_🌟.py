import cv2
import streamlit as st

# Set page config ==================================================
st.set_page_config(
    page_title="Detection Aksara",
    page_icon="🌟",
)

# Set header page ===================================================
st.header("🌟 Detection Aksara :sparkles:", divider="rainbow")

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

# Ketika tombol "Start" diklik ===================================================
if start_button:
    # Mengganti status video_running menjadi True
    video_running = True
    
    # Menyembunyikan tombol "Start"
    start_button = False
    
    # Mengubah pilihan kamera menjadi indeks numerik
    if camera_option == "Camera 1":
        camera_option = 0
    elif camera_option == "Camera 2":
        camera_option = 1
    elif camera_option == "Camera 3":
        camera_option = 2
    elif camera_option == "Camera 4":
        camera_option = 3
        
    cap = cv2.VideoCapture(camera_option)

    if camera_option == None:
        time.sleep(.5)
        msg = st.toast("Anda belum memilih kamera")
    
    if not cap.isOpened():
        time.sleep(.5)
        msg = st.toast("Kamera tidak tersedia. Silakan pilih kamera lain atau periksa koneksi kamera Anda.")
    
    while cap.isOpened() and video_running:
        ret, frame = cap.read()

        if not ret:
            st.write('Video Capture Has Ended')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        frame_placeholder.image(frame, use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q') or not video_running:
            break

    cap.release()
    cv2.destroyAllWindows()

