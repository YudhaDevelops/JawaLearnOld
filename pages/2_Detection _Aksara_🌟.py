import cv2
import streamlit as st
import time

# Set page config ==================================================
st.set_page_config(
    page_title="Detection Aksara",
    page_icon="🌟",
)

# Set header page ===================================================
st.header("🌟 Detection Aksara :sparkles:", divider="rainbow")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

