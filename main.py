import pandas as pd
import streamlit as st
import cv2
import numpy as np
from backend import process_frame, detected_data

st.markdown(
    "<h2 style='text-align: center;'>🚗🇮🇳 Indian Number Plate Detection 🚗</h2>",
    unsafe_allow_html=True
)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
stframe = st.empty()  # Placeholder for video streaming

if st.button("Start Detection", key="start"):
    stop = st.button("Stop Detection", key="stop")  # Ensure unique key

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            break

        processed_frame, _ = process_frame(frame, detected_data)
        img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # Update video stream
        stframe.image(img_rgb, caption="Live Detection", use_column_width=True)

        # Stop streaming when the "Stop Detection" button is clicked
        if stop:
            break

cap.release()
cv2.destroyAllWindows()

if st.button("Show Detected Plates", key="show"):
    df = pd.DataFrame(detected_data)
    st.dataframe(df)
