import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_image(image):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image

st.set_page_config(page_title="MediaPipe Pose Estimation", layout="wide")
st.title("🤸 MediaPipe 姿態估計")

option = st.radio("選擇輸入方式", ['上傳圖片', '上傳影片'])

if option == '上傳圖片':
    uploaded_file = st.file_uploader("請選擇圖片", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = uploaded_file.read()
        np_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        col1, col2 = st.columns(2)
        col1.image(np_img, caption='原始圖片', use_column_width=True)
        processed = process_image(np_img.copy())
        col2.image(processed, caption='姿態估計結果', use_column_width=True)

elif option == '上傳影片':
    uploaded_video = st.file_uploader("請上傳影片", type=["mp4", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        with mp_pose.Pose() as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                stframe.image(frame, channels="BGR", use_column_width=True)
        cap.release()
        os.unlink(tfile.name)

