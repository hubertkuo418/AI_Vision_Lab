import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.model_registry import run_model
from src.dnn_face_detection import detect_faces_dnn


# ======================
# UI Configuration
# ======================
st.set_page_config(page_title="AI Vision Toolkit", layout="wide")

st.title("AI Vision Toolkit")


# ======================
# Sidebar Navigation
# ======================
# Sidebar allows users to switch between different functional modules
mode = st.sidebar.radio(
    "Choose Mode",
    ["Image Processing", "AI Vision", "Webcam"]
)


# ======================
# Image Processing Mode
# ======================
if mode == "Image Processing":

    # Upload image from local filesystem
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        # Read image and convert it into NumPy array format
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Display original image
        st.image(image, caption="Original Image", use_container_width=True)

        # Select classical computer vision operation
        option = st.selectbox(
            "Choose Operation",
            [
                "Gray",
                "Gaussian Blur",
                "Canny Edge",
                "Histogram Equalization",
                "Dilate",
                "Erode"
            ]
        )

        params = {}

        # Hyperparameter tuning for Gaussian Blur
        if option == "Gaussian Blur":
            params["ksize"] = st.slider(
                "Kernel Size",
                1, 21, 9,
                step=2
            )

        # Hyperparameter tuning for Canny Edge Detection
        elif option == "Canny Edge":
            params["low"] = st.slider(
                "Low Threshold",
                50, 150, 100
            )
            params["high"] = st.slider(
                "High Threshold",
                150, 300, 200
            )

        # Execute selected image processing pipeline
        result = run_model(option, image, params)

        # Display processed output image
        st.image(result, caption="Result", use_container_width=True)


# ======================
# AI Vision Mode
# ======================
elif mode == "AI Vision":

    # Upload image for AI-based inference
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:

        # Load image and convert to NumPy array
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Show input image
        st.image(image, caption="Original Image", use_container_width=True)

        # Select AI-based model
        option = st.selectbox(
            "Choose AI Model",
            [
                "Face Detection (Haar)",
                "Face Detection (DNN)",
                "Object Detection (YOLO)"
            ]
        )

        params = {}

        # Confidence threshold control for DNN-based face detection
        if option == "Face Detection (DNN)":
            params["conf_threshold"] = st.slider(
                "Confidence Threshold",
                0.1, 0.9, 0.5,
                step=0.05
            )

        # Run selected AI model pipeline
        result = run_model(option, image, params)

        # Display inference result
        st.image(result, caption="Result", use_container_width=True)


# ======================
# Webcam Mode
# ======================
elif mode == "Webcam":

    # Instruction for user interaction
    st.warning("Click Start to enable webcam (press ESC to stop)")

    # Toggle for starting webcam stream
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])

    # Initialize webcam capture (device index 0)
    cap = cv2.VideoCapture(0)

    while run:

        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not available")
            break

        # Mirror effect for more intuitive user experience
        frame = cv2.flip(frame, 1)

        # Apply real-time DNN face detection
        frame = detect_faces_dnn(frame, conf_threshold=0.5)

        # Convert BGR (OpenCV) to RGB (Streamlit display requirement)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update Streamlit image container
        FRAME_WINDOW.image(frame)

    # Release webcam resource when loop ends
    cap.release()