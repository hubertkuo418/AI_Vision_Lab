from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.dnn_face_detection import detect_faces_dnn
from src.model_registry import run_model


st.set_page_config(page_title="AI Vision Workbench", layout="wide")


PROCESSING_MODES = {
    "Image Processing": [
        "Gray",
        "Gaussian Blur",
        "Canny Edge",
        "Histogram Equalization",
        "Dilate",
        "Erode",
    ],
    "AI Vision": [
        "Face Detection (Haar)",
        "Face Detection (DNN)",
        "Object Detection (YOLO)",
    ],
}


def init_state():
    defaults = {
        "history": [],
        "current_result": None,
        "current_original": None,
        "current_summary": None,
        "selected_history": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def image_to_array(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def build_params(operation):
    params = {}

    if operation == "Gaussian Blur":
        params["ksize"] = st.slider("Kernel Size", 1, 21, 9, step=2)
    elif operation == "Canny Edge":
        params["low"] = st.slider("Low Threshold", 50, 150, 100)
        params["high"] = st.slider("High Threshold", 150, 300, 200)
    elif operation == "Face Detection (DNN)":
        params["conf_threshold"] = st.slider(
            "Confidence Threshold",
            0.1,
            0.9,
            0.5,
            step=0.05,
        )

    return params


def describe_result(operation, params, image, result):
    original_shape = f"{image.shape[1]} x {image.shape[0]}"
    result_channels = 1 if result.ndim == 2 else result.shape[2]
    summary = [
        f"Mode: {operation}",
        f"Input: {original_shape}px",
        f"Output channels: {result_channels}",
    ]

    if params:
        formatted_params = ", ".join(f"{key}={value}" for key, value in params.items())
        summary.append(f"Params: {formatted_params}")

    if operation == "Object Detection (YOLO)":
        summary.append("Status: placeholder pipeline, original image returned")

    return summary


def save_history(file_name, group, operation, params, original, result, summary):
    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_name": file_name,
        "group": group,
        "operation": operation,
        "params": params,
        "original": original.copy(),
        "result": result.copy(),
        "summary": summary,
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:12]


def render_history():
    st.sidebar.markdown("### Analysis History")

    if not st.session_state.history:
        st.sidebar.caption("No saved runs yet.")
        return

    if st.sidebar.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.selected_history = None
        st.rerun()

    for index, item in enumerate(st.session_state.history):
        label = f"{item['operation']} · {item['file_name']}"
        with st.sidebar.expander(label, expanded=index == 0):
            st.caption(item["time"])
            st.image(item["result"], use_container_width=True)
            if st.button("Open Result", key=f"open_{item['id']}", use_container_width=True):
                st.session_state.selected_history = item["id"]
                st.session_state.current_original = item["original"]
                st.session_state.current_result = item["result"]
                st.session_state.current_summary = item["summary"]
                st.rerun()


def render_workbench():
    st.title("AI Vision Workbench")

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background: #f7f7f4;
            border: 1px solid #deded8;
            border-radius: 8px;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stExpander"] {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.34, 0.66], gap="large")

    with left:
        st.subheader("Input")
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
        )

        mode_group = st.radio(
            "Analysis Type",
            list(PROCESSING_MODES.keys()),
            horizontal=True,
        )
        operation = st.selectbox("Mode", PROCESSING_MODES[mode_group])
        params = build_params(operation)

        run_analysis = st.button(
            "Run Analysis",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

        if uploaded_file is not None:
            image = image_to_array(uploaded_file)
            st.session_state.current_original = image
            st.image(image, caption="Preview", use_container_width=True)
        else:
            image = None
            st.info("Upload an image to start a new analysis run.")

    if run_analysis and uploaded_file is not None and image is not None:
        with st.spinner("Running vision pipeline..."):
            result = run_model(operation, image.copy(), params)
            summary = describe_result(operation, params, image, result)
            st.session_state.current_result = result
            st.session_state.current_summary = summary
            save_history(
                uploaded_file.name,
                mode_group,
                operation,
                params,
                image,
                result,
                summary,
            )
        st.success("Analysis saved to history.")

    with right:
        st.subheader("Result")

        if st.session_state.current_result is None:
            st.info("Your analysis result will appear here.")
            return

        before, after = st.columns(2, gap="medium")
        with before:
            st.image(
                st.session_state.current_original,
                caption="Original",
                use_container_width=True,
            )
        with after:
            st.image(
                st.session_state.current_result,
                caption="Processed Result",
                use_container_width=True,
            )

        st.divider()
        metrics = st.columns(3)
        metrics[0].metric("Saved Runs", len(st.session_state.history))
        metrics[1].metric("Current Mode", operation)
        result_shape = st.session_state.current_result.shape
        metrics[2].metric("Result Size", f"{result_shape[1]} x {result_shape[0]}")

        with st.expander("Run Details", expanded=True):
            for line in st.session_state.current_summary:
                st.write(line)


def render_webcam():
    st.title("Realtime Webcam")
    st.warning("Click Start to enable webcam. Press ESC in the camera window to stop.")

    run = st.checkbox("Start Webcam")
    frame_window = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not available")
            break

        frame = cv2.flip(frame, 1)
        frame = detect_faces_dnn(frame, conf_threshold=0.5)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    cap.release()


init_state()

page = st.sidebar.radio("Workspace", ["Vision Workbench", "Webcam"])
render_history()

if page == "Vision Workbench":
    render_workbench()
else:
    render_webcam()
