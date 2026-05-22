import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.dnn_face_detection import detect_faces_dnn
from src.history_store import clear_analysis_runs, init_history_db, list_analysis_runs, save_analysis_run
from src.model_registry import get_pipeline, list_categories, list_pipelines, run_pipeline


st.set_page_config(page_title="AI Vision Workbench", layout="wide")


def init_state():
    init_history_db()
    defaults = {
        "history": list_analysis_runs(),
        "current_result": None,
        "current_original": None,
        "current_summary": None,
        "current_export": None,
        "selected_history": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def image_to_array(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def build_params(pipeline):
    params = {}

    for spec in pipeline.params:
        value = st.slider(
            spec.label,
            spec.min_value,
            spec.max_value,
            spec.default,
            step=spec.step,
        )
        params[spec.name] = int(value) if spec.kind == "int" else float(value)

    return params


def describe_result(pipeline, params, image, result):
    original_shape = f"{image.shape[1]} x {image.shape[0]}"
    output = result.image
    result_channels = 1 if output.ndim == 2 else output.shape[2]
    summary = [
        f"Pipeline: {pipeline.name}",
        f"Task: {pipeline.task_type}",
        f"Category: {pipeline.category}",
        f"Input: {original_shape}px",
        f"Output channels: {result_channels}",
        f"Annotations: {len(result.annotations)}",
    ]

    if params:
        formatted_params = ", ".join(f"{key}={value}" for key, value in params.items())
        summary.append(f"Params: {formatted_params}")

    if pipeline.status != "ready":
        summary.append(f"Status: {pipeline.status}")

    return summary


def annotation_rows(export_data):
    if not export_data:
        return []

    rows = []
    for index, item in enumerate(export_data.get("annotations", []), start=1):
        bbox = item["bbox"]
        rows.append(
            {
                "id": index,
                "label": item["label"],
                "category": item["category"],
                "confidence": item["confidence"],
                "x": bbox["x"],
                "y": bbox["y"],
                "width": bbox["width"],
                "height": bbox["height"],
            }
        )
    return rows


def save_history(file_name, group, pipeline, params, original, result, summary, export_data):
    entry = save_analysis_run(
        file_name=file_name,
        category=group,
        pipeline=pipeline,
        params=params,
        original_image=original,
        result_image=result.image,
        summary=summary,
        export_data=export_data,
    )
    st.session_state.history = list_analysis_runs()
    return entry


def render_history():
    st.sidebar.markdown("### Analysis History")

    if not st.session_state.history:
        st.sidebar.caption("No saved runs yet.")
        return

    if st.sidebar.button("Clear History", use_container_width=True):
        clear_analysis_runs()
        st.session_state.history = []
        st.session_state.selected_history = None
        st.session_state.current_original = None
        st.session_state.current_result = None
        st.session_state.current_summary = None
        st.session_state.current_export = None
        st.rerun()

    for index, item in enumerate(st.session_state.history):
        label = f"{item['operation']} - {item['file_name']}"
        with st.sidebar.expander(label, expanded=index == 0):
            st.caption(item["time"])
            st.image(item["result"], use_container_width=True)
            if st.button("Open Result", key=f"open_{item['id']}", use_container_width=True):
                st.session_state.selected_history = item["id"]
                st.session_state.current_original = item["original"]
                st.session_state.current_result = item["result"]
                st.session_state.current_summary = item["summary"]
                st.session_state.current_export = item["export_data"]
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

        categories = list_categories()
        mode_group = st.radio(
            "Analysis Type",
            categories,
            horizontal=True,
        )
        available_pipelines = list_pipelines(mode_group)
        pipeline_names = [item.name for item in available_pipelines]
        operation = st.selectbox("Mode", pipeline_names)
        pipeline = get_pipeline(operation)
        st.caption(f"{pipeline.task_type} - {pipeline.description}")
        params = build_params(pipeline)

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
            result = run_pipeline(pipeline.id, image.copy(), params)
            summary = describe_result(pipeline, params, image, result)
            export_data = result.to_export_dict(pipeline, params, image.shape)
            history_entry = save_history(
                uploaded_file.name,
                mode_group,
                pipeline,
                params,
                image,
                result,
                summary,
                export_data,
            )
            st.session_state.current_result = history_entry["result"]
            st.session_state.current_summary = history_entry["summary"]
            st.session_state.current_export = history_entry["export_data"]
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

        export_data = st.session_state.current_export
        rows = annotation_rows(export_data)

        structured, download = st.columns([0.62, 0.38], gap="medium")
        with structured:
            st.subheader("Structured Output")
            if export_data:
                labels = export_data.get("labels", [])
                metrics = export_data.get("metrics", {})
                st.write(f"Labels: {', '.join(labels) if labels else 'None'}")
                st.json(metrics)
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.caption("No bounding-box annotations for this run.")

        with download:
            st.subheader("Export")
            if export_data:
                st.download_button(
                    "Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"{operation.lower().replace(' ', '_')}_result.json",
                    mime="application/json",
                    use_container_width=True,
                )


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
