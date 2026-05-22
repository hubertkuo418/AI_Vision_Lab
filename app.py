import csv
import io
import json

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.benchmark_runner import benchmark_leaderboard_rows, run_benchmark
from src.comparison_runner import MAX_PIPELINES_PER_COMPARE, run_comparison
from src.dnn_face_detection import detect_faces_dnn
from src.ground_truth import parse_ground_truth_file
from src.history_store import (
    clear_analysis_runs,
    clear_comparison_sessions,
    comparison_session_to_csv,
    comparison_session_to_export,
    init_history_db,
    list_analysis_runs,
    list_comparison_sessions,
    save_analysis_run,
    save_comparison_session,
)
from src.metrics_utils import DETECTION_GROUPS
from src.model_registry import (
    default_params,
    get_pipeline,
    list_categories,
    list_comparison_groups,
    list_comparable_pipelines,
    list_pipelines,
    run_pipeline,
)


st.set_page_config(page_title="AI Vision Workbench", layout="wide")


def init_state():
    init_history_db()
    defaults = {
        "history": list_analysis_runs(),
        "compare_history": list_comparison_sessions(),
        "current_result": None,
        "current_original": None,
        "current_summary": None,
        "current_export": None,
        "selected_history": None,
        "comparison_result": None,
        "comparison_original": None,
        "selected_compare_history": None,
        "benchmark_result": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def image_to_array(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def apply_global_styles():
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


def build_params(pipeline, key_prefix=""):
    params = {}
    for spec in pipeline.params:
        value = st.slider(
            spec.label,
            spec.min_value,
            spec.max_value,
            spec.default,
            step=spec.step,
            key=f"{key_prefix}{pipeline.id}_{spec.name}",
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
        f"Comparison group: {pipeline.comparison_group}",
        f"Input: {original_shape}px",
        f"Output channels: {result_channels}",
        f"Annotations: {len(result.annotations)}",
        f"Latency: {result.metrics.get('latency_ms', 'n/a')} ms",
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


def metrics_table_rows(entries, template):
    rows = []
    for entry in entries:
        metrics = entry.metrics if hasattr(entry, "metrics") else entry["metrics"]
        name = entry.pipeline_name if hasattr(entry, "pipeline_name") else entry["pipeline_name"]
        row = {
            "model": name,
            "latency_ms": metrics.get("latency_ms"),
        }
        if template == "detection":
            row["detections"] = metrics.get("detections")
            row["avg_confidence"] = metrics.get("avg_confidence")
        else:
            row["output_channels"] = metrics.get("output_channels")
            row["pixel_change_ratio"] = metrics.get("pixel_change_ratio")
            row["edge_pixels"] = metrics.get("edge_pixels")
        rows.append(row)
    return rows


def render_comparison_grid(original, entries):
    count = len(entries)
    column_count = 2 if count <= 2 else min(4, count)
    columns = st.columns(column_count)
    for index, entry in enumerate(entries):
        with columns[index % column_count]:
            image = entry.result_image if hasattr(entry, "result_image") else entry["result"]
            name = entry.pipeline_name if hasattr(entry, "pipeline_name") else entry["pipeline_name"]
            st.image(image, caption=name, use_container_width=True)

    if original is not None:
        st.image(original, caption="Original Input", use_container_width=True)


def render_comparison_metrics(entries, comparison_group):
    detection_entries = [
        entry
        for entry in entries
        if (
            entry.comparison_group if hasattr(entry, "comparison_group") else entry["comparison_group"]
        )
        in DETECTION_GROUPS
    ]
    processing_entries = [
        entry for entry in entries if entry not in detection_entries
    ]

    tabs = st.tabs(["Detection Metrics", "Processing Metrics", "Summary"])
    with tabs[0]:
        if detection_entries:
            st.dataframe(
                metrics_table_rows(detection_entries, "detection"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No detection pipelines in this comparison.")

    with tabs[1]:
        if processing_entries:
            st.dataframe(
                metrics_table_rows(processing_entries, "processing"),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No image-processing pipelines in this comparison.")

    with tabs[2]:
        summary = st.session_state.comparison_result
        if summary and hasattr(summary, "summary"):
            st.json(summary.summary)
        elif isinstance(summary, dict):
            st.json(summary.get("summary", {}))
        else:
            st.caption(f"Comparison group: {comparison_group}")


def render_history():
    st.sidebar.markdown("### History")
    history_tab, compare_tab = st.sidebar.tabs(["Single Runs", "Compare Sessions"])

    with history_tab:
        if not st.session_state.history:
            st.sidebar.caption("No saved runs yet.")
        else:
            if st.sidebar.button("Clear Single Runs", use_container_width=True, key="clear_single"):
                clear_analysis_runs()
                st.session_state.history = []
                st.session_state.selected_history = None
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

    with compare_tab:
        if not st.session_state.compare_history:
            st.sidebar.caption("No comparison sessions yet.")
        else:
            if st.sidebar.button("Clear Compare Sessions", use_container_width=True, key="clear_compare"):
                clear_comparison_sessions()
                st.session_state.compare_history = []
                st.session_state.selected_compare_history = None
                st.session_state.comparison_result = None
                st.rerun()

            for index, item in enumerate(st.session_state.compare_history):
                label = f"{item['comparison_group']} - {item['file_name']}"
                with st.sidebar.expander(label, expanded=index == 0):
                    st.caption(item["time"])
                    if st.button("Open Session", key=f"open_compare_{item['id']}", use_container_width=True):
                        st.session_state.selected_compare_history = item["id"]
                        st.session_state.comparison_original = item["original"]
                        st.session_state.comparison_result = item
                        st.rerun()


def render_workbench():
    st.title("AI Vision Workbench")
    apply_global_styles()

    left, right = st.columns([0.34, 0.66], gap="large")

    with left:
        st.subheader("Input")
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=False,
            key="workbench_upload",
        )

        categories = list_categories()
        mode_group = st.radio("Analysis Type", categories, horizontal=True)
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
            from time import perf_counter

            from src.metrics_utils import enrich_pipeline_metrics

            started = perf_counter()
            result = run_pipeline(pipeline.id, image.copy(), params)
            latency_ms = (perf_counter() - started) * 1000
            enrich_pipeline_metrics(result, image, latency_ms, original=image)
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
            st.image(st.session_state.current_original, caption="Original", use_container_width=True)
        with after:
            st.image(st.session_state.current_result, caption="Processed Result", use_container_width=True)

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
                metric_payload = export_data.get("metrics", {})
                st.write(f"Labels: {', '.join(labels) if labels else 'None'}")
                st.json(metric_payload)
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


def render_compare():
    st.title("Model Compare")
    apply_global_styles()
    st.caption(
        f"Run up to {MAX_PIPELINES_PER_COMPARE} pipelines on the same image. "
        "Prefer pipelines from the same comparison group."
    )

    left, right = st.columns([0.34, 0.66], gap="large")

    with left:
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            key="compare_upload",
        )
        groups = list_comparison_groups()
        selected_group = st.selectbox("Comparison Group", groups)
        pipelines = list_comparable_pipelines(selected_group)
        pipeline_options = {pipeline.name: pipeline.id for pipeline in pipelines}
        selected_names = st.multiselect(
            "Pipelines",
            list(pipeline_options.keys()),
            default=list(pipeline_options.keys()),
        )
        selected_ids = [pipeline_options[name] for name in selected_names]

        use_defaults = st.checkbox("Use default parameters", value=True)
        params_map = {}
        if not use_defaults and selected_ids:
            with st.expander("Per-model parameters"):
                for pipeline_id in selected_ids:
                    pipeline = get_pipeline(pipeline_id)
                    st.markdown(f"**{pipeline.name}**")
                    params_map[pipeline_id] = build_params(pipeline, key_prefix="compare_")

        run_compare = st.button(
            "Run Comparison",
            type="primary",
            disabled=uploaded_file is None or not selected_ids,
            use_container_width=True,
        )

        if uploaded_file is not None:
            image = image_to_array(uploaded_file)
            st.session_state.comparison_original = image
            st.image(image, caption="Preview", use_container_width=True)
        else:
            image = None

    if run_compare and uploaded_file is not None and image is not None:
        with st.spinner("Running model comparison..."):
            comparison = run_comparison(
                image,
                selected_ids,
                file_name=uploaded_file.name,
                params_map=params_map or None,
                comparison_group=selected_group,
            )
            saved = save_comparison_session(
                file_name=uploaded_file.name,
                comparison_group=comparison.comparison_group,
                original_image=image,
                comparison_result=comparison,
            )
            st.session_state.comparison_result = comparison
            st.session_state.compare_history = list_comparison_sessions()
            st.session_state.comparison_original = saved["original"]
        st.success("Comparison saved to history.")

    with right:
        st.subheader("Comparison Results")
        result = st.session_state.comparison_result
        if result is None:
            st.info("Run a comparison to see side-by-side outputs.")
            return

        if isinstance(result, dict):
            entries = result["runs"]
            original = result.get("original", st.session_state.comparison_original)
            comparison_group = result["comparison_group"]
            summary = result["summary"]
        else:
            entries = result.entries
            original = st.session_state.comparison_original
            comparison_group = result.comparison_group
            summary = result.summary

        render_comparison_grid(original, entries)
        render_comparison_metrics(entries, comparison_group)

        export_entry = result if isinstance(result, dict) else {
            "id": result.session_id,
            "time": result.created_at,
            "file_name": result.file_name,
            "comparison_group": result.comparison_group,
            "summary": summary,
            "runs": [
                {
                    "pipeline_id": entry.pipeline_id,
                    "pipeline_name": entry.pipeline_name,
                    "comparison_group": entry.comparison_group,
                    "task_type": entry.task_type,
                    "params": entry.params,
                    "metrics": entry.metrics,
                    "messages": entry.messages,
                    "export_data": entry.export_data,
                }
                for entry in result.entries
            ],
        }

        st.download_button(
            "Download Comparison JSON",
            data=json.dumps(comparison_session_to_export(export_entry), indent=2),
            file_name="comparison_session.json",
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Download Comparison CSV",
            data=comparison_session_to_csv(export_entry),
            file_name="comparison_session.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_benchmark():
    st.title("Benchmark")
    apply_global_styles()
    st.caption("Batch-evaluate pipelines across multiple images. Optional ground truth unlocks precision/recall/F1.")

    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="benchmark_upload",
    )
    gt_file = st.file_uploader(
        "Optional Ground Truth JSON",
        type=["json"],
        key="benchmark_gt",
    )

    groups = list_comparison_groups()
    selected_group = st.selectbox("Benchmark Group", groups, key="benchmark_group")
    pipelines = list_comparable_pipelines(selected_group)
    pipeline_options = {pipeline.name: pipeline.id for pipeline in pipelines}
    selected_names = st.multiselect(
        "Pipelines",
        list(pipeline_options.keys()),
        default=list(pipeline_options.keys()),
        key="benchmark_pipelines",
    )
    selected_ids = [pipeline_options[name] for name in selected_names]
    iou_threshold = st.slider("Ground-truth IoU threshold", 0.1, 0.9, 0.5, 0.05)

    run_benchmark_btn = st.button(
        "Run Benchmark",
        type="primary",
        disabled=not uploaded_files or not selected_ids,
        use_container_width=True,
    )

    if run_benchmark_btn and uploaded_files:
        images = {uploaded.name: image_to_array(uploaded) for uploaded in uploaded_files}
        ground_truth = parse_ground_truth_file(gt_file) if gt_file is not None else None

        with st.spinner("Running benchmark..."):
            benchmark = run_benchmark(
                images,
                selected_ids,
                ground_truth=ground_truth,
                iou_threshold=iou_threshold,
                comparison_group=selected_group,
            )
            st.session_state.benchmark_result = benchmark

    result = st.session_state.benchmark_result
    if result is None:
        st.info("Upload images and run a benchmark to see the leaderboard.")
        return

    st.subheader("Leaderboard")
    rows = benchmark_leaderboard_rows(result)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Leaderboard JSON",
        data=json.dumps(
            {
                "created_at": result.created_at,
                "comparison_group": result.comparison_group,
                "has_ground_truth": result.has_ground_truth,
                "leaderboard": rows,
            },
            indent=2,
        ),
        file_name="benchmark_leaderboard.json",
        mime="application/json",
        use_container_width=True,
    )

    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button(
            "Download Leaderboard CSV",
            data=buffer.getvalue(),
            file_name="benchmark_leaderboard.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_catalog():
    st.title("Model Catalog")
    apply_global_styles()
    st.caption("Registered vision pipelines available for workbench, comparison, and benchmark flows.")

    for pipeline in list_comparable_pipelines():
        with st.expander(f"{pipeline.name} ({pipeline.comparison_group})", expanded=False):
            st.write(pipeline.description)
            st.markdown(
                f"""
                - **Category:** {pipeline.category}
                - **Task:** {pipeline.task_type}
                - **Model family:** {pipeline.model_family}
                - **Model version:** {pipeline.model_version}
                - **Weights:** {pipeline.weights_path or "n/a"}
                - **Comparable:** {pipeline.comparable}
                """
            )
            if pipeline.params:
                st.markdown("**Default parameters**")
                st.json(default_params(pipeline.id))


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
apply_global_styles()

page = st.sidebar.radio(
    "Workspace",
    ["Vision Workbench", "Model Compare", "Benchmark", "Model Catalog", "Webcam"],
)
render_history()

if page == "Vision Workbench":
    render_workbench()
elif page == "Model Compare":
    render_compare()
elif page == "Benchmark":
    render_benchmark()
elif page == "Model Catalog":
    render_catalog()
else:
    render_webcam()
