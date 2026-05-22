# AI_Vision_Lab

> A modular computer vision workbench for running, comparing, and benchmarking vision pipelines on the same input.

---

## Overview

AI_Vision_Lab is an interactive computer vision playground that integrates:

- Classical OpenCV image processing
- AI-based face detection (Haar / DNN)
- YOLO object detection with tunable inference parameters
- Vision Workbench with before/after comparison, run history, and JSON export
- **Model Compare** for side-by-side multi-pipeline evaluation on one image
- **Benchmark** for batch runs and optional ground-truth metrics
- Real-time webcam inference
- Dynamic pipeline switching through a registry layer

The goal is to demonstrate a **modular AI vision architecture**, not just isolated scripts.

---

## Features

### Image Processing
- Grayscale conversion
- Gaussian Blur
- Canny Edge Detection
- Histogram Equalization
- Morphological operations (dilate / erode)

### AI Face Detection
- Haar Cascade face detection
- OpenCV DNN SSD face detection
- OpenCV YuNet ONNX face detection
- Comparable under `face_detection` in Model Compare / Benchmark

### AI Object Detection
- YOLOv8n / YOLOv8s / YOLOv8m (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`)
- Legacy pipeline id `object_detection_yolo` aliases to YOLOv8n
- Comparable under `object_detection` in Model Compare / Benchmark
- Confidence / IoU / max-detection tuning
- Structured bounding-box annotations and metrics
- Category-based pipeline selection (`Image Processing` / `AI Vision`)
- Before/after comparison, run history, and JSON export

### Model Compare
- Compare up to 4 pipelines on the same uploaded image
- Comparison groups (`face_detection`, `object_detection`, `edge_detection`, etc.)
- Side-by-side result grid with detection vs processing metric tabs
- Latency, detection count, confidence, and pixel-change metrics
- Saved comparison sessions with JSON/CSV export

### Benchmark
- Batch evaluation across multiple uploaded images with live progress
- Sortable leaderboard split by detection vs processing metrics
- Macro and micro precision / recall / F1 when ground truth is provided
- Optional YOLO warmup before timed runs (reduces cold-start latency bias)
- Per-image drill-down tables and full JSON / CSV export (leaderboard + per-image)
- Ground-truth template download, coverage warnings, and saved benchmark sessions in sidebar history

### Real-time Webcam
- Live video processing
- Real-time DNN face detection
- Mirror preview in Streamlit

---

## System Design

```
Streamlit UI (Workbench / Compare / Benchmark / Catalog / Webcam)
   ↓
Category + Pipeline Selector
   ↓
model_registry.py (VisionPipeline registry)
   ↓
comparison_runner.py / benchmark_runner.py
   ↓
OpenCV / DNN / YOLO pipelines
   ↓
PipelineResult (image + annotations + metrics)
   ↓
history_store.py (SQLite + saved images)
   ↓
Processed output + JSON / CSV export
```

Key idea: separation of UI, pipeline registry, comparison/benchmark runners, and persisted history for scalability.

---

## Project Structure

```
AI_Vision_Lab/
│
├── app.py
├── requirements.txt
├── yolov8n.pt
├── yolov8s.pt
├── yolov8m.pt
├── tests/
│   ├── test_comparison.py
│   ├── test_benchmark.py
│   └── test_model_registry.py
│
├── assets/
│   ├── face_detection.png
│   ├── image_processing.png
│   └── webcam.png
│
├── data/
│   ├── vision_history.sqlite
│   └── history_images/
│
├── models/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000.caffemodel
│   └── face_detection_yunet_2023mar.onnx
│
└── src/
    ├── filters.py
    ├── edges.py
    ├── histogram.py
    ├── morphology.py
    ├── haar_face_detection.py
    ├── dnn_face_detection.py
    ├── yunet_face_detection.py
    ├── yolo_detection.py
    ├── model_paths.py
    ├── pipeline_result.py
    ├── metrics_utils.py
    ├── comparison_runner.py
    ├── benchmark_runner.py
    ├── ground_truth.py
    ├── history_store.py
    └── model_registry.py
```

---

## Demo

### Image Processing
![image processing](assets/image_processing.png)

### Face Detection
![face detection](assets/face_detection.png)

### Webcam Mode
![webcam](assets/webcam.png)

---

## Installation

```bash
git clone https://github.com/hubertkuo418/ai-vision-lab.git
cd ai-vision-lab
pip install -r requirements.txt
```

> On first YOLO run, Ultralytics may download model weights if a `.pt` file is not present locally.

### Model weights

| Pipeline | Files | Notes |
|----------|-------|-------|
| Face DNN | `models/deploy.prototxt`, `models/res10_300x300_ssd_iter_140000.caffemodel` | Bundled with OpenCV face detector samples |
| Face YuNet | `models/face_detection_yunet_2023mar.onnx` | [Download from OpenCV Zoo](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) |
| YOLOv8n/s/m | `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` | Auto-download on first inference via Ultralytics |

Check **Model Catalog** in the app for per-pipeline weight status (`available` / `missing`).

### Comparing models

- Use **Model Compare** or **Benchmark** and select pipelines from the same **comparison group** (`face_detection` or `object_detection`).
- Compare up to 4 pipelines at once; avoid running three large YOLO variants simultaneously on limited RAM.
- Enable **Warmup YOLO** in Benchmark when measuring latency across YOLOv8n/s/m.

---

## Run

```bash
streamlit run app.py
```

Open the sidebar to switch between **Vision Workbench**, **Model Compare**, **Benchmark**, **Model Catalog**, and **Webcam**.

### Ground truth format (Benchmark)

Download the in-app **GT template** or use this structure:

```json
{
  "photo.jpg": [
    {"label": "face", "x": 120, "y": 80, "width": 64, "height": 64}
  ],
  "default": [
    {"label": "person", "x": 40, "y": 30, "width": 120, "height": 200}
  ]
}
```

- Per-file keys match uploaded image names.
- `default` applies the same boxes to every image when a file-specific entry is missing.
- Detection pipelines use macro F1 (per-image average) and micro F1 (global TP/FP/FN pool).
- Enable **Warmup YOLO before timing** when comparing YOLO latency fairly across runs.

### Benchmark exports

| File | Contents |
|------|----------|
| `benchmark_session.json` | Full session: leaderboard, per-image metrics, GT report, params |
| `benchmark_leaderboard.csv` | One row per model (ranked) |
| `benchmark_per_image.csv` | One row per model per image |

Past runs appear under sidebar **Benchmark Sessions**.

---

## Key Concepts

- Classical computer vision fundamentals
- Deep learning-based inference (OpenCV DNN + YOLO)
- Structured pipeline results (`PipelineResult`)
- Comparison groups and unified runtime metrics (`latency_ms`, detections, etc.)
- Real-time video processing pipeline
- Modular architecture design
- Model abstraction layer (`VisionPipeline` registry)
- UI + backend + persistence separation
- Single-run history and multi-model comparison sessions
- Benchmark sessions with macro/micro F1, processing aggregates, and CSV exports

---

## Future Work

- Face recognition (identity-level system) under a separate `face_recognition` group
- MediaPipe / RetinaFace detectors via the same registry factory pattern
- FPS overlay and live performance monitoring in Webcam
- Snapshot & recording system
- Webcam pipeline switching (Haar / YOLO compare mode)

---

## Author

Built by: Hubert Kuo  
Focus: Computer Vision / AI Systems / Machine Learning
