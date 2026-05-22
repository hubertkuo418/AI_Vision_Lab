# AI_Vision_Lab

> A modular computer vision workbench combining classical image processing, deep learning detection, and structured analysis output.

---

## Overview

AI_Vision_Lab is an interactive computer vision playground that integrates:

- Classical OpenCV image processing
- AI-based face detection (Haar / DNN)
- YOLO object detection with tunable inference parameters
- Vision Workbench with before/after comparison, run history, and JSON export
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
- Deep learning-based DNN face detection
- YOLOv8 object detection (`yolov8n.pt`)
- Confidence / IoU / max-detection tuning
- Structured bounding-box annotations and metrics
- Category-based pipeline selection (`Image Processing` / `AI Vision`)
- Before/after comparison, run history, and JSON export

### Real-time Webcam
- Live video processing
- Real-time DNN face detection
- Mirror preview in Streamlit

---

## System Design

```
Streamlit UI (Workbench / Webcam)
   ↓
Category + Pipeline Selector
   ↓
model_registry.py (VisionPipeline registry)
   ↓
OpenCV / DNN / YOLO pipelines
   ↓
PipelineResult (image + annotations + metrics)
   ↓
history_store.py (SQLite + saved images)
   ↓
Processed output + JSON export
```

Key idea: separation of UI, pipeline registry, inference modules, and persisted run history for scalability.

---

## Project Structure

```
AI_Vision_Lab/
│
├── app.py
├── requirements.txt
├── yolov8n.pt
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
│   └── res10_300x300_ssd.caffemodel
│
└── src/
    ├── filters.py
    ├── edges.py
    ├── histogram.py
    ├── morphology.py
    ├── haar_face_detection.py
    ├── dnn_face_detection.py
    ├── yolo_detection.py
    ├── pipeline_result.py
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
git clone https://github.com/hubertkuo418/opencv-image-toolkit.git
cd opencv-image-toolkit
pip install -r requirements.txt
```

> On first YOLO run, Ultralytics may download model weights if `yolov8n.pt` is not present locally.

---

## Run

```bash
streamlit run app.py
```

Open the sidebar to switch between **Vision Workbench** and **Webcam**.

---

## Key Concepts

- Classical computer vision fundamentals
- Deep learning-based inference (OpenCV DNN + YOLO)
- Structured pipeline results (`PipelineResult`)
- Real-time video processing pipeline
- Modular architecture design
- Model abstraction layer (`VisionPipeline` registry)
- UI + backend + persistence separation
- Analysis history and exportable JSON artifacts

---

## Future Work

- Face recognition (identity-level system)
- FPS / performance monitoring
- Snapshot & recording system
- Webcam pipeline switching (Haar / YOLO)
- Batch image analysis and dataset export

---

## Author

Built by: Hubert Kuo  
Focus: Computer Vision / AI Systems / Machine Learning
