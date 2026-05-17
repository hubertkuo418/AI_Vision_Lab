# 🧠 AI_Vision_Lab

A modular computer vision laboratory built with Python, OpenCV, and Streamlit.
It integrates classical image processing and deep learning-based face detection into a single interactive platform.

---

## 🚀 Overview

AI_Vision_Lab is an interactive CV playground that allows users to:

- Upload images for processing
- Apply classical OpenCV operations
- Perform AI-based face detection (Haar / DNN)
- Run real-time webcam face detection
- Switch between models dynamically

The goal is to build a **modular AI vision system**, not just standalone scripts.

---

## 🎯 Features

### 🖼 Image Processing
- Grayscale conversion
- Gaussian Blur
- Canny Edge Detection
- Histogram Equalization
- Morphological operations (dilate / erode)

### 🤖 AI Vision
- Face Detection (Haar Cascade)
- Face Detection (DNN - deep learning)
- Confidence threshold control

### 🎥 Webcam Mode
- Real-time webcam stream
- Live face detection
- Model switching (Haar / DNN)

---

## 🏗 System Design

```
Streamlit UI
   ↓
Model Selector
   ↓
model_registry.py
   ↓
OpenCV / DNN modules
   ↓
Processed output
```

Key idea: **separation of UI and model logic** for scalability.

---

## 📁 Project Structure

```
AI_Vision_Lab/
│
├── app.py
├── requirements.txt
├── README.md
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
    ├── face_detection.py
    ├── dnn_face_detection.py
    ├── model_registry.py
    └── webcam_runner.py
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/AI_Vision_Lab.git
cd AI_Vision_Lab
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
streamlit run app.py
```

---

## 🧠 Key Concepts

- Image preprocessing fundamentals
- Classical computer vision
- Deep learning face detection
- Real-time video processing
- Modular software architecture
- Model switching design pattern

---

## 🚀 Future Work

- YOLO object detection integration
- Face recognition (identity-level)
- FPS performance overlay
- Snapshot / recording feature
- UI redesign (dashboard layout)

---

## 👤 Author

Built by: HubertKuo
Focus: Computer Vision, AI Systems, Machine Learning

---
