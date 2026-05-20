# AI_Vision_Lab

> A modular computer vision system combining classical image processing and deep learning-based face detection.

---

## 🚀 Overview

AI_Vision_Lab is an interactive computer vision playground that integrates:

- Classical OpenCV image processing
- AI-based face detection (Haar / DNN)
- Real-time webcam inference
- Dynamic model switching system

The goal is to demonstrate a **modular AI vision architecture**, not just isolated scripts.

---

## ✨ Features

### 🖼 Image Processing
- Grayscale conversion
- Gaussian Blur
- Canny Edge Detection
- Histogram Equalization
- Morphological operations (dilate / erode)

### 🤖 AI Face Detection
- Haar Cascade face detection
- Deep learning-based DNN face detection
- Confidence threshold tuning

### 📹 Real-time Webcam
- Live video processing
- Real-time face detection
- Model switching (Haar / DNN)

---

## 🧠 System Design

```
Streamlit UI
   ↓
Model Selector
   ↓
model_registry.py
   ↓
OpenCV / DNN pipelines
   ↓
Processed output
```

Key idea: separation of UI, logic, and model pipelines for scalability.

---

## 📁 Project Structure

```
AI_Vision_Lab/
│
├── app.py
├── requirements.txt
│
├── assets/
│   ├── face_detection.png
│   ├── image_processing.png
│   └── webcam.png
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
    ├── model_registry.py
    └── webcam_runner.py
```

---

## 🎯 Demo

### Image Processing
![image processing](assets/image_processing.png)

### Face Detection
![face detection](assets/face_detection.png)

### Webcam Mode
![webcam](assets/webcam.png)

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

## 🧩 Key Concepts

- Classical computer vision fundamentals
- Deep learning-based inference (OpenCV DNN)
- Real-time video processing pipeline
- Modular architecture design
- Model abstraction layer
- UI + backend separation

---

## 🔮 Future Work

- YOLO object detection integration
- Face recognition (identity-level system)
- FPS / performance monitoring
- Snapshot & recording system
- UI redesign (dashboard-style interface)

---

## 👤 Author

Built by: Hubert Kuo  
Focus: Computer Vision / AI Systems / Machine Learning