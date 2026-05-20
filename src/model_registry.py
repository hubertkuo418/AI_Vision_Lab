from src.filters import to_gray, gaussian_blur
from src.edges import canny_edge
from src.histogram import equalize
from src.morphology import dilate, erode
from src.haar_face_detection import detect_faces_haar
from src.dnn_face_detection import detect_faces_dnn


def run_model(model_name, img, params=None):
   

    if params is None:
        params = {}

    # =========================
    # Classical Computer Vision Pipelines
    # =========================

    if model_name == "Gray":
        return to_gray(img)

    elif model_name == "Gaussian Blur":
        k = params.get("ksize", 9)
        return gaussian_blur(img, (k, k))

    elif model_name == "Canny Edge":
        gray = to_gray(img)
        low = params.get("low", 100)
        high = params.get("high", 200)
        return canny_edge(gray, low, high)

    elif model_name == "Histogram Equalization":
        gray = to_gray(img)
        return equalize(gray)

    elif model_name == "Dilate":
        gray = to_gray(img)
        return dilate(gray)

    elif model_name == "Erode":
        gray = to_gray(img)
        return erode(gray)

    # =========================
    # AI-based Models
    # =========================

    elif model_name == "Face Detection (Haar)":
        return detect_faces_haar(img)

    elif model_name == "Face Detection (DNN)":
        return detect_faces_dnn(img, params.get("conf_threshold", 0.5))

    else:
        # If model name is not recognized, return original image unchanged
        return img