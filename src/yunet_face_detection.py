from pathlib import Path

import cv2
import numpy as np

from src.model_paths import MODELS_DIR
from src.pipeline_result import Annotation, BoundingBox


YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
_DEFAULT_SCORE_THRESHOLD = 0.6
_DEFAULT_NMS_THRESHOLD = 0.3


def _ensure_model_path():
    if not YUNET_MODEL_PATH.exists():
        raise FileNotFoundError(
            "YuNet model weights not found. Download "
            f"{YUNET_MODEL_PATH.name} into the models/ directory. "
            "See README.md for the download link."
        )
    return str(YUNET_MODEL_PATH)


def _create_detector(image_shape, score_threshold, nms_threshold):
    height, width = image_shape[:2]
    detector = cv2.FaceDetectorYN.create(
        _ensure_model_path(),
        "",
        (width, height),
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        top_k=5000,
    )
    return detector


def detect_face_annotations_yunet(img, conf_threshold=0.6, nms_threshold=0.3):
    """Return structured YuNet face detections."""
    detector = _create_detector(img.shape, conf_threshold, nms_threshold)
    _, faces = detector.detect(img)
    annotations = []

    if faces is None:
        return annotations

    face_rows = faces[0] if faces.ndim == 3 else faces
    for face in face_rows:
        if face is None or len(face) < 5:
            continue
        x, y, width, height, score = face[:5]
        if width <= 0 or height <= 0:
            continue
        if score < conf_threshold:
            continue
        annotations.append(
            Annotation(
                label="Face",
                category="face",
                bbox=BoundingBox(int(x), int(y), int(width), int(height)),
                confidence=float(score),
            )
        )

    return annotations


def detect_faces_yunet(img, conf_threshold=0.6, nms_threshold=0.3):
    """Draw YuNet face detections on the image."""
    output = img.copy()
    annotations = detect_face_annotations_yunet(
        output,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )

    for annotation in annotations:
        bbox = annotation.bbox
        cv2.rectangle(
            output,
            (bbox.x, bbox.y),
            (bbox.x2, bbox.y2),
            (255, 128, 0),
            2,
        )
        if annotation.confidence is not None:
            label = f"{annotation.confidence:.2f}"
            cv2.putText(
                output,
                label,
                (bbox.x, max(18, bbox.y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 128, 0),
                2,
            )

    return output
