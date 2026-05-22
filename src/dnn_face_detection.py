import cv2
import numpy as np

from src.model_paths import MODELS_DIR
from src.pipeline_result import Annotation, BoundingBox

DNN_PROTO_PATH = str(MODELS_DIR / "deploy.prototxt")
DNN_WEIGHTS_PATH = str(MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel")

net = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_WEIGHTS_PATH)


def detect_face_annotations_dnn(img, conf_threshold=0.3):
    """Return structured DNN face detections."""
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()
    annotations = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))

            annotations.append(
                Annotation(
                    label="Face",
                    category="face",
                    bbox=BoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1)),
                    confidence=confidence,
                )
            )

    return annotations


def detect_faces_dnn(img, conf_threshold=0.3):
    """
    Perform face detection using OpenCV DNN module.

    Args:
        img (numpy.ndarray):
            Input image frame.

        conf_threshold (float):
            Minimum confidence threshold for valid detections.

    Returns:
        numpy.ndarray:
            Image with detected face bounding boxes and confidence scores.
    """

    annotations = detect_face_annotations_dnn(img, conf_threshold)

    for annotation in annotations:
        bbox = annotation.bbox
        cv2.rectangle(
            img,
            (bbox.x, bbox.y),
            (bbox.x2, bbox.y2),
            (0, 255, 0),
            2
        )

        label = f"{annotation.confidence:.2f}"

        cv2.putText(
            img,
            label,
            (bbox.x, bbox.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img
