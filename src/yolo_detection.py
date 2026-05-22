import cv2
import numpy as np

from src.pipeline_result import Annotation, BoundingBox


_YOLO_MODELS = {}


def _load_model(weights_path="yolov8n.pt"):
    if weights_path not in _YOLO_MODELS:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is required for YOLO object detection. "
                "Install project dependencies with: pip install -r requirements.txt"
            ) from exc

        _YOLO_MODELS[weights_path] = YOLO(weights_path)

    return _YOLO_MODELS[weights_path]


def detect_objects_yolo(
    img,
    *,
    weights_path="yolov8n.pt",
    conf_threshold=0.35,
    iou_threshold=0.5,
    max_det=100,
):
    """Detect objects with YOLO and return an annotated image plus structured boxes."""
    model = _load_model(weights_path)
    predictions = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        max_det=max_det,
        verbose=False,
    )

    result = predictions[0]
    annotated = img.copy()
    annotations = []

    if result.boxes is None:
        return annotated, annotations, weights_path

    class_names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = [int(value) for value in box]
        h, w = annotated.shape[:2]
        x1 = int(max(0, min(x1, w - 1)))
        y1 = int(max(0, min(y1, h - 1)))
        x2 = int(max(0, min(x2, w - 1)))
        y2 = int(max(0, min(y2, h - 1)))
        label = class_names.get(class_id, str(class_id))

        annotations.append(
            Annotation(
                label=label,
                category="object",
                bbox=BoundingBox(x1, y1, max(0, x2 - x1), max(0, y2 - y1)),
                confidence=float(confidence),
            )
        )

        color = _label_color(label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {confidence:.2f}"
        cv2.putText(
            annotated,
            caption,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated, annotations, weights_path


def _label_color(label):
    seed = sum(ord(char) for char in label)
    rng = np.random.default_rng(seed)
    return tuple(int(value) for value in rng.integers(64, 256, size=3))
