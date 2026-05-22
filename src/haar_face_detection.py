import cv2

from src.pipeline_result import Annotation, BoundingBox

# Load OpenCV built-in Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face_annotations_haar(img):
    """Return structured Haar Cascade face detections."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(30, 30)
    )

    return [
        Annotation(
            label="Face",
            category="face",
            bbox=BoundingBox(int(x), int(y), int(w), int(h)),
            confidence=None,
        )
        for (x, y, w, h) in faces
    ]


def detect_faces_haar(img):
    """
    Perform face detection using Haar Cascade classifier.

    Args:
        img (numpy.ndarray):
            Input image in BGR color format.

    Returns:
        numpy.ndarray:
            Output image with detected face bounding boxes and labels.
    """

    annotations = detect_face_annotations_haar(img)

    for annotation in annotations:
        bbox = annotation.bbox
        # Draw rectangle around detected face
        cv2.rectangle(img, (bbox.x, bbox.y), (bbox.x2, bbox.y2), (0, 255, 0), 3)

        # Add label above bounding box
        cv2.putText(
            img,
            annotation.label,
            (bbox.x, bbox.y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return img
