import cv2
import numpy as np

# Load pre-trained DNN face detection model
# deploy.prototxt: model architecture definition
# .caffemodel: pre-trained weights
net = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

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

    # Get image dimensions
    (h, w) = img.shape[:2]

    # Convert image into blob format for DNN inference
    # Resize to 300x300 and normalize with mean subtraction
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)),
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0)
    )

    # Set blob as network input
    net.setInput(blob)

    # Run forward pass to generate detections
    detections = net.forward()

    # Iterate through all detected objects
    for i in range(detections.shape[2]):

        # Extract confidence score
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > conf_threshold:

            # Compute bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw bounding box around detected face
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display confidence score
            label = f"{confidence:.2f}"

            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    return img