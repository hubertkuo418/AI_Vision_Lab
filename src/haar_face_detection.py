import cv2

# Load OpenCV built-in Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

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

    # Convert image to grayscale (Haar Cascade works on grayscale images)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=6,
        minSize=(30, 30)
    )

    # Draw bounding boxes and labels for each detected face
    for (x, y, w, h) in faces:

        # Draw rectangle around detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Add label above bounding box
        cv2.putText(
            img,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    return img