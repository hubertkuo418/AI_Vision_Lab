from dataclasses import dataclass
from typing import Callable

from src.dnn_face_detection import detect_face_annotations_dnn, detect_faces_dnn
from src.edges import canny_edge
from src.filters import gaussian_blur, to_gray
from src.haar_face_detection import detect_face_annotations_haar, detect_faces_haar
from src.histogram import equalize
from src.morphology import dilate, erode
from src.pipeline_result import PipelineResult


@dataclass(frozen=True)
class ParamSpec:
    name: str
    label: str
    kind: str
    default: int | float
    min_value: int | float
    max_value: int | float
    step: int | float


@dataclass(frozen=True)
class VisionPipeline:
    id: str
    name: str
    category: str
    task_type: str
    description: str
    runner: Callable
    params: tuple[ParamSpec, ...] = ()
    status: str = "ready"


def _gray(img, params):
    result = to_gray(img)
    return PipelineResult(
        image=result,
        labels=["grayscale"],
        metrics={"output_channels": 1},
        messages=["Converted image to grayscale."],
    )


def _gaussian_blur(img, params):
    ksize = int(params.get("ksize", 9))
    result = gaussian_blur(img, (ksize, ksize))
    return PipelineResult(
        image=result,
        labels=["blurred"],
        metrics={"kernel_size": ksize, "output_channels": int(result.shape[2]) if result.ndim == 3 else 1},
        messages=["Applied Gaussian blur."],
    )


def _canny_edge(img, params):
    gray = to_gray(img)
    low = int(params.get("low", 100))
    high = int(params.get("high", 200))
    result = canny_edge(gray, low, high)
    return PipelineResult(
        image=result,
        labels=["edges"],
        metrics={
            "low_threshold": low,
            "high_threshold": high,
            "edge_pixels": int((result > 0).sum()),
        },
        messages=["Detected edges with Canny."],
    )


def _histogram_equalization(img, params):
    gray = to_gray(img)
    result = equalize(gray)
    return PipelineResult(
        image=result,
        labels=["contrast_enhanced"],
        metrics={"output_channels": 1},
        messages=["Applied histogram equalization."],
    )


def _dilate(img, params):
    gray = to_gray(img)
    result = dilate(gray)
    return PipelineResult(
        image=result,
        labels=["morphology"],
        metrics={"operation": "dilate"},
        messages=["Applied dilation."],
    )


def _erode(img, params):
    gray = to_gray(img)
    result = erode(gray)
    return PipelineResult(
        image=result,
        labels=["morphology"],
        metrics={"operation": "erode"},
        messages=["Applied erosion."],
    )


def _face_detection_haar(img, params):
    annotations = detect_face_annotations_haar(img)
    result = detect_faces_haar(img)
    return PipelineResult(
        image=result,
        annotations=annotations,
        labels=sorted({annotation.label for annotation in annotations}),
        metrics={"detections": len(annotations)},
        messages=[f"Detected {len(annotations)} face(s)."],
    )


def _face_detection_dnn(img, params):
    threshold = float(params.get("conf_threshold", 0.5))
    annotations = detect_face_annotations_dnn(img, threshold)
    result = detect_faces_dnn(img, threshold)
    return PipelineResult(
        image=result,
        annotations=annotations,
        labels=sorted({annotation.label for annotation in annotations}),
        metrics={"detections": len(annotations), "confidence_threshold": threshold},
        messages=[f"Detected {len(annotations)} face(s)."],
    )


def _placeholder_yolo(img, params):
    return PipelineResult(
        image=img,
        labels=[],
        metrics={"detections": 0},
        messages=["YOLO object detection is reserved for a future model integration."],
    )


PIPELINES = (
    VisionPipeline(
        id="gray",
        name="Gray",
        category="Image Processing",
        task_type="Color Transform",
        description="Convert the input image to grayscale.",
        runner=_gray,
    ),
    VisionPipeline(
        id="gaussian_blur",
        name="Gaussian Blur",
        category="Image Processing",
        task_type="Denoising",
        description="Smooth the image with a configurable Gaussian kernel.",
        runner=_gaussian_blur,
        params=(
            ParamSpec("ksize", "Kernel Size", "int", 9, 1, 21, 2),
        ),
    ),
    VisionPipeline(
        id="canny_edge",
        name="Canny Edge",
        category="Image Processing",
        task_type="Edge Detection",
        description="Detect strong image edges with low and high thresholds.",
        runner=_canny_edge,
        params=(
            ParamSpec("low", "Low Threshold", "int", 100, 50, 150, 1),
            ParamSpec("high", "High Threshold", "int", 200, 150, 300, 1),
        ),
    ),
    VisionPipeline(
        id="histogram_equalization",
        name="Histogram Equalization",
        category="Image Processing",
        task_type="Contrast Enhancement",
        description="Enhance local contrast through grayscale histogram equalization.",
        runner=_histogram_equalization,
    ),
    VisionPipeline(
        id="dilate",
        name="Dilate",
        category="Image Processing",
        task_type="Morphology",
        description="Expand bright regions in a grayscale representation.",
        runner=_dilate,
    ),
    VisionPipeline(
        id="erode",
        name="Erode",
        category="Image Processing",
        task_type="Morphology",
        description="Shrink bright regions in a grayscale representation.",
        runner=_erode,
    ),
    VisionPipeline(
        id="face_detection_haar",
        name="Face Detection (Haar)",
        category="AI Vision",
        task_type="Face Detection",
        description="Detect frontal faces using OpenCV Haar Cascade.",
        runner=_face_detection_haar,
    ),
    VisionPipeline(
        id="face_detection_dnn",
        name="Face Detection (DNN)",
        category="AI Vision",
        task_type="Face Detection",
        description="Detect faces using the OpenCV DNN SSD face detector.",
        runner=_face_detection_dnn,
        params=(
            ParamSpec("conf_threshold", "Confidence Threshold", "float", 0.5, 0.1, 0.9, 0.05),
        ),
    ),
    VisionPipeline(
        id="object_detection_yolo",
        name="Object Detection (YOLO)",
        category="AI Vision",
        task_type="Object Detection",
        description="Reserved pipeline slot for a future YOLO detector.",
        runner=_placeholder_yolo,
        status="placeholder",
    ),
)


PIPELINE_BY_ID = {pipeline.id: pipeline for pipeline in PIPELINES}
PIPELINE_BY_NAME = {pipeline.name: pipeline for pipeline in PIPELINES}


def list_pipelines(category=None):
    if category is None:
        return list(PIPELINES)
    return [pipeline for pipeline in PIPELINES if pipeline.category == category]


def list_categories():
    seen = []
    for pipeline in PIPELINES:
        if pipeline.category not in seen:
            seen.append(pipeline.category)
    return seen


def get_pipeline(identifier):
    if identifier in PIPELINE_BY_ID:
        return PIPELINE_BY_ID[identifier]
    if identifier in PIPELINE_BY_NAME:
        return PIPELINE_BY_NAME[identifier]
    raise KeyError(f"Unknown vision pipeline: {identifier}")


def run_pipeline(identifier, img, params=None):
    pipeline = get_pipeline(identifier)
    if params is None:
        params = {}
    result = pipeline.runner(img, params)
    if isinstance(result, PipelineResult):
        return result
    return PipelineResult(image=result)


def run_model(model_name, img, params=None):
    return run_pipeline(model_name, img, params).image
