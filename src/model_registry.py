from dataclasses import dataclass
from typing import Callable

from src.dnn_face_detection import detect_face_annotations_dnn, detect_faces_dnn
from src.edges import canny_edge
from src.filters import gaussian_blur, to_gray
from src.haar_face_detection import detect_face_annotations_haar, detect_faces_haar
from src.histogram import equalize
from src.model_paths import pipeline_status, weights_available
from src.morphology import dilate, erode
from src.pipeline_result import PipelineResult
from src.yolo_detection import detect_objects_yolo
from src.yunet_face_detection import detect_face_annotations_yunet, detect_faces_yunet


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
    comparison_group: str
    params: tuple[ParamSpec, ...] = ()
    status: str = "ready"
    comparable: bool = True
    model_family: str = "opencv_classic"
    model_version: str = "default"
    weights_path: str | None = None
    needs_warmup: bool = False


FACE_CONF_PARAM = (
    ParamSpec("conf_threshold", "Confidence Threshold", "float", 0.5, 0.1, 0.9, 0.05),
)

YOLO_PARAM_SPECS = (
    ParamSpec("conf_threshold", "Confidence Threshold", "float", 0.35, 0.05, 0.95, 0.05),
    ParamSpec("iou_threshold", "IoU Threshold", "float", 0.5, 0.1, 0.9, 0.05),
    ParamSpec("max_det", "Max Detections", "int", 100, 1, 300, 1),
)


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


def _make_face_runner(backend):
    def _runner(img, params):
        threshold = float(params.get("conf_threshold", 0.5))

        if backend == "haar":
            annotations = detect_face_annotations_haar(img)
            result = detect_faces_haar(img)
        elif backend == "dnn":
            annotations = detect_face_annotations_dnn(img, threshold)
            result = detect_faces_dnn(img, threshold)
        elif backend == "yunet":
            annotations = detect_face_annotations_yunet(img, threshold)
            result = detect_faces_yunet(img, threshold)
        else:
            raise ValueError(f"Unknown face backend: {backend}")

        return PipelineResult(
            image=result,
            annotations=annotations,
            labels=sorted({annotation.label for annotation in annotations}),
            metrics={"detections": len(annotations), "confidence_threshold": threshold},
            messages=[f"Detected {len(annotations)} face(s)."],
        )

    return _runner


def _make_yolo_runner(weights_path, model_version):
    def _runner(img, params):
        conf_threshold = float(params.get("conf_threshold", 0.35))
        iou_threshold = float(params.get("iou_threshold", 0.5))
        max_det = int(params.get("max_det", 100))
        annotated, annotations, model_name = detect_objects_yolo(
            img,
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_det=max_det,
        )

        return PipelineResult(
            image=annotated,
            annotations=annotations,
            labels=sorted({annotation.label for annotation in annotations}),
            metrics={
                "detections": len(annotations),
                "confidence_threshold": conf_threshold,
                "iou_threshold": iou_threshold,
                "max_detections": max_det,
                "model": model_name,
                "model_version": model_version,
            },
            messages=[f"Detected {len(annotations)} object(s) with {model_name}."],
        )

    return _runner


def _yolo_pipeline(
    pipeline_id,
    name,
    weights_path,
    model_version,
    description=None,
):
    description = description or f"Detect common objects with YOLOv8{model_version}."
    return VisionPipeline(
        id=pipeline_id,
        name=name,
        category="AI Vision",
        task_type="Object Detection",
        description=description,
        runner=_make_yolo_runner(weights_path, model_version),
        comparison_group="object_detection",
        params=YOLO_PARAM_SPECS,
        status=pipeline_status(weights_path),
        model_family="yolov8",
        model_version=model_version,
        weights_path=weights_path,
        needs_warmup=True,
    )


PIPELINES = (
    VisionPipeline(
        id="gray",
        name="Gray",
        category="Image Processing",
        task_type="Color Transform",
        description="Convert the input image to grayscale.",
        runner=_gray,
        comparison_group="color_transform",
        model_family="opencv_classic",
    ),
    VisionPipeline(
        id="gaussian_blur",
        name="Gaussian Blur",
        category="Image Processing",
        task_type="Denoising",
        description="Smooth the image with a configurable Gaussian kernel.",
        runner=_gaussian_blur,
        comparison_group="denoising",
        params=(ParamSpec("ksize", "Kernel Size", "int", 9, 1, 21, 2),),
        model_family="opencv_classic",
    ),
    VisionPipeline(
        id="canny_edge",
        name="Canny Edge",
        category="Image Processing",
        task_type="Edge Detection",
        description="Detect strong image edges with low and high thresholds.",
        runner=_canny_edge,
        comparison_group="edge_detection",
        params=(
            ParamSpec("low", "Low Threshold", "int", 100, 50, 150, 1),
            ParamSpec("high", "High Threshold", "int", 200, 150, 300, 1),
        ),
        model_family="opencv_classic",
    ),
    VisionPipeline(
        id="histogram_equalization",
        name="Histogram Equalization",
        category="Image Processing",
        task_type="Contrast Enhancement",
        description="Enhance local contrast through grayscale histogram equalization.",
        runner=_histogram_equalization,
        comparison_group="contrast_enhancement",
        model_family="opencv_classic",
    ),
    VisionPipeline(
        id="dilate",
        name="Dilate",
        category="Image Processing",
        task_type="Morphology",
        description="Expand bright regions in a grayscale representation.",
        runner=_dilate,
        comparison_group="morphology",
        model_family="opencv_classic",
    ),
    VisionPipeline(
        id="erode",
        name="Erode",
        category="Image Processing",
        task_type="Morphology",
        description="Shrink bright regions in a grayscale representation.",
        runner=_erode,
        comparison_group="morphology",
        model_family="opencv_classic",
    ),
    VisionPipeline(
        id="face_detection_haar",
        name="Face Detection (Haar)",
        category="AI Vision",
        task_type="Face Detection",
        description="Detect frontal faces using OpenCV Haar Cascade.",
        runner=_make_face_runner("haar"),
        comparison_group="face_detection",
        model_family="opencv_haar",
    ),
    VisionPipeline(
        id="face_detection_dnn",
        name="Face Detection (DNN)",
        category="AI Vision",
        task_type="Face Detection",
        description="Detect faces using the OpenCV DNN SSD face detector.",
        runner=_make_face_runner("dnn"),
        comparison_group="face_detection",
        params=FACE_CONF_PARAM,
        status=pipeline_status("models/res10_300x300_ssd_iter_140000.caffemodel"),
        model_family="opencv_dnn",
        model_version="ssd_300",
        weights_path="models/res10_300x300_ssd_iter_140000.caffemodel",
    ),
    VisionPipeline(
        id="face_detection_yunet",
        name="Face Detection (YuNet)",
        category="AI Vision",
        task_type="Face Detection",
        description="Detect faces using OpenCV YuNet ONNX detector.",
        runner=_make_face_runner("yunet"),
        comparison_group="face_detection",
        params=FACE_CONF_PARAM,
        status=pipeline_status("models/face_detection_yunet_2023mar.onnx"),
        model_family="opencv_yunet",
        model_version="2023mar",
        weights_path="models/face_detection_yunet_2023mar.onnx",
    ),
    _yolo_pipeline(
        "object_detection_yolov8n",
        "Object Detection (YOLOv8n)",
        "yolov8n.pt",
        "n",
    ),
    _yolo_pipeline(
        "object_detection_yolov8s",
        "Object Detection (YOLOv8s)",
        "yolov8s.pt",
        "s",
    ),
    _yolo_pipeline(
        "object_detection_yolov8m",
        "Object Detection (YOLOv8m)",
        "yolov8m.pt",
        "m",
    ),
)


PIPELINE_BY_ID = {pipeline.id: pipeline for pipeline in PIPELINES}
PIPELINE_BY_NAME = {pipeline.name: pipeline for pipeline in PIPELINES}

# Backward-compatible alias for saved history entries.
PIPELINE_BY_ID["object_detection_yolo"] = PIPELINE_BY_ID["object_detection_yolov8n"]
_legacy_yolo = VisionPipeline(
    id="object_detection_yolo",
    name="Object Detection (YOLO)",
    category=PIPELINE_BY_ID["object_detection_yolov8n"].category,
    task_type=PIPELINE_BY_ID["object_detection_yolov8n"].task_type,
    description="Legacy alias for YOLOv8n object detection.",
    runner=PIPELINE_BY_ID["object_detection_yolov8n"].runner,
    comparison_group="object_detection",
    params=YOLO_PARAM_SPECS,
    status=PIPELINE_BY_ID["object_detection_yolov8n"].status,
    model_family="yolov8",
    model_version="n",
    weights_path="yolov8n.pt",
    needs_warmup=True,
)
PIPELINE_BY_NAME["Object Detection (YOLO)"] = _legacy_yolo


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


def list_comparable_pipelines(group=None, only_ready=False):
    pipelines = [pipeline for pipeline in PIPELINES if pipeline.comparable]
    if only_ready:
        pipelines = [pipeline for pipeline in pipelines if pipeline.status == "ready"]
    if group is None:
        return pipelines
    return [pipeline for pipeline in pipelines if pipeline.comparison_group == group]


def list_comparison_groups():
    groups = []
    for pipeline in list_comparable_pipelines():
        if pipeline.comparison_group not in groups:
            groups.append(pipeline.comparison_group)
    return groups


def default_params(identifier):
    pipeline = get_pipeline(identifier)
    params = {}
    for spec in pipeline.params:
        params[spec.name] = spec.default if spec.kind == "int" else float(spec.default)
    return params


def get_pipeline(identifier):
    if identifier in PIPELINE_BY_ID:
        return PIPELINE_BY_ID[identifier]
    if identifier in PIPELINE_BY_NAME:
        return PIPELINE_BY_NAME[identifier]
    raise KeyError(f"Unknown vision pipeline: {identifier}")


def run_pipeline(identifier, img, params=None):
    pipeline = get_pipeline(identifier)
    if pipeline.status == "missing_weights":
        raise FileNotFoundError(
            f"Pipeline '{pipeline.name}' is missing weights at {pipeline.weights_path}."
        )
    if params is None:
        params = {}
    result = pipeline.runner(img, params)
    if isinstance(result, PipelineResult):
        return result
    return PipelineResult(image=result)


def run_model(model_name, img, params=None):
    return run_pipeline(model_name, img, params).image


def weights_status(weights_path):
    return "available" if weights_available(weights_path) else "missing"
