import numpy as np

from src.pipeline_result import PipelineResult


DETECTION_GROUPS = {"face_detection", "object_detection"}


def _avg_confidence(annotations):
    scores = [
        annotation.confidence
        for annotation in annotations
        if annotation.confidence is not None
    ]
    if not scores:
        return None
    return round(sum(scores) / len(scores), 4)


def _pixel_change_ratio(original, output):
    original_array = np.asarray(original)
    output_array = np.asarray(output)

    if original_array.ndim == 3:
        original_array = original_array.mean(axis=2)
    if output_array.ndim == 3:
        output_array = output_array.mean(axis=2)

    if original_array.shape != output_array.shape:
        return None

    diff = np.abs(original_array.astype(np.float32) - output_array.astype(np.float32))
    return round(float(diff.mean() / 255.0), 4)


def enrich_pipeline_metrics(result, image, latency_ms, original=None):
    """Attach standardized comparison metrics to a pipeline result."""
    height, width = image.shape[:2]
    metrics = dict(result.metrics)
    metrics["latency_ms"] = round(latency_ms, 2)
    metrics["input_width"] = int(width)
    metrics["input_height"] = int(height)

    if result.annotations:
        metrics.setdefault("detections", len(result.annotations))
        avg_confidence = _avg_confidence(result.annotations)
        if avg_confidence is not None:
            metrics["avg_confidence"] = avg_confidence
        metrics["metric_template"] = "detection"
    else:
        output = result.image
        metrics.setdefault(
            "output_channels",
            1 if getattr(output, "ndim", 0) == 2 else int(output.shape[2]),
        )
        if original is not None:
            ratio = _pixel_change_ratio(original, output)
            if ratio is not None:
                metrics["pixel_change_ratio"] = ratio
        metrics["metric_template"] = "processing"

    result.metrics = metrics
    return result
