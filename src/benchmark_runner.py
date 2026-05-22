from dataclasses import dataclass, field
from datetime import datetime
from statistics import mean, pstdev
from time import perf_counter
from uuid import uuid4

from src.comparison_runner import MAX_PIPELINES_PER_COMPARE, default_params_map
from src.ground_truth import (
    aggregate_evaluations,
    build_ground_truth_report,
    evaluate_predictions,
    macro_average_evaluations,
    parse_ground_truth_payload,
)
from src.metrics_utils import DETECTION_GROUPS, enrich_pipeline_metrics
from src.model_registry import get_pipeline, run_pipeline

@dataclass
class BenchmarkOptions:
    warmup: bool = True
    iou_threshold: float = 0.5


@dataclass
class ImageBenchmarkResult:
    file_name: str
    pipeline_id: str
    pipeline_name: str
    comparison_group: str
    metric_template: str
    metrics: dict
    evaluation: dict | None = None


@dataclass
class PipelineBenchmarkSummary:
    pipeline_id: str
    pipeline_name: str
    comparison_group: str
    metric_template: str
    image_count: int
    avg_latency_ms: float
    latency_std_ms: float
    detections_std: float | None = None
    avg_detections: float | None = None
    avg_confidence: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    macro_f1: float | None = None
    micro_precision: float | None = None
    micro_recall: float | None = None
    micro_f1: float | None = None
    avg_pixel_change_ratio: float | None = None
    avg_edge_pixels: float | None = None
    per_image: list[ImageBenchmarkResult] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    session_id: str
    created_at: str
    comparison_group: str
    has_ground_truth: bool
    warmup_applied: bool
    ground_truth_report: dict
    pipeline_ids: list[str]
    params_map: dict
    summaries: list[PipelineBenchmarkSummary] = field(default_factory=list)


def _resolve_ground_truth(ground_truth, file_name):
    if ground_truth is None:
        return []

    if file_name in ground_truth:
        return ground_truth[file_name]
    if "default" in ground_truth:
        return ground_truth["default"]
    return []


def _warmup_pipelines(images, pipeline_ids, params_map, options):
    if not images or not options.warmup:
        return False

    first_image = next(iter(images.values()))
    warmed = False
    for pipeline_id in pipeline_ids:
        pipeline = get_pipeline(pipeline_id)
        if not pipeline.needs_warmup or pipeline.status == "missing_weights":
            continue
        params = params_map.get(pipeline_id, {})
        run_pipeline(pipeline_id, first_image.copy(), params)
        warmed = True
    return warmed


def _safe_mean(values):
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(mean(filtered), 4)


def _build_summary(pipeline, rows):
    latencies = [row.metrics["latency_ms"] for row in rows]
    detections = [row.metrics.get("detections") for row in rows]
    confidences = [row.metrics.get("avg_confidence") for row in rows]
    pixel_ratios = [row.metrics.get("pixel_change_ratio") for row in rows]
    edge_pixels = [row.metrics.get("edge_pixels") for row in rows]
    template = rows[0].metric_template if rows else "processing"

    summary = PipelineBenchmarkSummary(
        pipeline_id=pipeline.id,
        pipeline_name=pipeline.name,
        comparison_group=pipeline.comparison_group,
        metric_template=template,
        image_count=len(rows),
        avg_latency_ms=round(mean(latencies), 2),
        latency_std_ms=round(pstdev(latencies), 2) if len(latencies) > 1 else 0.0,
        per_image=rows,
    )

    if template == "detection":
        detection_values = [value for value in detections if value is not None]
        summary.avg_detections = round(mean(detection_values), 2) if detection_values else None
        summary.detections_std = (
            round(pstdev(detection_values), 2) if len(detection_values) > 1 else 0.0
        )
        summary.avg_confidence = _safe_mean(confidences)

        evaluated = [row.evaluation for row in rows if row.evaluation]
        if evaluated:
            macro_scores = macro_average_evaluations(evaluated)
            micro_scores = aggregate_evaluations(evaluated)
            summary.macro_precision = macro_scores["precision"]
            summary.macro_recall = macro_scores["recall"]
            summary.macro_f1 = macro_scores["f1"]
            summary.micro_precision = micro_scores["precision"]
            summary.micro_recall = micro_scores["recall"]
            summary.micro_f1 = micro_scores["f1"]
    else:
        summary.avg_pixel_change_ratio = _safe_mean(pixel_ratios)
        edge_values = [value for value in edge_pixels if value is not None]
        summary.avg_edge_pixels = round(mean(edge_values), 2) if edge_values else None

    return summary


def run_benchmark(
    images,
    pipeline_ids,
    *,
    params_map=None,
    ground_truth=None,
    options=None,
    comparison_group=None,
    progress_callback=None,
    iou_threshold=0.5,
):
    if not images:
        raise ValueError("Upload at least one image for benchmarking.")

    if len(pipeline_ids) > MAX_PIPELINES_PER_COMPARE:
        raise ValueError(
            f"Benchmark up to {MAX_PIPELINES_PER_COMPARE} pipelines at once."
        )

    options = options or BenchmarkOptions(iou_threshold=iou_threshold)
    params_map = params_map or default_params_map(pipeline_ids)
    gt_data = parse_ground_truth_payload(ground_truth) if ground_truth else None
    require_detection_gt = any(
        get_pipeline(pipeline_id).comparison_group in DETECTION_GROUPS
        for pipeline_id in pipeline_ids
    )
    ground_truth_report = build_ground_truth_report(
        list(images.keys()),
        gt_data,
        require_detection_gt=require_detection_gt and gt_data is not None,
    )

    warmup_applied = _warmup_pipelines(images, pipeline_ids, params_map, options)
    per_pipeline = {pipeline_id: [] for pipeline_id in pipeline_ids}
    total_steps = len(images) * len(pipeline_ids)
    current_step = 0

    for file_name, image in images.items():
        for pipeline_id in pipeline_ids:
            pipeline = get_pipeline(pipeline_id)
            params = params_map.get(pipeline_id, {})

            started = perf_counter()
            result = run_pipeline(pipeline_id, image.copy(), params)
            latency_ms = (perf_counter() - started) * 1000
            enrich_pipeline_metrics(result, image, latency_ms, original=image)
            if warmup_applied:
                result.metrics["warmup_applied"] = True
            export_data = result.to_export_dict(pipeline, params, image.shape)

            evaluation = None
            if gt_data is not None and pipeline.comparison_group in DETECTION_GROUPS:
                gt_boxes = _resolve_ground_truth(gt_data, file_name)
                if gt_boxes:
                    evaluation = evaluate_predictions(
                        export_data.get("annotations", []),
                        gt_boxes,
                        iou_threshold=options.iou_threshold,
                    )

            per_pipeline[pipeline_id].append(
                ImageBenchmarkResult(
                    file_name=file_name,
                    pipeline_id=pipeline.id,
                    pipeline_name=pipeline.name,
                    comparison_group=pipeline.comparison_group,
                    metric_template=result.metrics.get("metric_template", "processing"),
                    metrics=result.metrics,
                    evaluation=evaluation,
                )
            )

            current_step += 1
            if progress_callback is not None:
                progress_callback(
                    current_step,
                    total_steps,
                    f"{file_name} · {pipeline.name}",
                )

    summaries = []
    groups = []
    for pipeline_id in pipeline_ids:
        pipeline = get_pipeline(pipeline_id)
        rows = per_pipeline[pipeline_id]
        groups.append(pipeline.comparison_group)
        summaries.append(_build_summary(pipeline, rows))

    resolved_group = comparison_group or (
        groups[0] if len(set(groups)) == 1 else "mixed"
    )

    return BenchmarkResult(
        session_id=uuid4().hex,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        comparison_group=resolved_group,
        has_ground_truth=gt_data is not None,
        warmup_applied=warmup_applied,
        ground_truth_report=ground_truth_report,
        pipeline_ids=list(pipeline_ids),
        params_map=params_map,
        summaries=summaries,
    )


def benchmark_leaderboard_rows(benchmark_result, sort_by=None, descending=False):
    rows = []
    for summary in benchmark_result.summaries:
        row = {
            "model": summary.pipeline_name,
            "comparison_group": summary.comparison_group,
            "metric_template": summary.metric_template,
            "images": summary.image_count,
            "avg_latency_ms": summary.avg_latency_ms,
            "latency_std_ms": summary.latency_std_ms,
        }
        if summary.metric_template == "detection":
            row["avg_detections"] = summary.avg_detections
            row["detections_std"] = summary.detections_std
            row["avg_confidence"] = summary.avg_confidence
            row["macro_precision"] = summary.macro_precision
            row["macro_recall"] = summary.macro_recall
            row["macro_f1"] = summary.macro_f1
            row["micro_precision"] = summary.micro_precision
            row["micro_recall"] = summary.micro_recall
            row["micro_f1"] = summary.micro_f1
            row["f1"] = summary.macro_f1
        else:
            row["avg_pixel_change_ratio"] = summary.avg_pixel_change_ratio
            row["avg_edge_pixels"] = summary.avg_edge_pixels
        rows.append(row)

    if sort_by:
        rows.sort(
            key=lambda item: item.get(sort_by) if item.get(sort_by) is not None else float("-inf"),
            reverse=descending,
        )

    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def benchmark_per_image_rows(benchmark_result):
    rows = []
    for summary in benchmark_result.summaries:
        for image_result in summary.per_image:
            row = {
                "model": summary.pipeline_name,
                "file_name": image_result.file_name,
                "comparison_group": image_result.comparison_group,
                "metric_template": image_result.metric_template,
                "latency_ms": image_result.metrics.get("latency_ms"),
            }
            if image_result.metric_template == "detection":
                row["detections"] = image_result.metrics.get("detections")
                row["avg_confidence"] = image_result.metrics.get("avg_confidence")
                if image_result.evaluation:
                    row["precision"] = image_result.evaluation.get("precision")
                    row["recall"] = image_result.evaluation.get("recall")
                    row["f1"] = image_result.evaluation.get("f1")
            else:
                row["pixel_change_ratio"] = image_result.metrics.get("pixel_change_ratio")
                row["edge_pixels"] = image_result.metrics.get("edge_pixels")
                row["output_channels"] = image_result.metrics.get("output_channels")
            rows.append(row)
    return rows


def benchmark_to_export_dict(benchmark_result):
    return {
        "session_id": benchmark_result.session_id,
        "created_at": benchmark_result.created_at,
        "comparison_group": benchmark_result.comparison_group,
        "has_ground_truth": benchmark_result.has_ground_truth,
        "warmup_applied": benchmark_result.warmup_applied,
        "ground_truth_report": benchmark_result.ground_truth_report,
        "pipeline_ids": benchmark_result.pipeline_ids,
        "params_map": benchmark_result.params_map,
        "leaderboard": benchmark_leaderboard_rows(benchmark_result),
        "per_image": benchmark_per_image_rows(benchmark_result),
        "summaries": [
            {
                "pipeline_id": summary.pipeline_id,
                "pipeline_name": summary.pipeline_name,
                "comparison_group": summary.comparison_group,
                "metric_template": summary.metric_template,
                "image_count": summary.image_count,
                "avg_latency_ms": summary.avg_latency_ms,
                "latency_std_ms": summary.latency_std_ms,
                "detections_std": summary.detections_std,
                "avg_detections": summary.avg_detections,
                "avg_confidence": summary.avg_confidence,
                "macro_precision": summary.macro_precision,
                "macro_recall": summary.macro_recall,
                "macro_f1": summary.macro_f1,
                "micro_precision": summary.micro_precision,
                "micro_recall": summary.micro_recall,
                "micro_f1": summary.micro_f1,
                "avg_pixel_change_ratio": summary.avg_pixel_change_ratio,
                "avg_edge_pixels": summary.avg_edge_pixels,
            }
            for summary in benchmark_result.summaries
        ],
    }
