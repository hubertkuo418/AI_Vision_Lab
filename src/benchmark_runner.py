from dataclasses import dataclass, field
from statistics import mean, pstdev
from time import perf_counter

from src.comparison_runner import MAX_PIPELINES_PER_COMPARE, default_params_map
from src.ground_truth import evaluate_predictions, parse_ground_truth_payload
from src.metrics_utils import DETECTION_GROUPS, enrich_pipeline_metrics
from src.model_registry import get_pipeline, run_pipeline


@dataclass
class ImageBenchmarkResult:
    file_name: str
    pipeline_id: str
    pipeline_name: str
    comparison_group: str
    metrics: dict
    export_data: dict
    evaluation: dict | None = None


@dataclass
class PipelineBenchmarkSummary:
    pipeline_id: str
    pipeline_name: str
    comparison_group: str
    image_count: int
    avg_latency_ms: float
    latency_std_ms: float
    avg_detections: float | None = None
    avg_precision: float | None = None
    avg_recall: float | None = None
    avg_f1: float | None = None
    per_image: list[ImageBenchmarkResult] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    created_at: str
    comparison_group: str
    has_ground_truth: bool
    summaries: list[PipelineBenchmarkSummary] = field(default_factory=list)


def _resolve_ground_truth(ground_truth, file_name):
    if ground_truth is None:
        return []

    if file_name in ground_truth:
        return ground_truth[file_name]
    if "default" in ground_truth:
        return ground_truth["default"]
    return []


def run_benchmark(
    images,
    pipeline_ids,
    *,
    params_map=None,
    ground_truth=None,
    iou_threshold=0.5,
    comparison_group=None,
):
    if not images:
        raise ValueError("Upload at least one image for benchmarking.")

    if len(pipeline_ids) > MAX_PIPELINES_PER_COMPARE:
        raise ValueError(
            f"Benchmark up to {MAX_PIPELINES_PER_COMPARE} pipelines at once."
        )

    params_map = params_map or default_params_map(pipeline_ids)
    gt_data = parse_ground_truth_payload(ground_truth) if ground_truth else None
    per_pipeline = {pipeline_id: [] for pipeline_id in pipeline_ids}

    for file_name, image in images.items():
        for pipeline_id in pipeline_ids:
            pipeline = get_pipeline(pipeline_id)
            params = params_map.get(pipeline_id, {})

            started = perf_counter()
            result = run_pipeline(pipeline_id, image.copy(), params)
            latency_ms = (perf_counter() - started) * 1000
            enrich_pipeline_metrics(result, image, latency_ms, original=image)
            export_data = result.to_export_dict(pipeline, params, image.shape)

            evaluation = None
            if gt_data is not None and pipeline.comparison_group in DETECTION_GROUPS:
                gt_boxes = _resolve_ground_truth(gt_data, file_name)
                if gt_boxes:
                    evaluation = evaluate_predictions(
                        export_data.get("annotations", []),
                        gt_boxes,
                        iou_threshold=iou_threshold,
                    )

            per_pipeline[pipeline_id].append(
                ImageBenchmarkResult(
                    file_name=file_name,
                    pipeline_id=pipeline.id,
                    pipeline_name=pipeline.name,
                    comparison_group=pipeline.comparison_group,
                    metrics=result.metrics,
                    export_data=export_data,
                    evaluation=evaluation,
                )
            )

    summaries = []
    groups = []

    for pipeline_id in pipeline_ids:
        pipeline = get_pipeline(pipeline_id)
        rows = per_pipeline[pipeline_id]
        groups.append(pipeline.comparison_group)
        latencies = [row.metrics["latency_ms"] for row in rows]

        summary = PipelineBenchmarkSummary(
            pipeline_id=pipeline.id,
            pipeline_name=pipeline.name,
            comparison_group=pipeline.comparison_group,
            image_count=len(rows),
            avg_latency_ms=round(mean(latencies), 2),
            latency_std_ms=round(pstdev(latencies), 2) if len(latencies) > 1 else 0.0,
            per_image=rows,
        )

        if pipeline.comparison_group in DETECTION_GROUPS:
            summary.avg_detections = round(
                mean(row.metrics.get("detections", 0) for row in rows),
                2,
            )
            evaluated = [row.evaluation for row in rows if row.evaluation]
            if evaluated:
                summary.avg_precision = round(
                    mean(item["precision"] for item in evaluated),
                    4,
                )
                summary.avg_recall = round(
                    mean(item["recall"] for item in evaluated),
                    4,
                )
                summary.avg_f1 = round(mean(item["f1"] for item in evaluated), 4)

        summaries.append(summary)

    resolved_group = comparison_group or (
        groups[0] if len(set(groups)) == 1 else "mixed"
    )

    from datetime import datetime

    return BenchmarkResult(
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        comparison_group=resolved_group,
        has_ground_truth=gt_data is not None,
        summaries=summaries,
    )


def benchmark_leaderboard_rows(benchmark_result):
    rows = []
    for summary in benchmark_result.summaries:
        row = {
            "model": summary.pipeline_name,
            "comparison_group": summary.comparison_group,
            "images": summary.image_count,
            "avg_latency_ms": summary.avg_latency_ms,
            "latency_std_ms": summary.latency_std_ms,
        }
        if summary.avg_detections is not None:
            row["avg_detections"] = summary.avg_detections
        if summary.avg_precision is not None:
            row["precision"] = summary.avg_precision
            row["recall"] = summary.avg_recall
            row["f1"] = summary.avg_f1
        rows.append(row)
    return rows
