from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from uuid import uuid4

from src.metrics_utils import DETECTION_GROUPS
from src.metrics_utils import enrich_pipeline_metrics
from src.model_registry import default_params, get_pipeline, list_comparable_pipelines, run_pipeline


MAX_PIPELINES_PER_COMPARE = 4


@dataclass
class PipelineComparisonEntry:
    pipeline_id: str
    pipeline_name: str
    comparison_group: str
    task_type: str
    params: dict
    result_image: object
    export_data: dict
    metrics: dict
    messages: list[str]
    latency_ms: float


@dataclass
class ComparisonResult:
    session_id: str
    file_name: str
    comparison_group: str
    created_at: str
    entries: list[PipelineComparisonEntry] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def default_params_map(pipeline_ids):
    return {pipeline_id: default_params(pipeline_id) for pipeline_id in pipeline_ids}


def build_comparison_summary(entries):
    if not entries:
        return {}

    detection_entries = [
        entry for entry in entries if entry.comparison_group in DETECTION_GROUPS
    ]
    processing_entries = [
        entry for entry in entries if entry.comparison_group not in DETECTION_GROUPS
    ]

    fastest = min(entries, key=lambda item: item.latency_ms)
    slowest = max(entries, key=lambda item: item.latency_ms)

    summary = {
        "pipeline_count": len(entries),
        "fastest_pipeline": fastest.pipeline_name,
        "fastest_latency_ms": fastest.latency_ms,
        "slowest_pipeline": slowest.pipeline_name,
        "slowest_latency_ms": slowest.latency_ms,
        "latency_spread_ms": round(slowest.latency_ms - fastest.latency_ms, 2),
    }

    if detection_entries:
        most_detections = max(
            detection_entries,
            key=lambda item: item.metrics.get("detections", 0),
        )
        least_detections = min(
            detection_entries,
            key=lambda item: item.metrics.get("detections", 0),
        )
        summary["most_detections_pipeline"] = most_detections.pipeline_name
        summary["most_detections"] = most_detections.metrics.get("detections", 0)
        summary["least_detections_pipeline"] = least_detections.pipeline_name
        summary["least_detections"] = least_detections.metrics.get("detections", 0)

    if processing_entries:
        summary["processing_pipelines"] = [
            entry.pipeline_name for entry in processing_entries
        ]

    return summary


def run_comparison(
    image,
    pipeline_ids,
    *,
    file_name="upload",
    params_map=None,
    comparison_group=None,
):
    if not pipeline_ids:
        raise ValueError("Select at least one pipeline to compare.")

    if len(pipeline_ids) > MAX_PIPELINES_PER_COMPARE:
        raise ValueError(
            f"Compare up to {MAX_PIPELINES_PER_COMPARE} pipelines at once."
        )

    params_map = params_map or default_params_map(pipeline_ids)
    entries = []
    groups_seen = []

    for pipeline_id in pipeline_ids:
        pipeline = get_pipeline(pipeline_id)
        params = params_map.get(pipeline_id, default_params(pipeline_id))
        groups_seen.append(pipeline.comparison_group)

        started = perf_counter()
        result = run_pipeline(pipeline_id, image.copy(), params)
        latency_ms = (perf_counter() - started) * 1000
        enrich_pipeline_metrics(result, image, latency_ms, original=image)
        export_data = result.to_export_dict(pipeline, params, image.shape)

        entries.append(
            PipelineComparisonEntry(
                pipeline_id=pipeline.id,
                pipeline_name=pipeline.name,
                comparison_group=pipeline.comparison_group,
                task_type=pipeline.task_type,
                params=params,
                result_image=result.image,
                export_data=export_data,
                metrics=result.metrics,
                messages=result.messages,
                latency_ms=latency_ms,
            )
        )

    resolved_group = comparison_group or (
        groups_seen[0] if len(set(groups_seen)) == 1 else "mixed"
    )

    return ComparisonResult(
        session_id=uuid4().hex,
        file_name=file_name,
        comparison_group=resolved_group,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        entries=entries,
        summary=build_comparison_summary(entries),
    )
