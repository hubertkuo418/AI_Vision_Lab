import numpy as np

from src.benchmark_runner import (
    BenchmarkOptions,
    benchmark_leaderboard_rows,
    run_benchmark,
)
from src.ground_truth import (
    aggregate_evaluations,
    build_ground_truth_report,
    evaluate_predictions,
)


def test_aggregate_evaluations_micro_scores():
    evaluations = [
        {
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        {
            "true_positives": 0,
            "false_positives": 1,
            "false_negatives": 1,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
    ]
    micro = aggregate_evaluations(evaluations)
    assert micro["true_positives"] == 1
    assert micro["false_positives"] == 1
    assert micro["false_negatives"] == 1
    assert micro["precision"] == 0.5
    assert micro["recall"] == 0.5
    assert micro["f1"] == 0.5


def test_build_ground_truth_report_missing_files():
    gt_data = {"other.jpg": [{"label": "face", "x": 0, "y": 0, "width": 10, "height": 10}]}
    report = build_ground_truth_report(
        ["photo.jpg"],
        gt_data,
        require_detection_gt=True,
    )
    assert report["matched_count"] == 0
    assert report["missing_gt_images"] == ["photo.jpg"]
    assert "other.jpg" in report["unused_gt_keys"]


def test_run_benchmark_with_warmup_flag():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    images = {"sample.png": image}
    result = run_benchmark(
        images,
        ["gray"],
        options=BenchmarkOptions(warmup=False),
        comparison_group="color_transform",
    )
    assert result.warmup_applied is False
    assert result.summaries[0].metric_template == "processing"
    rows = benchmark_leaderboard_rows(result, sort_by="avg_latency_ms")
    assert rows[0]["rank"] == 1
    assert "avg_pixel_change_ratio" in rows[0]


def test_leaderboard_includes_detection_metrics():
    predictions = [
        {
            "label": "face",
            "bbox": {"x": 5, "y": 5, "width": 10, "height": 10},
        }
    ]
    ground_truth = [{"label": "face", "x": 6, "y": 6, "width": 10, "height": 10}]
    evaluation = evaluate_predictions(predictions, ground_truth, iou_threshold=0.3)
    assert evaluation["f1"] > 0


def test_benchmark_warmup_uses_needs_warmup_flag():
    from src.benchmark_runner import _warmup_pipelines
    from src.comparison_runner import default_params_map

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    images = {"sample.png": image}
    pipeline_ids = ["object_detection_yolov8n", "gray"]
    params_map = default_params_map(pipeline_ids)

    warmed = _warmup_pipelines(images, pipeline_ids, params_map, BenchmarkOptions(warmup=True))
    assert warmed is True


def test_yolo_variant_registry_metadata():
    from src.model_registry import get_pipeline

    pipeline = get_pipeline("object_detection_yolov8s")
    assert pipeline.model_version == "s"
    assert pipeline.weights_path == "yolov8s.pt"
    assert pipeline.needs_warmup is True
