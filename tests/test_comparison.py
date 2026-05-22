import numpy as np

from src.benchmark_runner import benchmark_leaderboard_rows, run_benchmark
from src.comparison_runner import build_comparison_summary, run_comparison
from src.ground_truth import evaluate_predictions
from src.model_registry import list_comparable_pipelines


def _sample_image():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[12:36, 20:44] = 255
    return image


def test_run_comparison_face_detection_group():
    image = _sample_image()
    pipeline_ids = [
        pipeline.id
        for pipeline in list_comparable_pipelines("face_detection", only_ready=True)
    ]
    assert pipeline_ids

    result = run_comparison(
        image,
        pipeline_ids,
        file_name="sample.png",
        comparison_group="face_detection",
    )

    assert len(result.entries) == len(pipeline_ids)
    assert result.summary["pipeline_count"] == len(pipeline_ids)
    for entry in result.entries:
        assert entry.metrics["latency_ms"] >= 0
        assert entry.metrics["input_width"] == 64


def test_build_comparison_summary_latency_spread():
    from src.comparison_runner import PipelineComparisonEntry

    entries = [
        PipelineComparisonEntry(
            pipeline_id="a",
            pipeline_name="A",
            comparison_group="face_detection",
            task_type="Face Detection",
            params={},
            result_image=np.zeros((10, 10, 3), dtype=np.uint8),
            export_data={},
            metrics={"detections": 1, "latency_ms": 10},
            messages=[],
            latency_ms=10,
        ),
        PipelineComparisonEntry(
            pipeline_id="b",
            pipeline_name="B",
            comparison_group="face_detection",
            task_type="Face Detection",
            params={},
            result_image=np.zeros((10, 10, 3), dtype=np.uint8),
            export_data={},
            metrics={"detections": 2, "latency_ms": 30},
            messages=[],
            latency_ms=30,
        ),
    ]

    summary = build_comparison_summary(entries)
    assert summary["fastest_pipeline"] == "A"
    assert summary["latency_spread_ms"] == 20


def test_ground_truth_evaluation():
    predictions = [
        {
            "label": "face",
            "bbox": {"x": 10, "y": 10, "width": 20, "height": 20, "x2": 30, "y2": 30},
        }
    ]
    ground_truth = [{"label": "face", "x": 12, "y": 12, "width": 18, "height": 18}]
    scores = evaluate_predictions(predictions, ground_truth, iou_threshold=0.3)
    assert scores["true_positives"] == 1
    assert scores["f1"] > 0


def test_run_benchmark_without_ground_truth():
    images = {"sample.png": _sample_image()}
    pipeline_ids = ["gray", "face_detection_haar"]

    result = run_benchmark(images, pipeline_ids, comparison_group="mixed")
    rows = benchmark_leaderboard_rows(result)

    assert len(result.summaries) == 2
    assert len(rows) == 2
    assert rows[0]["avg_latency_ms"] >= 0
