from src.model_registry import (
    PIPELINE_BY_ID,
    get_pipeline,
    list_comparable_pipelines,
)


def test_face_detection_pipeline_count():
    pipelines = list_comparable_pipelines("face_detection")
    ids = {pipeline.id for pipeline in pipelines}
    assert "face_detection_haar" in ids
    assert "face_detection_dnn" in ids
    assert "face_detection_yunet" in ids
    assert len(ids) >= 3


def test_only_ready_excludes_missing_yunet_when_absent():
    from src.model_paths import weights_available

    ready = list_comparable_pipelines("face_detection", only_ready=True)
    ready_ids = {pipeline.id for pipeline in ready}
    if not weights_available("models/face_detection_yunet_2023mar.onnx"):
        assert "face_detection_yunet" not in ready_ids
    assert "face_detection_haar" in ready_ids


def test_yolov8_pipeline_variants():
    ids = {pipeline.id for pipeline in list_comparable_pipelines("object_detection")}
    assert "object_detection_yolov8n" in ids
    assert "object_detection_yolov8s" in ids
    assert "object_detection_yolov8m" in ids


def test_legacy_yolo_alias_points_to_yolov8n():
    legacy = get_pipeline("object_detection_yolo")
    current = get_pipeline("object_detection_yolov8n")
    assert legacy.weights_path == current.weights_path
    assert legacy.runner == current.runner


def test_yolo_pipelines_need_warmup():
    for pipeline_id in (
        "object_detection_yolov8n",
        "object_detection_yolov8s",
        "object_detection_yolov8m",
        "object_detection_yolo",
    ):
        assert get_pipeline(pipeline_id).needs_warmup is True


def test_dnn_weights_path_matches_actual_file():
    pipeline = get_pipeline("face_detection_dnn")
    assert pipeline.weights_path == "models/res10_300x300_ssd_iter_140000.caffemodel"


def test_pipeline_by_id_includes_alias():
    assert "object_detection_yolo" in PIPELINE_BY_ID
