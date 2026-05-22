from types import SimpleNamespace

import numpy as np

from src import history_store


def _sample_image():
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    image[3:9, 4:12] = 255
    return image


def _configure_temp_history(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    monkeypatch.setattr(history_store, "DATA_DIR", data_dir)
    monkeypatch.setattr(history_store, "IMAGE_DIR", data_dir / "history_images")
    monkeypatch.setattr(history_store, "DB_PATH", data_dir / "vision_history.sqlite")


def _save_comparison_session():
    image = _sample_image()
    comparison_result = SimpleNamespace(
        session_id="compare-1",
        created_at="2026-05-22 10:00:00",
        summary={"pipeline_count": 1},
        entries=[
            SimpleNamespace(
                pipeline_id="gray",
                pipeline_name="Gray",
                comparison_group="color_transform",
                task_type="Processing",
                params={},
                metrics={"latency_ms": 1.2},
                messages=[],
                export_data={},
                result_image=image,
            )
        ],
    )
    return history_store.save_comparison_session(
        file_name="sample.png",
        comparison_group="color_transform",
        original_image=image,
        comparison_result=comparison_result,
    )


def test_clear_analysis_runs_keeps_comparison_images(monkeypatch, tmp_path):
    _configure_temp_history(monkeypatch, tmp_path)
    image = _sample_image()
    pipeline = SimpleNamespace(id="gray", name="Gray")

    history_store.save_analysis_run(
        file_name="sample.png",
        category="color_transform",
        pipeline=pipeline,
        params={},
        original_image=image,
        result_image=image,
        summary={"ok": True},
        export_data={},
    )
    comparison = _save_comparison_session()
    comparison_paths = [
        comparison["original_image_path"],
        comparison["runs"][0]["result_image_path"],
    ]

    history_store.clear_analysis_runs()

    assert history_store.list_analysis_runs() == []
    assert all(history_store.Path(path).exists() for path in comparison_paths)
    assert len(history_store.list_comparison_sessions()) == 1


def test_missing_comparison_image_does_not_crash_history_load(monkeypatch, tmp_path):
    _configure_temp_history(monkeypatch, tmp_path)
    comparison = _save_comparison_session()
    history_store.Path(comparison["runs"][0]["result_image_path"]).unlink()

    assert history_store.list_comparison_sessions() == []
    assert history_store.load_comparison_session(comparison["id"]) is None
