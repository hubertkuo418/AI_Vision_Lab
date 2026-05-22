import json
import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "history_images"
DB_PATH = DATA_DIR / "vision_history.sqlite"


def init_history_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                file_name TEXT NOT NULL,
                category TEXT NOT NULL,
                pipeline_id TEXT NOT NULL,
                pipeline_name TEXT NOT NULL,
                params_json TEXT NOT NULL,
                annotations_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                original_image_path TEXT NOT NULL,
                result_image_path TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_analysis_runs_created_at "
            "ON analysis_runs(created_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS comparison_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                file_name TEXT NOT NULL,
                comparison_group TEXT NOT NULL,
                original_image_path TEXT NOT NULL,
                runs_json TEXT NOT NULL,
                summary_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_comparison_sessions_created_at "
            "ON comparison_sessions(created_at DESC)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                comparison_group TEXT NOT NULL,
                image_count INTEGER NOT NULL,
                pipeline_ids_json TEXT NOT NULL,
                has_ground_truth INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                leaderboard_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_benchmark_sessions_created_at "
            "ON benchmark_sessions(created_at DESC)"
        )


def _save_image(image_array, prefix):
    image_id = uuid4().hex
    image_path = IMAGE_DIR / f"{prefix}_{image_id}.png"
    array = np.asarray(image_array)

    if array.ndim == 2:
        image = Image.fromarray(array)
    else:
        image = Image.fromarray(array.astype("uint8")).convert("RGB")

    image.save(image_path)
    return image_path


def _load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)


def _load_entry_or_none(row, converter):
    try:
        return converter(row)
    except FileNotFoundError:
        return None


def _existing_entries(rows, converter):
    entries = []
    for row in rows:
        entry = _load_entry_or_none(row, converter)
        if entry is not None:
            entries.append(entry)
    return entries


def _delete_image_paths(paths):
    for image_path in paths:
        Path(image_path).unlink(missing_ok=True)


def save_analysis_run(
    *,
    file_name,
    category,
    pipeline,
    params,
    original_image,
    result_image,
    summary,
    export_data,
):
    init_history_db()
    run_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    original_path = _save_image(original_image, "original")
    result_path = _save_image(result_image, "result")

    export_with_paths = {
        **export_data,
        "artifacts": {
            "original_image_path": str(original_path),
            "result_image_path": str(result_path),
        },
    }

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs (
                id,
                created_at,
                file_name,
                category,
                pipeline_id,
                pipeline_name,
                params_json,
                annotations_json,
                result_json,
                original_image_path,
                result_image_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                created_at,
                file_name,
                category,
                pipeline.id,
                pipeline.name,
                json.dumps(params),
                json.dumps(export_data.get("annotations", [])),
                json.dumps(
                    {
                        "summary": summary,
                        "export": export_with_paths,
                    }
                ),
                str(original_path),
                str(result_path),
            ),
        )

    return load_analysis_run(run_id)


def list_analysis_runs(limit=12):
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM analysis_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return _existing_entries(rows, _row_to_entry)


def load_analysis_run(run_id):
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM analysis_runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    return _load_entry_or_none(row, _row_to_entry)


def clear_analysis_runs():
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT original_image_path, result_image_path FROM analysis_runs"
        ).fetchall()
        conn.execute("DELETE FROM analysis_runs")

    _delete_image_paths(
        path
        for row in rows
        for path in (row["original_image_path"], row["result_image_path"])
    )


def save_comparison_session(*, file_name, comparison_group, original_image, comparison_result):
    init_history_db()
    session_id = comparison_result.session_id
    created_at = comparison_result.created_at
    original_path = _save_image(original_image, "compare_original")

    runs_payload = []
    for entry in comparison_result.entries:
        result_path = _save_image(entry.result_image, "compare_result")
        runs_payload.append(
            {
                "pipeline_id": entry.pipeline_id,
                "pipeline_name": entry.pipeline_name,
                "comparison_group": entry.comparison_group,
                "task_type": entry.task_type,
                "params": entry.params,
                "metrics": entry.metrics,
                "messages": entry.messages,
                "export_data": entry.export_data,
                "result_image_path": str(result_path),
            }
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO comparison_sessions (
                id,
                created_at,
                file_name,
                comparison_group,
                original_image_path,
                runs_json,
                summary_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                created_at,
                file_name,
                comparison_group,
                str(original_path),
                json.dumps(runs_payload),
                json.dumps(comparison_result.summary),
            ),
        )

    return load_comparison_session(session_id)


def list_comparison_sessions(limit=12):
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM comparison_sessions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return _existing_entries(rows, _comparison_row_to_entry)


def load_comparison_session(session_id):
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM comparison_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

    if row is None:
        return None

    return _load_entry_or_none(row, _comparison_row_to_entry)


def clear_comparison_sessions():
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT original_image_path, runs_json FROM comparison_sessions"
        ).fetchall()
        conn.execute("DELETE FROM comparison_sessions")

    image_paths = []
    for row in rows:
        image_paths.append(row["original_image_path"])
        image_paths.extend(
            run["result_image_path"]
            for run in json.loads(row["runs_json"])
            if "result_image_path" in run
        )
    _delete_image_paths(image_paths)


def clear_benchmark_sessions():
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM benchmark_sessions")


def clear_all_history():
    clear_analysis_runs()
    clear_comparison_sessions()
    clear_benchmark_sessions()


def _benchmark_result_to_storage(benchmark_result):
    per_image = []
    for summary in benchmark_result.summaries:
        for image_result in summary.per_image:
            per_image.append(
                {
                    "pipeline_id": image_result.pipeline_id,
                    "pipeline_name": image_result.pipeline_name,
                    "file_name": image_result.file_name,
                    "comparison_group": image_result.comparison_group,
                    "metric_template": image_result.metric_template,
                    "metrics": image_result.metrics,
                    "evaluation": image_result.evaluation,
                }
            )

    from src.benchmark_runner import benchmark_leaderboard_rows, benchmark_to_export_dict

    export_payload = benchmark_to_export_dict(benchmark_result)
    export_payload["per_image"] = per_image
    return export_payload


def save_benchmark_session(benchmark_result):
    init_history_db()
    export_payload = _benchmark_result_to_storage(benchmark_result)
    leaderboard = export_payload["leaderboard"]

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO benchmark_sessions (
                id,
                created_at,
                comparison_group,
                image_count,
                pipeline_ids_json,
                has_ground_truth,
                result_json,
                leaderboard_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                benchmark_result.session_id,
                benchmark_result.created_at,
                benchmark_result.comparison_group,
                benchmark_result.summaries[0].image_count
                if benchmark_result.summaries
                else 0,
                json.dumps(benchmark_result.pipeline_ids),
                int(benchmark_result.has_ground_truth),
                json.dumps(export_payload),
                json.dumps(leaderboard),
            ),
        )

    return load_benchmark_session(benchmark_result.session_id)


def list_benchmark_sessions(limit=12):
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT *
            FROM benchmark_sessions
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [_benchmark_row_to_entry(row) for row in rows]


def load_benchmark_session(session_id):
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM benchmark_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

    if row is None:
        return None

    return _benchmark_row_to_entry(row)


def benchmark_session_to_export(entry):
    return entry["export"]


def benchmark_session_to_csv(entry, *, per_image=False):
    import csv
    import io

    buffer = io.StringIO()
    rows = entry["per_image"] if per_image else entry["leaderboard"]
    if not rows:
        return buffer.getvalue()

    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _benchmark_row_to_entry(row):
    export_payload = json.loads(row["result_json"])
    return {
        "id": row["id"],
        "time": row["created_at"],
        "comparison_group": row["comparison_group"],
        "image_count": row["image_count"],
        "pipeline_ids": json.loads(row["pipeline_ids_json"]),
        "has_ground_truth": bool(row["has_ground_truth"]),
        "export": export_payload,
        "leaderboard": json.loads(row["leaderboard_json"]),
        "per_image": export_payload.get("per_image", []),
    }


def restore_benchmark_result(entry):
    return _restore_benchmark_result_from_export(entry["export"])


def _restore_benchmark_result_from_export(export_payload):
    from src.benchmark_runner import BenchmarkResult, ImageBenchmarkResult, PipelineBenchmarkSummary

    summaries = []
    per_image_by_pipeline = {}
    for row in export_payload.get("per_image", []):
        per_image_by_pipeline.setdefault(row["pipeline_id"], []).append(
            ImageBenchmarkResult(
                file_name=row["file_name"],
                pipeline_id=row["pipeline_id"],
                pipeline_name=row["pipeline_name"],
                comparison_group=row["comparison_group"],
                metric_template=row["metric_template"],
                metrics=row["metrics"],
                evaluation=row.get("evaluation"),
            )
        )

    for summary_data in export_payload.get("summaries", []):
        pipeline_id = summary_data["pipeline_id"]
        summaries.append(
            PipelineBenchmarkSummary(
                pipeline_id=pipeline_id,
                pipeline_name=summary_data["pipeline_name"],
                comparison_group=summary_data["comparison_group"],
                metric_template=summary_data["metric_template"],
                image_count=summary_data["image_count"],
                avg_latency_ms=summary_data["avg_latency_ms"],
                latency_std_ms=summary_data["latency_std_ms"],
                detections_std=summary_data.get("detections_std"),
                avg_detections=summary_data.get("avg_detections"),
                avg_confidence=summary_data.get("avg_confidence"),
                macro_precision=summary_data.get("macro_precision"),
                macro_recall=summary_data.get("macro_recall"),
                macro_f1=summary_data.get("macro_f1"),
                micro_precision=summary_data.get("micro_precision"),
                micro_recall=summary_data.get("micro_recall"),
                micro_f1=summary_data.get("micro_f1"),
                avg_pixel_change_ratio=summary_data.get("avg_pixel_change_ratio"),
                avg_edge_pixels=summary_data.get("avg_edge_pixels"),
                per_image=per_image_by_pipeline.get(pipeline_id, []),
            )
        )

    return BenchmarkResult(
        session_id=export_payload["session_id"],
        created_at=export_payload["created_at"],
        comparison_group=export_payload["comparison_group"],
        has_ground_truth=export_payload["has_ground_truth"],
        warmup_applied=export_payload.get("warmup_applied", False),
        ground_truth_report=export_payload.get("ground_truth_report", {}),
        pipeline_ids=export_payload.get("pipeline_ids", []),
        params_map=export_payload.get("params_map", {}),
        summaries=summaries,
    )


def comparison_session_to_export(entry):
    return {
        "session_id": entry["id"],
        "created_at": entry["time"],
        "file_name": entry["file_name"],
        "comparison_group": entry["comparison_group"],
        "summary": entry["summary"],
        "runs": [
            {
                "pipeline_id": run["pipeline_id"],
                "pipeline_name": run["pipeline_name"],
                "comparison_group": run["comparison_group"],
                "task_type": run["task_type"],
                "params": run["params"],
                "metrics": run["metrics"],
                "messages": run["messages"],
                "export_data": run["export_data"],
            }
            for run in entry["runs"]
        ],
    }


def comparison_session_to_csv(entry):
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "pipeline_name",
            "comparison_group",
            "task_type",
            "latency_ms",
            "detections",
            "avg_confidence",
            "output_channels",
            "pixel_change_ratio",
        ],
    )
    writer.writeheader()
    for run in entry["runs"]:
        metrics = run["metrics"]
        writer.writerow(
            {
                "pipeline_name": run["pipeline_name"],
                "comparison_group": run["comparison_group"],
                "task_type": run["task_type"],
                "latency_ms": metrics.get("latency_ms"),
                "detections": metrics.get("detections"),
                "avg_confidence": metrics.get("avg_confidence"),
                "output_channels": metrics.get("output_channels"),
                "pixel_change_ratio": metrics.get("pixel_change_ratio"),
            }
        )
    return buffer.getvalue()


def _comparison_row_to_entry(row):
    runs = json.loads(row["runs_json"])
    for run in runs:
        run["result"] = _load_image(run["result_image_path"])

    return {
        "id": row["id"],
        "time": row["created_at"],
        "file_name": row["file_name"],
        "comparison_group": row["comparison_group"],
        "original": _load_image(row["original_image_path"]),
        "runs": runs,
        "summary": json.loads(row["summary_json"]),
        "original_image_path": row["original_image_path"],
    }


def _row_to_entry(row):
    result_payload = json.loads(row["result_json"])
    export_data = result_payload["export"]

    return {
        "id": row["id"],
        "time": row["created_at"],
        "file_name": row["file_name"],
        "group": row["category"],
        "operation": row["pipeline_name"],
        "pipeline_id": row["pipeline_id"],
        "params": json.loads(row["params_json"]),
        "annotations": json.loads(row["annotations_json"]),
        "original": _load_image(row["original_image_path"]),
        "result": _load_image(row["result_image_path"]),
        "summary": result_payload["summary"],
        "export_data": export_data,
        "original_image_path": row["original_image_path"],
        "result_image_path": row["result_image_path"],
    }
