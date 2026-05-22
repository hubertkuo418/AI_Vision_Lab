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

    return [_row_to_entry(row) for row in rows]


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

    return _row_to_entry(row)


def clear_analysis_runs():
    init_history_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM analysis_runs")

    for image_path in IMAGE_DIR.glob("*.png"):
        image_path.unlink()


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
