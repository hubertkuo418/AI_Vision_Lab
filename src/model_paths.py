from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def resolve_weights_path(weights_path):
    if not weights_path:
        return None
    path = Path(weights_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def weights_available(weights_path):
    resolved = resolve_weights_path(weights_path)
    return resolved is not None and resolved.exists()


def pipeline_status(weights_path):
    if weights_path is None:
        return "ready"
    return "ready" if weights_available(weights_path) else "missing_weights"
