from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL_CFG = ROOT / "ultralytics" / "cfg" / "models" / "11" / "yolo11-mcu.yaml"
DEFAULT_DATA_YAML = ROOT / "datasets" / "my_dataset_yolo" / "my_dataset.yaml"
DEFAULT_RUNS_DIR = ROOT / "runs" / "mcu"
DEFAULT_TRAIN_NAME = "yolo11-mcu"
DEFAULT_VAL_NAME = "val"
DEFAULT_EXPORT_NAME = "export"
DEFAULT_TRAIN_WEIGHTS = DEFAULT_RUNS_DIR / DEFAULT_TRAIN_NAME / "weights" / "best.pt"
DEFAULT_TRAIN_LAST = DEFAULT_RUNS_DIR / DEFAULT_TRAIN_NAME / "weights" / "last.pt"


def require_existing_path(path: str | Path, label: str, hint: str | None = None) -> Path:
    path = Path(path)
    if path.exists():
        return path

    message = f"{label} not found: {path}"
    if hint:
        message = f"{message}\n{hint}"
    raise FileNotFoundError(message)


def dataset_hint() -> str:
    archive = ROOT / "dataset" / "boll.zip"
    return (
        f"Extract {archive} to {ROOT / 'datasets'} so that "
        f"{DEFAULT_DATA_YAML.relative_to(ROOT)} exists."
    )


def resolve_model_source(path: str | Path) -> Path:
    path = Path(path)
    return require_existing_path(
        path,
        "Model source",
        hint=f"Run train.py first or pass a different checkpoint/model file. Expected a trained weight file such as {DEFAULT_TRAIN_WEIGHTS.relative_to(ROOT)}.",
    )

